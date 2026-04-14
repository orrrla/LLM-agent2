from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import requests

import prompts
from config.runtime import get_app_settings, get_model_settings
from memory_module_v2.api import build_memory_context
from utils import logger
from utils.llm_client import call_text_model


class SearchExecutor(Protocol):
    def search(self, query: str, *, max_results: int, timeout: float) -> list[dict[str, Any]]: ...


class FetchExecutor(Protocol):
    def fetch(self, url: str, *, timeout: float) -> str: ...

FetchCallable = Callable[..., str]


class BrowseExecutor(Protocol):
    """
    Reserved for future upgrade to a browser agent.
    """

    def browse(self, url: str, *, goal: str, timeout: float) -> dict[str, Any]: ...


@dataclass
class ResearchDoc:
    title: str
    url: str
    content: str
    published_date: str = ""
    score: float = 0.0
    fetched_at: float = field(default_factory=time.time)


@dataclass
class ResearchPlan:
    goal: str
    subquestions: list[str] = field(default_factory=list)
    initial_queries: list[str] = field(default_factory=list)
    stop_criteria: list[str] = field(default_factory=list)


@dataclass
class CritiqueResult:
    action: str  # continue | finalize
    reason: str = ""
    new_queries: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass
class ResearchState:
    query: str
    query_time: str
    sender_id: str
    session_id: str
    plan: ResearchPlan | None = None
    executed_queries: list[str] = field(default_factory=list)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    docs: list[ResearchDoc] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""


class TavilySearch:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, *, max_results: int, timeout: float) -> list[dict[str, Any]]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "max_results": max_results,
        }
        try:
            resp = requests.post("https://api.tavily.com/search", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", []) or []
        except Exception as exc:
            logger.warning(f"tavily search failed: {exc}")
            return []


def run_iterative_research(
    query: str,
    *,
    sender_id: str,
    session_id: str,
    max_rounds: int = 3,
    max_queries: int = 6,
    max_docs_total: int = 6,
    fetch_top_n_per_round: int = 2,
    timeout: float = 18.0,
    search_executor: SearchExecutor | None = None,
    fetch_executor: FetchExecutor | FetchCallable | None = None,
    browse_executor: BrowseExecutor | None = None,
) -> dict[str, Any]:
    settings = get_app_settings()
    query_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = ResearchState(
        query=query,
        query_time=query_time,
        sender_id=sender_id,
        session_id=session_id,
    )

    if search_executor is None:
        if not settings.tavily_api_key:
            return _fallback_to_memory_or_empty(state, timeout=timeout)
        search_executor = TavilySearch(settings.tavily_api_key)

    if fetch_executor is None:
        # defer import to keep engine lean
        from client.deep_research import fetch_url_text as _fetch_url_text  # type: ignore

        fetch_executor = _fetch_url_text

    # Reserved for future: browse executor can enrich evidence beyond plain fetch.
    # Current iterative research does not invoke it yet, but the interface is now stable.
    _ = browse_executor

    # Plan (LLM), fallback to a minimal plan when model is unavailable.
    plan = _build_plan(state, timeout=timeout)
    state.plan = plan

    # record stop criteria for transparency in state; enforcement is in loop guards
    state.observations.append(
        {
            "round": 0,
            "query": "",
            "critique": {
                "action": "continue",
                "reason": "init_plan",
                "new_queries": plan.initial_queries,
                "conflicts": [],
            },
            "stop_criteria": plan.stop_criteria,
        }
    )

    pending_queries = _dedupe_queries([*plan.initial_queries] or [query], cap=max_queries)

    round_index = 0
    while pending_queries and round_index < max_rounds and len(state.docs) < max_docs_total:
        round_index += 1
        current = pending_queries.pop(0)
        if current in state.executed_queries:
            continue
        state.executed_queries.append(current)

        results = search_executor.search(current, max_results=settings.deep_research_max_results, timeout=timeout)
        if results:
            state.search_results.extend(results)

        # Fetch a small set of URLs per round.
        fetched = _fetch_docs(results, fetch_executor, top_n=fetch_top_n_per_round, timeout=timeout)
        state.docs.extend(_dedupe_docs(fetched, cap=max_docs_total - len(state.docs)))

        # Observe + critique to decide next action
        if _has_enough_sources(state, min_domains=2, min_docs=3):
            state.stop_reason = "enough_sources"
            break
        critique = _critique(state, timeout=timeout)
        state.observations.append({"round": round_index, "query": current, "critique": asdict(critique)})

        if critique.conflicts:
            # keep conflicts in state for final prompt
            state.observations[-1]["conflicts"] = critique.conflicts

        if critique.action == "finalize":
            state.stop_reason = critique.reason or "enough_evidence"
            break

        # Add refined queries
        for q in critique.new_queries:
            if len(state.executed_queries) + len(pending_queries) >= max_queries:
                break
            if q and q not in state.executed_queries and q not in pending_queries:
                pending_queries.append(q)

        # If model failed or no new query, stop to avoid useless loops
        if not critique.new_queries:
            state.stop_reason = critique.reason or "no_new_queries"
            break

    if not state.stop_reason:
        state.stop_reason = "budget_exhausted"

    answer = _finalize(state, timeout=timeout)
    if answer:
        sources = _build_sources(state)
        return {
            "mode": "research",
            "answer": answer.strip(),
            "sources": sources,
            "query_time": state.query_time,
        }

    # If finalize failed, fall back to old manual summary style using available artifacts.
    manual = _manual_summary_like(state)
    if manual:
        return {
            "mode": "research",
            "answer": manual,
            "sources": _build_sources(state),
            "query_time": state.query_time,
        }

    return _fallback_to_memory_or_empty(state, timeout=timeout)


def _build_plan(state: ResearchState, *, timeout: float) -> ResearchPlan:
    models = get_model_settings()
    prompt = prompts.DEEP_RESEARCH_PLAN_PROMPT.format(state.query)
    text = call_text_model(models.deep_research_model, prompt, max_tokens=800, timeout=timeout, temperature=0.2)
    parsed = _parse_json(text)
    if parsed:
        return ResearchPlan(
            goal=str(parsed.get("goal") or "完成联网调研并给出结论").strip(),
            subquestions=[str(x).strip() for x in (parsed.get("subquestions") or []) if str(x).strip()],
            initial_queries=[str(x).strip() for x in (parsed.get("initial_queries") or []) if str(x).strip()],
            stop_criteria=[str(x).strip() for x in (parsed.get("stop_criteria") or []) if str(x).strip()],
        )
    # Fallback minimal plan
    return ResearchPlan(
        goal="完成联网调研并给出结论",
        subquestions=[],
        initial_queries=[state.query, f"{state.query} 官网", f"{state.query} 对比"],
        stop_criteria=["至少覆盖2个不同来源", "达到最大轮次/预算"],
    )


def _critique(state: ResearchState, *, timeout: float) -> CritiqueResult:
    models = get_model_settings()
    payload = {
        "query_time": state.query_time,
        "executed_queries": state.executed_queries[-3:],
        "docs": [asdict(doc) for doc in state.docs[-4:]],
        "stop_reason": state.stop_reason,
    }
    prompt = prompts.DEEP_RESEARCH_CRITIQUE_PROMPT.format(state.query, json.dumps(payload, ensure_ascii=False, indent=2))
    text = call_text_model(models.deep_research_model, prompt, max_tokens=500, timeout=timeout, temperature=0.2)
    parsed = _parse_json(text)
    if not parsed:
        return CritiqueResult(action="finalize", reason="critique_model_unavailable", new_queries=[])
    action = str(parsed.get("action") or "").strip().lower()
    if action not in {"continue", "finalize"}:
        action = "finalize"
    new_queries = [str(x).strip() for x in (parsed.get("new_queries") or []) if str(x).strip()]
    conflicts = [str(x).strip() for x in (parsed.get("conflicts") or []) if str(x).strip()]
    return CritiqueResult(action=action, reason=str(parsed.get("reason") or "").strip(), new_queries=new_queries[:4], conflicts=conflicts[:4])


def _finalize(state: ResearchState, *, timeout: float) -> str:
    models = get_model_settings()
    research_payload = {
        "query_time": state.query_time,
        "plan": asdict(state.plan) if state.plan else {},
        "executed_queries": state.executed_queries,
        "search_results": state.search_results[:15],
        "fetched_docs": [asdict(doc) for doc in state.docs[:8]],
        "observations": state.observations[-6:],
        "stop_reason": state.stop_reason,
    }
    prompt = prompts.DEEP_RESEARCH_FINAL_PROMPT.format(
        state.query,
        json.dumps(research_payload, ensure_ascii=False, indent=2),
    )
    return call_text_model(models.deep_research_model, prompt, max_tokens=1800, timeout=timeout, temperature=0.2).strip()


def _fetch_docs(
    results: list[dict[str, Any]],
    fetch_executor: FetchExecutor | FetchCallable,
    *,
    top_n: int,
    timeout: float,
) -> list[ResearchDoc]:
    docs: list[ResearchDoc] = []
    for item in results[: max(0, top_n)]:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        content = ""
        try:
            if hasattr(fetch_executor, "fetch"):
                content = fetch_executor.fetch(url, timeout=timeout)  # type: ignore[union-attr]
            else:
                # fetch_url_text signature doesn't accept timeout
                content = fetch_executor(url)  # type: ignore[misc]
        except Exception as exc:
            logger.warning(f"fetch executor failed: {exc}")
            content = ""
        if not content:
            continue
        docs.append(
            ResearchDoc(
                title=str(item.get("title", "") or ""),
                url=url,
                content=content[:5000],
                published_date=str(item.get("published_date", "") or ""),
                score=float(item.get("score", 0.0) or 0.0),
            )
        )
    return docs


def _build_sources(state: ResearchState) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for doc in state.docs:
        url = doc.url.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({"title": doc.title, "url": url})
    if sources:
        return sources[:10]

    # fallback to raw search results
    for item in state.search_results:
        url = str(item.get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({"title": str(item.get("title", "") or ""), "url": url})
    return sources[:10]


def _dedupe_docs(docs: list[ResearchDoc], *, cap: int) -> list[ResearchDoc]:
    result: list[ResearchDoc] = []
    seen: set[str] = set()
    for doc in docs:
        if len(result) >= cap:
            break
        key = _norm_url(doc.url)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def _dedupe_queries(items: list[str], *, cap: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for q in items:
        qn = " ".join(str(q).split()).strip()
        if not qn or qn in seen:
            continue
        seen.add(qn)
        out.append(qn)
        if len(out) >= cap:
            break
    return out


def _parse_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.splitlines() if not line.startswith("```")).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _manual_summary_like(state: ResearchState) -> str:
    if not state.search_results and not state.docs:
        return ""
    parts = ["结论：已完成基础联网检索，但自动总结模型不可用，先返回整理后的候选资料。", "", "依据："]
    for idx, doc in enumerate(state.docs[:3] or [], start=1):
        content = (doc.content or "")[:180].replace("\n", " ")
        parts.append(f"{idx}. {doc.title}：{content}")
    if not state.docs:
        for idx, item in enumerate(state.search_results[:3], start=1):
            content = str(item.get('content', '') or '')[:180].replace('\\n', ' ')
            parts.append(f"{idx}. {item.get('title', '')}：{content}")
    parts.append("")
    parts.append("来源：")
    for item in _build_sources(state)[:5]:
        parts.append(f"- {item.get('title', '')}: {item.get('url', '')}")
    parts.append("")
    parts.append(f"时间说明：查询时间 {state.query_time}")
    return "\n".join(parts)


def _fallback_to_memory_or_empty(state: ResearchState, *, timeout: float) -> dict[str, Any]:
    memory_context = build_memory_context(state.query, state.session_id)
    if memory_context:
        models = get_model_settings()
        fallback = call_text_model(
            models.deep_research_model,
            prompts.DEEP_RESEARCH_FALLBACK_PROMPT.format(state.query, memory_context),
            max_tokens=800,
            timeout=timeout,
            temperature=0.2,
        )
        return {
            "mode": "memory",
            "answer": (fallback or memory_context).strip(),
            "sources": [],
            "query_time": state.query_time,
        }
    return {"mode": "empty", "answer": "", "sources": [], "query_time": state.query_time}


def _norm_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").rstrip("/")
        return f"{host}{path}"
    except Exception:
        return url


def _has_enough_sources(state: ResearchState, *, min_domains: int, min_docs: int) -> bool:
    if len(state.docs) < min_docs:
        return False
    domains: set[str] = set()
    for doc in state.docs:
        try:
            domains.add((urlparse(doc.url).netloc or "").lower())
        except Exception:
            continue
    domains.discard("")
    return len(domains) >= min_domains

