from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import httpx
import requests

try:
    import html2text
except Exception:  # pragma: no cover
    html2text = None

import prompts
from config.runtime import get_app_settings, get_model_settings
from memory_module_v2.api import build_memory_context
from utils import logger
from utils.llm_client import call_text_model
from utils.redis_tool import RedisClient
from client.research_engine import run_iterative_research


MAX_HIS = 6
TTL = 45
REDIS_KEY = "voice:chat_history:{}"
_redis_client = RedisClient()

RESEARCH_KEYWORDS = (
    "deep research",
    "deepresearch",
    "调研",
    "深度调研",
    "深度研究",
    "帮我查",
    "查一下官网",
    "查官网",
    "对比",
    "横向比较",
    "给出处",
    "给链接",
    "引用来源",
    "最新信息",
    "最新动态",
    "资料总结",
    "研究一下",
    "分析一下",
)


def should_use_deep_research(query: str, sender_id: str, session_id: str | None = None) -> bool:
    settings = get_app_settings()
    models = get_model_settings()
    if not settings.deep_research_enabled:
        return False

    normalized = (query or "").strip().lower()
    if any(keyword in normalized for keyword in RESEARCH_KEYWORDS):
        return True

    if models.deep_research_trigger_model:
        result = _call_llm(
            models.deep_research_trigger_model,
            prompts.DEEP_RESEARCH_CLASSIFIER_PROMPT.format(query),
            max_tokens=8,
        )
        if result and result[:1] == "是":
            return True

    return False


def request_deep_research(query: str, sender_id: str, session_id: str | None = None) -> dict[str, Any]:
    session_id = session_id or sender_id
    settings = get_app_settings()
    return run_iterative_research(
        query,
        sender_id=sender_id,
        session_id=session_id,
        max_rounds=3,
        max_queries=6,
        max_docs_total=max(settings.deep_research_fetch_top_n, 3) * 2,
        fetch_top_n_per_round=max(1, min(2, settings.deep_research_fetch_top_n)),
        timeout=settings.deep_research_timeout,
    )


def process_research(result: dict[str, Any], query: str, sender_id: str):
    answer = str(result.get("answer", "") or "").strip()
    if not answer:
        return

    fragments = _split_text(answer)
    for item in fragments:
        if item:
            yield item

    history = _redis_client.get(REDIS_KEY.format(sender_id))
    if history:
        history = json.loads(history)
    else:
        history = []
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    history = history[-MAX_HIS:]
    _redis_client.set(REDIS_KEY.format(sender_id), json.dumps(history, ensure_ascii=False), ex=TTL)


def fetch_url_text(url: str) -> str:
    try:
        with httpx.Client(follow_redirects=True, timeout=15) as client:
            response = client.get(url)
            response.raise_for_status()
    except Exception as exc:
        logger.warning(f"fetch url failed: {exc}")
        return ""

    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        try:
            return json.dumps(response.json(), ensure_ascii=False, indent=2)[:5000]
        except Exception:
            return response.text[:5000]
    if "html" in content_type and html2text is not None:
        parser = html2text.HTML2Text()
        parser.ignore_links = False
        parser.ignore_images = True
        return parser.handle(response.text)[:5000]
    return response.text[:5000]


def _search_web(query: str) -> list[dict[str, Any]]:
    settings = get_app_settings()
    if not settings.tavily_api_key:
        return []
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": settings.deep_research_max_results,
    }
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=settings.deep_research_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", []) or []
    except Exception as exc:
        logger.warning(f"tavily search failed: {exc}")
        return []


def _call_llm(model: str, user_prompt: str, *, max_tokens: int = 1200) -> str:
    return call_text_model(
        model,
        user_prompt,
        max_tokens=max_tokens,
        timeout=get_app_settings().deep_research_timeout,
        temperature=0.2,
    )


def _manual_summary(query_time: str, search_results: list[dict[str, Any]], fetched_docs: list[dict[str, Any]]) -> str:
    if not search_results:
        return ""
    parts = ["结论：已完成基础联网检索，但自动总结模型不可用，先返回整理后的候选资料。", "", "依据："]
    for idx, item in enumerate(fetched_docs[:3] or search_results[:3], start=1):
        content = str(item.get("content", "") or "")[:180].replace("\n", " ")
        parts.append(f"{idx}. {item.get('title', '')}：{content}")
    parts.append("")
    parts.append("来源：")
    for item in search_results[:3]:
        parts.append(f"- {item.get('title', '')}: {item.get('url', '')}")
    parts.append("")
    parts.append(f"时间说明：查询时间 {query_time}")
    return "\n".join(parts)


def _split_text(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])", text)
    result = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        buffer += part
        if len(buffer) >= 80 or re.search(r"[。！？；]\s*$", buffer):
            result.append(buffer)
            buffer = ""
    if buffer.strip():
        result.append(buffer)
    return result
