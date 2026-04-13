from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import prompts
from client.deep_research import request_deep_research
from client.nlu import request_nlu
from config.runtime import get_app_settings, get_model_settings
from memory_module_v2.api import build_memory_context, search_memory
from utils.llm_client import call_text_model


PLANNER_KEYWORDS = (
    "先",
    "然后",
    "再",
    "接着",
    "顺便",
    "同时",
    "并且",
    "另外",
    "规划",
    "安排",
    "一步步",
    "帮我完成",
    "组合",
)


@dataclass
class PlanStep:
    step_id: str
    step_type: str
    goal: str
    query: str = ""
    status: str = "pending"
    result_summary: str = ""


@dataclass
class PlannerResult:
    mode: str
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)


def should_use_planner(query: str, session_id: str | None = None) -> bool:
    settings = get_app_settings()
    models = get_model_settings()
    if not settings.planner_enabled:
        return False

    normalized = (query or "").strip().lower()
    if any(keyword in normalized for keyword in PLANNER_KEYWORDS):
        return True

    if models.planner_gate_model:
        result = _call_llm(
            models.planner_gate_model,
            prompts.PLANNER_GATE_PROMPT.format(query),
            max_tokens=8,
            timeout=settings.planner_timeout,
        )
        if result and result[:1] == "是":
            return True

    return False


def run_planner(query: str, sender_id: str, session_id: str, trace_id: str) -> dict[str, Any]:
    settings = get_app_settings()
    plan_steps = _build_initial_plan(query, session_id)
    observations: list[dict[str, Any]] = []
    replans = 0

    for index in range(min(len(plan_steps), settings.planner_max_steps)):
        step = plan_steps[index]
        if step.status != "pending":
            continue

        if step.step_type == "respond":
            answer = _finalize_answer(query, plan_steps, observations)
            step.status = "success"
            step.result_summary = answer[:300]
            return asdict(PlannerResult(
                mode="planner",
                answer=answer,
                steps=[asdict(item) for item in plan_steps],
                observations=observations,
                sources=_collect_sources(observations),
            ))

        observation = _execute_step(step, sender_id, session_id, trace_id)
        observations.append(observation)
        step.status = observation.get("status", "success")
        step.result_summary = observation.get("summary", "")

        if observation.get("status") == "need_user":
            answer = observation.get("message", "我需要更多信息才能继续完成这个任务。")
            return asdict(PlannerResult(
                mode="planner_need_user",
                answer=answer,
                steps=[asdict(item) for item in plan_steps],
                observations=observations,
                sources=observation.get("sources", []),
            ))
        if replans < settings.planner_max_replans:
            maybe_new_steps = _replan(query, plan_steps, observations)
            if maybe_new_steps:
                completed = [item for item in plan_steps if item.status != "pending"]
                plan_steps = completed + maybe_new_steps
                replans += 1

    answer = _finalize_answer(query, plan_steps, observations)
    return asdict(PlannerResult(
        mode="planner",
        answer=answer,
        steps=[asdict(item) for item in plan_steps],
        observations=observations,
        sources=_collect_sources(observations),
    ))


def process_planner_result(result: dict[str, Any], query: str):
    answer = str(result.get("answer", "") or "").strip()
    if not answer:
        return
    for chunk in _split_text(answer):
        if chunk:
            yield chunk


def _build_initial_plan(query: str, session_id: str) -> list[PlanStep]:
    memory_context = build_memory_context(query, session_id, top_k=2)
    response = _call_llm(
        get_model_settings().planner_model,
        prompts.PLANNER_PLAN_PROMPT.format(query, memory_context or "无"),
        max_tokens=1200,
        timeout=get_app_settings().planner_timeout,
    )
    parsed = _parse_json(response)
    steps = _steps_from_payload(parsed)
    if steps:
        return steps
    return _fallback_plan(query)


def _replan(query: str, current_steps: list[PlanStep], observations: list[dict[str, Any]]) -> list[PlanStep] | None:
    pending_steps = [asdict(step) for step in current_steps if step.status == "pending"]
    if not pending_steps:
        return None
    response = _call_llm(
        get_model_settings().planner_replan_model,
        prompts.PLANNER_REPLAN_PROMPT.format(
            query,
            json.dumps(pending_steps, ensure_ascii=False, indent=2),
            json.dumps(observations[-2:], ensure_ascii=False, indent=2),
        ),
        max_tokens=800,
        timeout=get_app_settings().planner_timeout,
    )
    parsed = _parse_json(response)
    action = str(parsed.get("action", "")).strip().lower()
    if action == "update":
        steps = _steps_from_payload(parsed)
        return steps or None
    return None


def _execute_step(step: PlanStep, sender_id: str, session_id: str, trace_id: str) -> dict[str, Any]:
    if step.step_type == "task":
        result = request_nlu(step.query or step.goal or "", trace_id, True, None, session_id)
        if result.get("needs_disambiguation"):
            message = _build_candidate_message(result.get("candidates", []))
            return {
                "type": "task",
                "status": "need_user",
                "summary": message,
                "message": message,
                "candidates": result.get("candidates", []),
            }
        if result.get("function") and result.get("function") != "Unknown":
            summary = result.get("nlg") or f"已执行 {result.get('function')}"
            return {
                "type": "task",
                "status": "success",
                "summary": summary,
                "result": result,
            }
        return {
            "type": "task",
            "status": "failed",
            "summary": result.get("nlg") or "任务工具未命中，后续将尝试调整计划。",
            "result": result,
        }

    if step.step_type == "research":
        result = request_deep_research(step.query or step.goal or "", sender_id, session_id)
        status = "success" if result.get("answer") else "failed"
        return {
            "type": "research",
            "status": status,
            "summary": str(result.get("answer", "") or "")[:300],
            "result": result,
            "sources": result.get("sources", []),
        }

    if step.step_type == "memory":
        result = search_memory(step.query or step.goal or "", session_ids=[session_id], top_k=3)
        summary = _format_memory_hits(result)
        return {
            "type": "memory",
            "status": "success" if summary else "failed",
            "summary": summary or "未检索到可用长期记忆。",
        }

    if step.step_type == "respond":
        answer = _finalize_answer(step.goal or step.query, [], [])
        return {
            "type": "respond",
            "status": "success",
            "summary": answer,
            "answer": answer,
        }

    return {
        "type": step.step_type,
        "status": "failed",
        "summary": f"未知步骤类型：{step.step_type}",
    }


def _finalize_answer(query: str, steps: list[PlanStep], observations: list[dict[str, Any]]) -> str:
    serialized_steps = [asdict(step) for step in steps] if steps else []
    response = _call_llm(
        get_model_settings().planner_model,
        prompts.PLANNER_FINAL_PROMPT.format(
            query,
            json.dumps(serialized_steps, ensure_ascii=False, indent=2),
            json.dumps(observations, ensure_ascii=False, indent=2),
        ),
        max_tokens=1500,
        timeout=get_app_settings().planner_timeout,
    )
    if response:
        return response.strip()

    summaries = [item.get("summary", "") for item in observations if item.get("summary")]
    if summaries:
        return "已按步骤执行完成：\n" + "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(summaries))
    return "我已经完成规划，但暂时无法生成最终总结。"


def _fallback_plan(query: str) -> list[PlanStep]:
    normalized = (query or "").strip()
    if _needs_research(normalized):
        return [
            PlanStep(step_id="step1", step_type="research", goal="检索外部资料", query=normalized),
            PlanStep(step_id="step2", step_type="respond", goal="整合并回复用户"),
        ]

    subqueries = _split_complex_query(normalized)
    if len(subqueries) > 1:
        steps = [
            PlanStep(step_id=f"step{index + 1}", step_type="task", goal=item, query=item)
            for index, item in enumerate(subqueries[:3])
        ]
        steps.append(PlanStep(step_id=f"step{len(steps) + 1}", step_type="respond", goal="总结所有执行结果"))
        return steps

    return [
        PlanStep(step_id="step1", step_type="task", goal=normalized, query=normalized),
        PlanStep(step_id="step2", step_type="respond", goal="总结执行结果"),
    ]


def _needs_research(query: str) -> bool:
    keywords = ("调研", "对比", "分析", "查官网", "最新", "资料", "论文", "报道")
    return any(keyword in query for keyword in keywords)


def _split_complex_query(query: str) -> list[str]:
    parts = re.split(r"(?:然后|再|并且|同时|顺便|接着|之后)", query)
    cleaned = [item.strip(" ，。；;") for item in parts if item.strip(" ，。；;")]
    return cleaned


def _steps_from_payload(payload: dict[str, Any]) -> list[PlanStep]:
    steps_raw = payload.get("steps")
    if not isinstance(steps_raw, list):
        return []
    steps: list[PlanStep] = []
    for index, item in enumerate(steps_raw, start=1):
        if not isinstance(item, dict):
            continue
        step_type = str(item.get("step_type", "")).strip().lower()
        if step_type not in {"task", "research", "memory", "respond"}:
            continue
        steps.append(
            PlanStep(
                step_id=str(item.get("step_id") or f"step{index}"),
                step_type=step_type,
                goal=str(item.get("goal", "")).strip(),
                query=str(item.get("query", "")).strip(),
            )
        )
    return steps


def _format_memory_hits(result: Any) -> str:
    hits = getattr(result, "hits", None)
    if not hits:
        return ""
    lines = []
    for hit in hits[:3]:
        intent = getattr(hit, "intent", "") or ""
        function_name = getattr(hit, "function_name", "") or ""
        prefix = f"intent={intent} function={function_name}".strip()
        snippet = getattr(hit, "verbatim_snippet", "") or ""
        lines.append(f"{prefix}\n{snippet[:200]}".strip())
    return "\n\n".join(lines)


def _build_candidate_message(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "当前步骤需要你补充更明确的信息。"
    lines = ["当前任务存在多个候选，请回复序号或名称确认："]
    for index, item in enumerate(candidates[:5], start=1):
        lines.append(f"{index}. {item.get('display_name') or item.get('intent') or item.get('function')}")
    return "\n".join(lines)


def _collect_sources(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in observations:
        for source in item.get("sources", []) or []:
            key = (str(source.get("title", "")), str(source.get("url", "")))
            if key in seen:
                continue
            seen.add(key)
            sources.append({"title": key[0], "url": key[1]})
    return sources


def _call_llm(model: str, user_prompt: str, *, max_tokens: int, timeout: float) -> str:
    return call_text_model(
        model,
        user_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=0.2,
    )


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _split_text(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])", text)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        buffer += part
        if len(buffer) >= 80 or re.search(r"[。！？；]\s*$", buffer):
            chunks.append(buffer)
            buffer = ""
    if buffer.strip():
        chunks.append(buffer)
    return chunks
