from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import requests

from config.runtime import get_app_settings, get_model_settings
from utils import logger

from .models import DistilledObject, Exchange


DISTILL_SYSTEM_PROMPT = """你是车载任务型对话长期记忆蒸馏器。
请把一段多轮对话蒸馏成 JSON，字段必须包含：
- exchange_core: 这段交互的核心任务或问题
- specific_context: 这段交互里的关键条件、地点、时间、约束、结果
- route: task/chat/reject/unknown
- intent: 意图名称，没有就写空字符串
- function_name: 函数或技能名，没有就写空字符串
- slots: 对话里可确定的槽位对象，没有就返回 {}
- keywords: 5到10个短关键词数组
只返回 JSON，不要解释。"""


def distill_exchange(exchange: Exchange) -> DistilledObject:
    metadata = _extract_metadata(exchange.messages)
    raw = _call_distill_llm(exchange.verbatim_snippet or exchange.verbatim_text)

    exchange_core = str(raw.get("exchange_core") or metadata.get("query") or exchange.verbatim_snippet[:120]).strip()
    specific_context = str(raw.get("specific_context") or _fallback_specific_context(exchange.verbatim_text)).strip()
    route = str(raw.get("route") or metadata.get("route") or "unknown").strip()
    intent = str(raw.get("intent") or metadata.get("intent") or "").strip()
    function_name = str(raw.get("function_name") or metadata.get("function") or "").strip()
    slots = raw.get("slots") if isinstance(raw.get("slots"), dict) else metadata.get("slots") or {}
    keywords = raw.get("keywords") if isinstance(raw.get("keywords"), list) else _fallback_keywords(exchange.verbatim_text, metadata)

    distill_text = "\n".join(
        line for line in [
            exchange_core,
            specific_context,
            f"route={route}" if route else "",
            f"intent={intent}" if intent else "",
            f"function={function_name}" if function_name else "",
            f"slots={json.dumps(slots, ensure_ascii=False)}" if slots else "",
            "keywords=" + " ".join(str(item) for item in keywords) if keywords else "",
        ]
        if line
    )

    return DistilledObject(
        object_id=exchange.exchange_id,
        exchange_id=exchange.exchange_id,
        session_id=exchange.session_id,
        ply_start=exchange.ply_start,
        ply_end=exchange.ply_end,
        exchange_core=exchange_core or exchange.verbatim_snippet[:120],
        specific_context=specific_context or exchange.verbatim_snippet[:240],
        distill_text=distill_text,
        route=route,
        intent=intent,
        function_name=function_name,
        slots=slots,
        keywords=[str(item) for item in keywords][:10],
        distilled_at=datetime.now(timezone.utc),
        distill_model=get_model_settings().distill_model,
    )


def _call_distill_llm(exchange_text: str) -> dict[str, Any]:
    settings = get_app_settings()
    models = get_model_settings()
    headers = {
        "Authorization": settings.api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": models.distill_model,
        "messages": [
            {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
            {"role": "user", "content": exchange_text[:6000]},
        ],
        "temperature": 0.1,
        "top_p": 0.8,
    }
    try:
        response = requests.post(
            settings.base_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=settings.memory_distill_timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = "\n".join(line for line in content.splitlines() if not line.startswith("```")).strip()
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.warning(f"memory distill fallback: {exc}")
    return {}


def _extract_metadata(messages: list[dict[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for message in messages:
        metadata = message.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        for key in ("route", "intent", "function", "slots", "query"):
            if metadata.get(key) not in (None, "", {}, []):
                data[key] = metadata.get(key)
    return data


def _fallback_specific_context(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:300]


def _fallback_keywords(text: str, metadata: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    for value in (metadata.get("intent"), metadata.get("function"), metadata.get("route")):
        if value:
            keywords.append(str(value))
    slots = metadata.get("slots") or {}
    if isinstance(slots, dict):
        for key, value in slots.items():
            keywords.append(str(key))
            keywords.append(str(value))
    keywords.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", text)[:8])
    seen = set()
    result = []
    for item in keywords:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result[:10]
