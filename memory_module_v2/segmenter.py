from __future__ import annotations

import hashlib
from typing import Any

from .models import Exchange


def _message_text(message: dict[str, Any]) -> str:
    content = str(message.get("content", "") or "").strip()
    metadata = message.get("metadata") or {}
    if isinstance(metadata, dict):
        extras = []
        route = metadata.get("route")
        intent = metadata.get("intent")
        function_name = metadata.get("function")
        slots = metadata.get("slots")
        if route:
            extras.append(f"route={route}")
        if intent:
            extras.append(f"intent={intent}")
        if function_name:
            extras.append(f"function={function_name}")
        if slots:
            extras.append(f"slots={slots}")
        if extras:
            content = f"{content}\n" + "\n".join(extras) if content else "\n".join(extras)
    return content


def segment_exchanges(
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    min_exchange_chars: int = 40,
) -> list[Exchange]:
    exchanges: list[Exchange] = []
    pending_user_index: int | None = None
    pending_messages: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            if pending_user_index is not None and pending_messages:
                exchange = _build_exchange(session_id, pending_user_index, index - 1, pending_messages)
                if len(exchange.verbatim_text) >= min_exchange_chars:
                    exchanges.append(exchange)
            pending_user_index = index
            pending_messages = [message]
            continue

        if pending_user_index is not None:
            pending_messages.append(message)

    if pending_user_index is not None and pending_messages:
        exchange = _build_exchange(session_id, pending_user_index, len(messages) - 1, pending_messages)
        if exchange.verbatim_text.strip():
            exchanges.append(exchange)

    return exchanges


def _build_exchange(session_id: str, ply_start: int, ply_end: int, messages: list[dict[str, Any]]) -> Exchange:
    lines = []
    snippet = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        text = _message_text(message)
        if not text:
            continue
        lines.append(f"{role}: {text}")
        if role in {"USER", "ASSISTANT", "TOOL"}:
            snippet.append(f"{role}: {text}")

    verbatim_text = "\n\n".join(lines)
    exchange_id = hashlib.md5(f"{session_id}:{ply_start}:{ply_end}:{verbatim_text}".encode("utf-8")).hexdigest()
    return Exchange(
        exchange_id=exchange_id,
        session_id=session_id,
        ply_start=ply_start,
        ply_end=ply_end,
        messages=messages,
        verbatim_text=verbatim_text,
        verbatim_snippet="\n\n".join(snippet)[:2000],
        message_count=len(messages),
    )
