from __future__ import annotations

from typing import Any

from config.runtime import get_app_settings
from utils import logger

from .config import get_memory_config
from .distill import distill_exchange
from .models import SearchResponse
from .retrieval import search_memory as _search_memory
from .segmenter import segment_exchanges
from .session_reader import load_session_raw, read_session
from .store import ExchangesRepo, ObjectsRepo, ensure_schema


_SCHEMA_READY = False


def distill_session(session_id: str, *, force: bool = False) -> dict[str, Any]:
    config = get_memory_config()
    if not config.enabled:
        return {"session_id": session_id, "enabled": False}

    _maybe_init_schema()

    messages = read_session(session_id)
    if not messages:
        return {"session_id": session_id, "exchanges_total": 0, "objects_created": 0}

    exchanges = segment_exchanges(session_id, messages, min_exchange_chars=config.min_exchange_chars)
    exchange_repo = ExchangesRepo()
    object_repo = ObjectsRepo()
    existing_ids = set() if force else exchange_repo.get_exchange_ids_for_session(session_id)
    to_process = exchanges if force else [exchange for exchange in exchanges if exchange.exchange_id not in existing_ids]

    exchange_repo.upsert_batch(exchanges if force else to_process)

    created = 0
    for exchange in to_process:
        try:
            obj = distill_exchange(exchange)
            object_repo.upsert(obj)
            created += 1
        except Exception as exc:
            logger.warning(f"memory distill exchange failed: {exc}")

    return {
        "session_id": session_id,
        "exchanges_total": len(exchanges),
        "exchanges_new": len(to_process),
        "objects_created": created,
    }


def search_memory(query: str, *, session_ids: list[str] | None = None, top_k: int | None = None):
    config = get_memory_config()
    settings = get_app_settings()
    if not config.enabled or not settings.memory_search_enabled:
        return SearchResponse(query=query, hits=[])
    _maybe_init_schema()
    return _search_memory(query, session_ids=session_ids, top_k=top_k)


def build_memory_context(query: str, session_id: str, *, top_k: int | None = None) -> str:
    config = get_memory_config()
    if not config.enabled or config.inject_mode == "off":
        return ""
    try:
        response = search_memory(query, session_ids=[session_id], top_k=top_k or config.inject_top_k)
    except Exception as exc:
        logger.warning(f"memory context build failed: {exc}")
        return ""
    if not response.hits:
        return ""

    parts = ["[长期记忆检索结果]"]
    for hit in response.hits:
        header = f"{hit.rank}. session={hit.session_id} ply={hit.ply_start}-{hit.ply_end}"
        tags = []
        if hit.route:
            tags.append(f"route={hit.route}")
        if hit.intent:
            tags.append(f"intent={hit.intent}")
        if hit.function_name:
            tags.append(f"function={hit.function_name}")
        if tags:
            header += " " + " ".join(tags)
        parts.append(header + "\n" + hit.verbatim_snippet[:500])
    return "\n\n".join(parts)


def get_exchange(session_id: str, ply_start: int, ply_end: int) -> dict[str, Any]:
    data = load_session_raw(session_id)
    if not data:
        return {"session_id": session_id, "ply_start": ply_start, "ply_end": ply_end, "messages": []}
    messages = data.get("messages", [])[ply_start: ply_end + 1]
    return {
        "session_id": session_id,
        "ply_start": ply_start,
        "ply_end": ply_end,
        "messages": messages,
    }


def _maybe_init_schema() -> None:
    global _SCHEMA_READY
    settings = get_app_settings()
    if _SCHEMA_READY or not settings.memory_init_schema:
        return
    ensure_schema()
    _SCHEMA_READY = True
