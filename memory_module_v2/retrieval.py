from __future__ import annotations

import json
import re
from typing import Any

import requests
try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

from config.runtime import get_app_settings, get_model_settings
from utils import logger

from .config import get_memory_config
from .models import MemoryHit, SearchResponse
from .store import ExchangesRepo, ObjectsRepo

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None


def search_memory(query: str, *, session_ids: list[str] | None = None, top_k: int | None = None) -> SearchResponse:
    config = get_memory_config()
    final_top_k = top_k or config.inject_top_k
    dense_candidates = _dense_search(query, session_ids=session_ids)
    keyword_candidates = _keyword_search(query, session_ids=session_ids)
    fused = _weighted_sum_fusion(
        dense_candidates,
        keyword_candidates,
        dense_weight=config.dense_weight,
        keyword_weight=config.keyword_weight,
        top_k=final_top_k,
    )

    repo = ExchangesRepo()
    hits: list[MemoryHit] = []
    for rank, item in enumerate(fused, start=1):
        exchange_row = repo.get_by_backref(item["session_id"], item["ply_start"], item["ply_end"])
        hits.append(
            MemoryHit(
                rank=rank,
                session_id=item["session_id"],
                exchange_id=item["exchange_id"],
                ply_start=item["ply_start"],
                ply_end=item["ply_end"],
                verbatim_snippet=(exchange_row or {}).get("verbatim_snippet", item.get("verbatim_snippet", "")),
                object_id=item.get("object_id"),
                scores={
                    "dense": float(item.get("dense_score", 0.0)),
                    "keyword": float(item.get("keyword_score", 0.0)),
                    "fused": float(item.get("fused_score", 0.0)),
                },
                intent=str(item.get("intent", "")),
                route=str(item.get("route", "")),
                function_name=str(item.get("function_name", "")),
            )
        )
    return SearchResponse(query=query, hits=hits)


def _dense_search(query: str, *, session_ids: list[str] | None = None) -> list[dict[str, Any]]:
    query_embedding = _embed_query(query)
    if not query_embedding:
        return []
    repo = ObjectsRepo()
    try:
        return repo.dense_search(query_embedding, top_k=get_memory_config().dense_top_k, session_ids=session_ids)
    except Exception as exc:
        logger.warning(f"memory dense search fallback: {exc}")
        return []


def _keyword_search(query: str, *, session_ids: list[str] | None = None) -> list[dict[str, Any]]:
    if BM25Okapi is None:
        return []
    try:
        corpus = ExchangesRepo().fetch_bm25_corpus(session_ids=session_ids)
    except Exception as exc:
        logger.warning(f"memory keyword corpus fallback: {exc}")
        return []
    docs = []
    exchange_ids = []
    metadata = {}
    for item in corpus:
        tokens = _tokenize(item.get("verbatim_snippet") or item.get("verbatim_text") or "")
        if not tokens:
            continue
        exchange_id = item["exchange_id"]
        exchange_ids.append(exchange_id)
        docs.append(tokens)
        metadata[exchange_id] = item
    if not docs:
        return []
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:get_memory_config().keyword_top_k]
    results = []
    for idx, score in ranked:
        if score <= 0:
            continue
        exchange_id = exchange_ids[idx]
        row = metadata[exchange_id]
        results.append(
            {
                "exchange_id": exchange_id,
                "session_id": row["session_id"],
                "ply_start": row["ply_start"],
                "ply_end": row["ply_end"],
                "verbatim_snippet": row.get("verbatim_snippet", ""),
                "keyword_score": float(score),
            }
        )
    return results


def _weighted_sum_fusion(
    dense_candidates: list[dict[str, Any]],
    keyword_candidates: list[dict[str, Any]],
    *,
    dense_weight: float,
    keyword_weight: float,
    top_k: int,
) -> list[dict[str, Any]]:
    dense_max = max((item.get("dense_score", 0.0) for item in dense_candidates), default=1.0) or 1.0
    keyword_max = max((item.get("keyword_score", 0.0) for item in keyword_candidates), default=1.0) or 1.0
    scores: dict[str, float] = {}
    merged: dict[str, dict[str, Any]] = {}

    for item in dense_candidates:
        exchange_id = item["exchange_id"]
        merged[exchange_id] = dict(merged.get(exchange_id, {}), **item)
        scores[exchange_id] = scores.get(exchange_id, 0.0) + dense_weight * (item.get("dense_score", 0.0) / dense_max)

    for item in keyword_candidates:
        exchange_id = item["exchange_id"]
        merged[exchange_id] = dict(merged.get(exchange_id, {}), **item)
        scores[exchange_id] = scores.get(exchange_id, 0.0) + keyword_weight * (item.get("keyword_score", 0.0) / keyword_max)

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    results = []
    for exchange_id, fused_score in ranked:
        item = merged[exchange_id]
        item["fused_score"] = fused_score
        item.setdefault("dense_score", 0.0)
        item.setdefault("keyword_score", 0.0)
        results.append(item)
    return results


def _embed_query(query: str) -> list[float] | None:
    settings = get_app_settings()
    models = get_model_settings()
    if not settings.memory_embedding_url or not models.embedding_model:
        return None
    headers = {
        "Authorization": settings.memory_embedding_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": models.embedding_model,
        "input": query,
    }
    try:
        response = requests.post(
            settings.memory_embedding_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()["data"][0]["embedding"]
        return [float(item) for item in data]
    except Exception as exc:
        logger.warning(f"memory embedding fallback: {exc}")
        return None


def _tokenize(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if jieba is not None:
        return [token.strip() for token in jieba.cut(text) if token.strip()]
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text)
