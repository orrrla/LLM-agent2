from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

from config.runtime import get_app_settings

from .models import DistilledObject, Exchange


SCHEMA_SQL = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")


def get_connection(*, autocommit: bool = False):
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    settings = get_app_settings()
    return psycopg.connect(settings.postgres_resolved_dsn, autocommit=autocommit)


def ensure_schema() -> None:
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


class ExchangesRepo:
    def upsert_batch(self, exchanges: list[Exchange]) -> None:
        if not exchanges:
            return
        sql = """
            INSERT INTO memory_v2.memory_exchanges
                (exchange_id, session_id, ply_start, ply_end,
                 verbatim_text, verbatim_snippet, message_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (exchange_id) DO UPDATE SET
                verbatim_text = EXCLUDED.verbatim_text,
                verbatim_snippet = EXCLUDED.verbatim_snippet,
                message_count = EXCLUDED.message_count,
                updated_at = now()
        """
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                for exchange in exchanges:
                    cur.execute(
                        sql,
                        (
                            exchange.exchange_id,
                            exchange.session_id,
                            exchange.ply_start,
                            exchange.ply_end,
                            exchange.verbatim_text,
                            exchange.verbatim_snippet,
                            exchange.message_count,
                        ),
                    )

    def get_exchange_ids_for_session(self, session_id: str) -> set[str]:
        sql = "SELECT exchange_id FROM memory_v2.memory_exchanges WHERE session_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
                return {row[0] for row in cur.fetchall()}

    def get_by_backref(self, session_id: str, ply_start: int, ply_end: int) -> dict[str, Any] | None:
        sql = """
            SELECT exchange_id, session_id, ply_start, ply_end, verbatim_snippet
            FROM memory_v2.memory_exchanges
            WHERE session_id = %s AND ply_start = %s AND ply_end = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id, ply_start, ply_end))
                row = cur.fetchone()
                if row is None:
                    return None
                return {
                    "exchange_id": row[0],
                    "session_id": row[1],
                    "ply_start": row[2],
                    "ply_end": row[3],
                    "verbatim_snippet": row[4],
                }

    def fetch_bm25_corpus(self, session_ids: list[str] | None = None) -> list[dict[str, Any]]:
        conditions = []
        params: list[Any] = []
        if session_ids:
            conditions.append("session_id = ANY(%s)")
            params.append(session_ids)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT exchange_id, session_id, ply_start, ply_end, verbatim_text, verbatim_snippet
            FROM memory_v2.memory_exchanges
            {where}
            ORDER BY created_at DESC
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [item[0] for item in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]


class ObjectsRepo:
    def upsert(self, obj: DistilledObject) -> None:
        sql = """
            INSERT INTO memory_v2.memory_objects
                (object_id, exchange_id, session_id, ply_start, ply_end,
                 exchange_core, specific_context, distill_text, route, intent,
                 function_name, slots, keywords, distill_model, distilled_at, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (object_id) DO UPDATE SET
                exchange_core = EXCLUDED.exchange_core,
                specific_context = EXCLUDED.specific_context,
                distill_text = EXCLUDED.distill_text,
                route = EXCLUDED.route,
                intent = EXCLUDED.intent,
                function_name = EXCLUDED.function_name,
                slots = EXCLUDED.slots,
                keywords = EXCLUDED.keywords,
                distill_model = EXCLUDED.distill_model,
                distilled_at = EXCLUDED.distilled_at,
                embedding = EXCLUDED.embedding
        """
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        obj.object_id,
                        obj.exchange_id,
                        obj.session_id,
                        obj.ply_start,
                        obj.ply_end,
                        obj.exchange_core,
                        obj.specific_context,
                        obj.distill_text,
                        obj.route,
                        obj.intent,
                        obj.function_name,
                        json.dumps(obj.slots, ensure_ascii=False),
                        json.dumps(obj.keywords, ensure_ascii=False),
                        obj.distill_model,
                        obj.distilled_at or datetime.now(timezone.utc),
                        _format_vector(obj.embedding) if obj.embedding else None,
                    ),
                )

    def dense_search(self, query_embedding: list[float], top_k: int, session_ids: list[str] | None = None) -> list[dict[str, Any]]:
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if session_ids:
            conditions.append("session_id = ANY(%s)")
            params.append(session_ids)
        where = "WHERE " + " AND ".join(conditions)
        emb = _format_vector(query_embedding)
        sql = f"""
            SELECT object_id, exchange_id, session_id, ply_start, ply_end, route, intent,
                   function_name, distill_text, 1.0 / (1.0 + (embedding <=> %s)) AS dense_score
            FROM memory_v2.memory_objects
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [emb] + params + [emb, top_k])
                cols = [item[0] for item in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]


def _format_vector(vector: list[float]) -> str:
    return "[" + ",".join(str(item) for item in vector) + "]"
