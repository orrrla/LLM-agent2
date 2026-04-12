from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Exchange:
    exchange_id: str
    session_id: str
    ply_start: int
    ply_end: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    verbatim_text: str = ""
    verbatim_snippet: str = ""
    message_count: int = 0


@dataclass
class DistilledObject:
    object_id: str
    exchange_id: str
    session_id: str
    ply_start: int
    ply_end: int
    exchange_core: str
    specific_context: str
    distill_text: str
    route: str = ""
    intent: str = ""
    function_name: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    distilled_at: datetime | None = None
    distill_model: str = ""


@dataclass
class MemoryHit:
    rank: int
    session_id: str
    exchange_id: str
    ply_start: int
    ply_end: int
    verbatim_snippet: str
    object_id: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    intent: str = ""
    route: str = ""
    function_name: str = ""


@dataclass
class SearchResponse:
    query: str
    hits: list[MemoryHit] = field(default_factory=list)
