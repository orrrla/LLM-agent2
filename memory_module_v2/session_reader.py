from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.runtime import get_app_settings


def session_path(session_id: str) -> Path:
    settings = get_app_settings()
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    return settings.sessions_dir / f"{session_id}.json"


def load_session_raw(session_id: str) -> dict[str, Any] | None:
    path = session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_session(session_id: str) -> list[dict[str, Any]]:
    data = load_session_raw(session_id)
    if not data:
        return []
    return data.get("messages", [])
