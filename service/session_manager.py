from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SessionManager:
    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _default_record(self, session_id: str) -> dict[str, Any]:
        now = time.time()
        return {
            "id": session_id,
            "title": session_id,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

    def load_record(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        if not path.exists():
            record = self._default_record(session_id)
            self._write(record)
            return record

        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            record = self._default_record(session_id)
            record["messages"] = raw
            self._write(record)
            return record

        raw.setdefault("id", session_id)
        raw.setdefault("title", session_id)
        raw.setdefault("created_at", time.time())
        raw.setdefault("updated_at", raw["created_at"])
        raw.setdefault("messages", [])
        return raw

    def _write(self, record: dict[str, Any]) -> None:
        record["updated_at"] = time.time()
        self._path(str(record["id"])).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self.load_record(session_id)
        message: dict[str, Any] = {
            "role": role,
            "content": content,
            "created_at": time.time(),
        }
        if metadata:
            message["metadata"] = metadata
        record["messages"].append(message)
        self._write(record)
        return message

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self.load_record(session_id).get("messages", [])
