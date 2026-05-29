from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SESSION_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_sessions.json"


def new_session() -> dict[str, Any]:
    return {
        "state": "idle",
        "subject": None,
        "grade": None,
        "name": None,
        "active_agent": "general",
        "pending_match_step": None,
        "query_text": None,
        "pending_booking_choice": None,
        "matches": [],
        "messages": [],
    }


def load_sessions() -> dict[str, dict[str, Any]]:
    if not SESSION_STORE_PATH.exists():
        return {}

    try:
        raw = SESSION_STORE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    sessions: dict[str, dict[str, Any]] = {}
    for session_id, payload in data.items():
        if isinstance(session_id, str) and isinstance(payload, dict):
            session = new_session()
            session.update(payload)
            if not isinstance(session.get("messages"), list):
                session["messages"] = []
            if not isinstance(session.get("matches"), list):
                session["matches"] = []
            sessions[session_id] = session
    return sessions


def save_sessions(sessions: dict[str, dict[str, Any]]) -> None:
    SESSION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = SESSION_STORE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(SESSION_STORE_PATH)


sessions: dict[str, dict[str, Any]] = load_sessions()


def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = new_session()
    return sessions[session_id]


def reset_session(session_id: str) -> None:
    sessions[session_id] = new_session()
    save_sessions(sessions)