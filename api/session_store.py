from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any


# Each session gets its own file (mirrors api/memory_store.py). The old
# design kept every session in one chat_sessions.json and rewrote the WHOLE
# file - every user's data - on every single chat turn from any user. That
# made each request's cost grow with the total number of sessions ever
# created (nothing expired them), and two uvicorn workers writing the same
# file could clobber each other's latest state. Per-session files fix both:
# a turn only touches its own small file, and stale sessions can be swept.
SESSION_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "chat_sessions"
_LEGACY_SESSION_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_sessions.json"

# How long a per-session file can sit untouched before it's treated as stale
# and deleted. Override with SESSION_TTL_DAYS; set to 0 to disable cleanup.
SESSION_TTL_DAYS = int(os.environ.get("SESSION_TTL_DAYS", "30"))

# save_session() also has a small chance of sweeping stale files so they
# don't pile up between deploys on a long-running process.
_CLEANUP_PROBABILITY_ON_SAVE = 0.01

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_\-]")


def _safe_path(session_id: str) -> Path:
    safe_id = _SAFE_ID_RE.sub("_", session_id)[:128]
    return SESSION_STORE_DIR / f"{safe_id}.json"


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


def _cleanup_stale_sessions(ttl_days: int = SESSION_TTL_DAYS) -> int:
    """Delete per-session files that haven't been written to in `ttl_days`
    days. Never raises - a failed sweep should never break an active chat.
    """
    if ttl_days <= 0:
        return 0
    removed = 0
    try:
        if not SESSION_STORE_DIR.exists():
            return 0
        cutoff = time.time() - (ttl_days * 86400)
        for path in SESSION_STORE_DIR.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def _migrate_legacy_store() -> None:
    """One-time migration: split the old single-file store (if present)
    into per-session files, then get out of the way.
    """
    if not _LEGACY_SESSION_STORE_PATH.exists():
        return
    try:
        raw = _LEGACY_SESSION_STORE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        data = {}

    if isinstance(data, dict):
        SESSION_STORE_DIR.mkdir(parents=True, exist_ok=True)
        for session_id, payload in data.items():
            if not (isinstance(session_id, str) and isinstance(payload, dict)):
                continue
            target = _safe_path(session_id)
            if target.exists():
                continue
            session = new_session()
            session.update(payload)
            try:
                target.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                continue

    try:
        _LEGACY_SESSION_STORE_PATH.unlink()
    except OSError:
        pass


_migrate_legacy_store()
_cleanup_stale_sessions()

# In-memory cache of loaded sessions for this process, keyed by session_id.
_session_cache: dict[str, dict[str, Any]] = {}


def load_session(session_id: str) -> dict[str, Any]:
    path = _safe_path(session_id)
    if not path.exists():
        return new_session()

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return new_session()

    if not isinstance(data, dict):
        return new_session()

    session = new_session()
    session.update(data)
    if not isinstance(session.get("messages"), list):
        session["messages"] = []
    if not isinstance(session.get("matches"), list):
        session["matches"] = []
    return session


def get_session(session_id: str) -> dict[str, Any]:
    if session_id not in _session_cache:
        _session_cache[session_id] = load_session(session_id)
    return _session_cache[session_id]


def save_session(session_id: str) -> None:
    """Persist the given session's current in-memory state to disk."""
    session = _session_cache.get(session_id)
    if session is None:
        return
    SESSION_STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = _safe_path(session_id)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    if random.random() < _CLEANUP_PROBABILITY_ON_SAVE:
        _cleanup_stale_sessions()


def reset_session(session_id: str) -> None:
    _session_cache[session_id] = new_session()
    save_session(session_id)
