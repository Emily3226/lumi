from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from api.db import get_db

logger = logging.getLogger(__name__)

# Chat sessions used to be per-session JSON files on local disk. That breaks
# under Cloud Run (and any multi-instance deploy): each instance has its own
# ephemeral disk, so consecutive turns of one conversation can land on
# different instances and silently look like a brand-new session. MongoDB
# Atlas is already the backing store for everything else in this app, so
# sessions live there too now - one document per session_id.
COLLECTION_NAME = "chat_sessions"

# How long a session document can go untouched before MongoDB's TTL index
# expires it. Override with SESSION_TTL_DAYS; set to 0 to disable expiry.
SESSION_TTL_DAYS = int(os.environ.get("SESSION_TTL_DAYS", "30"))

# Mongo's own bookkeeping fields - never merged into the session dict itself.
_RESERVED_FIELDS = ("_id", "updated_at")


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


def _ensure_ttl_index() -> None:
    if SESSION_TTL_DAYS <= 0:
        return
    try:
        get_db()[COLLECTION_NAME].create_index(
            "updated_at", expireAfterSeconds=SESSION_TTL_DAYS * 86400
        )
    except Exception:
        logger.warning("Could not ensure TTL index on %s", COLLECTION_NAME, exc_info=True)


_ensure_ttl_index()

# In-memory cache of loaded sessions for this process, keyed by session_id.
_session_cache: dict[str, dict[str, Any]] = {}


def load_session(session_id: str) -> dict[str, Any]:
    try:
        doc = get_db()[COLLECTION_NAME].find_one({"_id": session_id})
    except Exception:
        logger.warning("Failed to load session %r from MongoDB", session_id, exc_info=True)
        return new_session()

    if not doc:
        return new_session()

    # agents.py stashes several ad-hoc keys on the session dict beyond the
    # fixed set in new_session() (pending_slots, pending_mentee_email, etc.)
    # - persist and restore whatever is actually there, not a fixed allowlist,
    # or those fields silently vanish on the next turn.
    session = new_session()
    session.update({k: v for k, v in doc.items() if k not in _RESERVED_FIELDS})
    if not isinstance(session.get("messages"), list):
        session["messages"] = []
    if not isinstance(session.get("matches"), list):
        session["matches"] = []
    return session


def get_session(session_id: str) -> dict[str, Any]:
    # Always re-read from MongoDB rather than trusting a cached in-memory
    # copy. With multiple instances/workers, consecutive turns of the same
    # conversation can land on different processes with no session affinity;
    # a stale in-memory copy would silently look like a brand-new session and
    # reset the whole flow (e.g. mid-booking state getting wiped).
    _session_cache[session_id] = load_session(session_id)
    return _session_cache[session_id]


def save_session(session_id: str) -> None:
    """Persist the given session's current in-memory state to MongoDB."""
    session = _session_cache.get(session_id)
    if session is None:
        return
    doc = {k: v for k, v in session.items() if k not in _RESERVED_FIELDS}
    doc["updated_at"] = datetime.now(timezone.utc)
    try:
        get_db()[COLLECTION_NAME].replace_one({"_id": session_id}, doc, upsert=True)
    except Exception:
        logger.warning("Failed to save session %r to MongoDB", session_id, exc_info=True)


def reset_session(session_id: str) -> None:
    _session_cache[session_id] = new_session()
    save_session(session_id)
