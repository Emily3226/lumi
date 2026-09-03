from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from api.db import get_db

logger = logging.getLogger(__name__)

# Per-user memory (facts/summary/examples) used to be per-session JSON files
# on local disk - broken under Cloud Run / any multi-instance deploy for the
# same reason as api/session_store.py. MongoDB Atlas is already the backing
# store for the rest of the app, so memory lives there too now.
COLLECTION_NAME = "user_memory"
MAX_FACTS = 60
MAX_EXAMPLES = 12

# How long a memory document can go untouched before MongoDB's TTL index
# expires it. Override with MEMORY_TTL_DAYS; set it to 0 to disable expiry.
MEMORY_TTL_DAYS = int(os.environ.get("MEMORY_TTL_DAYS", "30"))


def _ensure_ttl_index() -> None:
    if MEMORY_TTL_DAYS <= 0:
        return
    try:
        get_db()[COLLECTION_NAME].create_index(
            "updated_at", expireAfterSeconds=MEMORY_TTL_DAYS * 86400
        )
    except Exception:
        logger.warning("Could not ensure TTL index on %s", COLLECTION_NAME, exc_info=True)


_ensure_ttl_index()


def _default_memory() -> dict[str, Any]:
    return {
        "summary": "",
        "facts": [],
        "examples": [],
        "updated_at": "",
    }


def _memory_key(session_id: str | None) -> str:
    return session_id or ""


def load_memory(session_id: str | None = None) -> dict[str, Any]:
    try:
        doc = get_db()[COLLECTION_NAME].find_one({"_id": _memory_key(session_id)})
    except Exception:
        logger.warning("Failed to load memory for session %r from MongoDB", session_id, exc_info=True)
        return _default_memory()

    if not doc:
        return _default_memory()

    memory = _default_memory()
    memory["summary"] = doc.get("summary") or ""
    memory["facts"] = doc["facts"] if isinstance(doc.get("facts"), list) else []
    memory["examples"] = doc["examples"] if isinstance(doc.get("examples"), list) else []
    updated_at = doc.get("updated_at")
    memory["updated_at"] = updated_at.isoformat() if hasattr(updated_at, "isoformat") else (updated_at or "")
    return memory


# In-memory cache of per-session memory dicts, keyed by session_id
# ("" used for the legacy/no-session fallback).
_memory_cache: dict[str, dict[str, Any]] = {}


def _get_memory(session_id: str | None) -> dict[str, Any]:
    # Always re-read from MongoDB rather than trusting a cached in-memory
    # copy - see the identical comment in api/session_store.py:get_session().
    # With multiple instances, a stale per-process cache can silently hide
    # facts/examples another instance already persisted for this session.
    key = _memory_key(session_id)
    _memory_cache[key] = load_memory(session_id)
    return _memory_cache[key]


def save_memory(session_id: str | None = None) -> None:
    # Persist whatever the caller mutated in place on the cached dict (see
    # observe_turn/clear_session_memory below) - this must NOT call
    # _get_memory() here, since that would reload the pre-mutation copy from
    # MongoDB and silently discard every fact/example just extracted.
    key = _memory_key(session_id)
    memory = _memory_cache.get(key)
    if memory is None:
        return
    now = datetime.now(timezone.utc)
    memory["updated_at"] = now.isoformat()
    doc = {
        "summary": memory.get("summary", ""),
        "facts": memory.get("facts", []),
        "examples": memory.get("examples", []),
        "updated_at": now,
    }
    try:
        get_db()[COLLECTION_NAME].replace_one({"_id": key}, doc, upsert=True)
    except Exception:
        logger.warning("Failed to save memory for session %r to MongoDB", session_id, exc_info=True)


def clear_session_memory(session_id: str | None = None) -> None:
    """Reset ONE session's persistent memory to defaults.

    Intended to be called whenever that specific conversation/session is
    reset, so facts, summaries, and examples from its previous history don't
    leak into the new one. This only ever touches the given session_id's
    memory document - never any other user's.
    """
    key = _memory_key(session_id)
    _memory_cache[key] = _default_memory()
    save_memory(session_id)


def _add_fact(memory: dict[str, Any], fact: str) -> None:
    fact = fact.strip()
    if not fact:
        return
    facts = memory.setdefault("facts", [])
    if fact.lower().startswith("user name:"):
        facts = [item for item in facts if not str(item).lower().startswith("user name:")]
    if fact in facts:
        return
    facts.append(fact)
    memory["facts"] = facts[-MAX_FACTS:]


def _add_example(memory: dict[str, Any], example: str) -> None:
    example = example.strip()
    if not example:
        return
    examples = memory.setdefault("examples", [])
    if example in examples:
        return
    examples.append(example)
    memory["examples"] = examples[-MAX_EXAMPLES:]


def _extract_facts_from_text(text: str) -> list[str]:
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    facts: list[str] = []

    def _extract_clause(after_phrase: str) -> str:
        candidate = after_phrase.strip()
        candidate = re.split(r"\b(?:and\s+i\s+am|and\s+i'm|and\s+im|and\s+i|because|but|so)\b", candidate, maxsplit=1, flags=re.I)[0]
        candidate = re.split(r"[.?!,]", candidate, maxsplit=1)[0]
        return candidate.strip()

    patterns = [
        (r"\bi am in grade (\d{1,2})\b", "grade"),
        (r"\bi(?:'m| am) in grade (\d{1,2})\b", "grade"),
        (r"\bi(?:'m| am) in the gifted program(?: at ([^.?!]{2,80}))?", "program"),
        (r"\bi(?:'m| am) in ap\b(?: ([^.?!]{2,80}))?", "program"),
        (r"\bi(?:'m| am) in ib\b(?: ([^.?!]{2,80}))?", "program"),
        (r"\bi like ([^.?!]{2,80})", "interest"),
        (r"\bi enjoy ([^.?!]{2,80})", "interest"),
        (r"\bmy favorite subject is ([^.?!]{2,80})", "preference"),
        (r"\bi attend ([^.?!]{2,80})", "school"),
        (r"\bmy school is ([^.?!]{2,80})", "school"),
        (r"\bi go to ([^.?!]{2,80})", "school"),
        (r"\bmy goal is to ([^.?!]{2,120})", "goal"),
        (r"\bi want to go to ([^.?!]{2,120})", "goal"),
        (r"\bi want to study ([^.?!]{2,120})", "goal"),
        (r"\bi am considering ([^.?!]{2,120})", "goal"),
        (r"\bremember that ([^.?!]{2,120})", "memory"),
        (r"\bplease remember that ([^.?!]{2,120})", "memory"),
    ]

    name_match = re.search(r"\bmy name is\s+(.+)$", normalized, re.I)
    if name_match:
        candidate = _extract_clause(name_match.group(1))
        candidate = re.sub(r"\s+\b(i|im|i'm)\b.*$", "", candidate, flags=re.I).strip()
        candidate = re.sub(r"[^A-Za-z\-']+", " ", candidate).strip()
        if candidate:
            facts.append(f"User name: {candidate.title()}")

    for pattern, label in patterns:
        match = re.search(pattern, lowered, re.I)
        if not match:
            continue
        value = match.group(1).strip().rstrip(".")
        if label == "name":
            facts.append(f"User name: {value.title()}")
        elif label == "grade":
            facts.append(f"User grade: {value}")
        elif label == "interest":
            facts.append(f"User interest: {value}")
        elif label == "preference":
            facts.append(f"User preference: {value}")
        elif label == "school":
            facts.append(f"User school: {value}")
        elif label == "program":
            facts.append(f"User program: {value or 'gifted/AP/IB'}")
        elif label == "goal":
            facts.append(f"User goal: {value}")
        else:
            facts.append(f"User asked to remember: {value}")

    if lowered.startswith(("my name is ", "i am ", "i'm ", "im ")) and "grade" not in lowered:
        facts.append(normalized)

    return facts


def observe_turn(session_id: str, user_message: str, assistant_reply: str, agent: str | None = None) -> None:
    memory = _get_memory(session_id)

    extracted_facts = _extract_facts_from_text(user_message)
    for fact in extracted_facts:
        _add_fact(memory, fact)

    if extracted_facts:
        memory["summary"] = "; ".join(str(item) for item in memory.get("facts", [])[-8:])

    if agent:
        _add_example(memory, f"[{agent}] {user_message} -> {assistant_reply}")
    else:
        _add_example(memory, f"{user_message} -> {assistant_reply}")

    save_memory(session_id)


def get_memory_context(session_id: str | None = None, limit_facts: int = 12, limit_examples: int = 6) -> str:
    memory = _get_memory(session_id)
    facts = memory.get("facts", [])[-limit_facts:]
    examples = memory.get("examples", [])[-limit_examples:]

    sections: list[str] = []
    if memory.get("summary"):
        sections.append(f"Summary: {memory['summary']}")
    if facts:
        sections.append("Known facts:\n- " + "\n- ".join(str(item) for item in facts))
    if examples:
        sections.append("Recent memory examples:\n- " + "\n- ".join(str(item) for item in examples))

    return "\n\n".join(sections)
