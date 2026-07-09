from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# Each session gets its own memory file so one user's conversation can never
# wipe or bleed into another user's. Legacy single-file store (pre-fix) is
# kept only as a fallback default filename.
MEMORY_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "user_memory"
_LEGACY_MEMORY_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "user_memory.json"
MAX_FACTS = 60
MAX_EXAMPLES = 12

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_\-]")


def _default_memory() -> dict[str, Any]:
    return {
        "summary": "",
        "facts": [],
        "examples": [],
        "updated_at": "",
    }


def _memory_path(session_id: str | None) -> Path:
    if not session_id:
        return _LEGACY_MEMORY_STORE_PATH
    safe_id = _SAFE_ID_RE.sub("_", session_id)[:128]
    return MEMORY_STORE_DIR / f"{safe_id}.json"


def load_memory(session_id: str | None = None) -> dict[str, Any]:
    path = _memory_path(session_id)
    if not path.exists():
        return _default_memory()

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return _default_memory()

    if not isinstance(data, dict):
        return _default_memory()

    memory = _default_memory()
    memory.update(data)
    if not isinstance(memory.get("facts"), list):
        memory["facts"] = []
    if not isinstance(memory.get("examples"), list):
        memory["examples"] = []
    return memory


# In-memory cache of per-session memory dicts, keyed by session_id
# ("" used for the legacy/no-session fallback).
_memory_cache: dict[str, dict[str, Any]] = {}


def _get_memory(session_id: str | None) -> dict[str, Any]:
    key = session_id or ""
    if key not in _memory_cache:
        _memory_cache[key] = load_memory(session_id)
    return _memory_cache[key]


def save_memory(session_id: str | None = None) -> None:
    memory = _get_memory(session_id)
    path = _memory_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def clear_session_memory(session_id: str | None = None) -> None:
    """Reset ONE session's persistent memory to defaults.

    Intended to be called whenever that specific conversation/session is
    reset, so facts, summaries, and examples from its previous history don't
    leak into the new one. This only ever touches the given session_id's
    memory file - never any other user's.
    """
    key = session_id or ""
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
    memory["updated_at"] = session_id

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