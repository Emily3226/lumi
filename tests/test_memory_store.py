"""api/memory_store.py - MongoDB-backed per-user memory persistence."""

from __future__ import annotations

import api.memory_store as ms


def test_load_memory_returns_defaults_when_missing(mongo_db):
    memory = ms.load_memory("nobody")
    assert memory == {"summary": "", "facts": [], "examples": [], "updated_at": ""}


def test_observe_turn_persists_extracted_facts(mongo_db):
    """Regression test for a real bug: save_memory() used to call
    _get_memory() internally, which unconditionally reloaded from storage -
    discarding whatever observe_turn() had just mutated in memory before the
    save actually happened. Every extracted fact/example was silently lost;
    reloading a session's memory always showed the stale pre-turn state.
    """
    ms.observe_turn(
        "m1",
        "Hi, my name is Priya and I am in grade 9",
        "Nice to meet you, Priya!",
        agent="general",
    )

    reloaded = ms.load_memory("m1")
    assert any("Priya" in fact for fact in reloaded["facts"])
    assert any("grade: 9" in fact.lower() for fact in reloaded["facts"])
    assert reloaded["examples"]
    assert reloaded["updated_at"]


def test_observe_turn_accumulates_across_multiple_turns(mongo_db):
    ms.observe_turn("m2", "I like chemistry", "Great, chemistry it is.")
    ms.observe_turn("m2", "I enjoy robotics", "Robotics is awesome.")

    reloaded = ms.load_memory("m2")
    assert len(reloaded["examples"]) == 2
    assert any("chemistry" in fact.lower() for fact in reloaded["facts"])
    assert any("robotics" in fact.lower() for fact in reloaded["facts"])


def test_clear_session_memory_resets_and_persists(mongo_db):
    ms.observe_turn("m3", "my name is Sam", "hi Sam")
    assert ms.load_memory("m3")["facts"]

    ms.clear_session_memory("m3")

    reloaded = ms.load_memory("m3")
    assert reloaded["facts"] == []
    assert reloaded["examples"] == []


def test_clear_session_memory_never_touches_other_sessions(mongo_db):
    ms.observe_turn("m4", "my name is Alex", "hi Alex")
    ms.observe_turn("m5", "my name is Jordan", "hi Jordan")

    ms.clear_session_memory("m4")

    assert ms.load_memory("m4")["facts"] == []
    assert any("Jordan" in fact for fact in ms.load_memory("m5")["facts"])


def test_get_memory_context_formats_facts_and_examples(mongo_db):
    ms.observe_turn("m6", "my name is Riley", "hi Riley")
    context = ms.get_memory_context("m6")
    assert "Riley" in context
    assert "Known facts" in context
    assert "Recent memory examples" in context


def test_get_memory_context_empty_when_no_memory(mongo_db):
    assert ms.get_memory_context("brand-new-session") == ""


def test_none_session_id_uses_legacy_fallback_key(mongo_db):
    ms.observe_turn(None, "my name is Legacy", "hi Legacy")  # type: ignore[arg-type]
    reloaded = ms.load_memory(None)
    assert any("Legacy" in fact for fact in reloaded["facts"])
