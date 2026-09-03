"""api/session_store.py - MongoDB-backed chat session persistence."""

from __future__ import annotations

import api.session_store as ss


def test_new_session_has_expected_defaults():
    s = ss.new_session()
    assert s["state"] == "idle"
    assert s["active_agent"] == "general"
    assert s["messages"] == []
    assert s["matches"] == []


def test_load_session_returns_fresh_default_when_missing(mongo_db):
    s = ss.load_session("does-not-exist-yet")
    assert s == ss.new_session()


def test_save_and_reload_round_trips_known_fields(mongo_db):
    session = ss.get_session("s1")
    session["state"] = "awaiting_match_details"
    session["grade"] = 9
    session["subject"] = "math"
    ss.save_session("s1")

    reloaded = ss.get_session("s1")
    assert reloaded["state"] == "awaiting_match_details"
    assert reloaded["grade"] == 9
    assert reloaded["subject"] == "math"


def test_arbitrary_extra_keys_survive_save_and_reload(mongo_db):
    """Regression test for a real bug: session_store originally only
    persisted a fixed allowlist of fields, so ad-hoc keys agents.py stashes
    on the session dict during booking (pending_slots, pending_slot_id,
    pending_mentee_email, ...) silently vanished on save. A mentee picking a
    time slot would see "(No slots available)" on the very next turn because
    pending_slots was always empty after reload.
    """
    session = ss.get_session("s2")
    session["pending_slots"] = [
        {"id": 1, "day_of_week": "Mondays", "start_time": "14:00", "end_time": "15:00"},
        {"id": 2, "day_of_week": "Tuesdays", "start_time": "15:00", "end_time": "16:00"},
    ]
    session["pending_slot_id"] = None
    session["pending_slot_label"] = None
    session["pending_mentee_email"] = "mentee@example.com"
    session["state"] = "awaiting_slot_selection"
    ss.save_session("s2")

    reloaded = ss.get_session("s2")
    assert reloaded["pending_slots"] == session["pending_slots"]
    assert reloaded["pending_mentee_email"] == "mentee@example.com"
    assert reloaded["state"] == "awaiting_slot_selection"


def test_reset_session_clears_state(mongo_db):
    session = ss.get_session("s3")
    session["state"] = "showing_results"
    session["grade"] = 10
    ss.save_session("s3")

    ss.reset_session("s3")

    reloaded = ss.get_session("s3")
    assert reloaded == ss.new_session()


def test_save_session_is_a_noop_if_never_loaded(mongo_db):
    # save_session() reads from the in-process cache; calling it for a
    # session_id that was never fetched via get_session() should not raise
    # or write anything.
    ss.save_session("never-touched")
    assert ss.load_session("never-touched") == ss.new_session()


def test_load_session_ignores_reserved_mongo_fields(mongo_db):
    mongo_db[ss.COLLECTION_NAME].insert_one({
        "_id": "s4",
        "state": "idle",
        "updated_at": "2026-01-01T00:00:00Z",
    })
    session = ss.load_session("s4")
    assert "_id" not in session
    assert "updated_at" not in session
