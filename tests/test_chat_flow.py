"""End-to-end /chat and /admin tests against the real app.

These are integration tests: they boot the actual FastAPI app and talk to
your real MongoDB Atlas cluster and Gemini key (from .env). They're skipped
automatically if those aren't configured. Every session/memory document a
test creates is cleaned up afterward via a unique, clearly-marked session_id
prefix - nothing here touches your real mentor/mentee/booking data.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from tests.conftest import HAS_ADMIN_KEY, requires_gemini, requires_mongodb


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


@pytest.fixture
def session_id():
    sid = f"pytest_{uuid.uuid4().hex[:12]}"
    yield sid
    # Clean up whatever this test wrote for this session, real Mongo included.
    from api.db import get_db

    get_db()["chat_sessions"].delete_one({"_id": sid})
    get_db()["user_memory"].delete_one({"_id": sid})


@requires_mongodb
@requires_gemini
def test_general_chat_returns_a_reply(client, session_id):
    r = client.post("/chat", json={"session_id": session_id, "message": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"]
    assert body["active_agent"] == "general"


@requires_mongodb
@requires_gemini
def test_match_flow_returns_ranked_mentors(client, session_id):
    client.post("/chat", json={"session_id": session_id, "message": "switch to match agent"})
    client.post("/chat", json={"session_id": session_id, "message": "9"})
    r = client.post("/chat", json={"session_id": session_id, "message": "math"})

    assert r.status_code == 200
    body = r.json()
    assert body["matches"]
    for mentor in body["matches"]:
        assert "name" in mentor
        assert "similarity_score" in mentor


@requires_mongodb
@requires_gemini
def test_slot_list_survives_to_the_next_turn(client, session_id):
    """End-to-end regression test for the session_store bug: picking a
    mentor used to show a slot list that vanished on the very next request
    ("(No slots available)"), because session_store only persisted a fixed
    field allowlist and silently dropped `pending_slots`. Each /chat call
    here is a fully separate HTTP request, so this only passes if session
    state genuinely round-trips through MongoDB between requests.
    """
    client.post("/chat", json={"session_id": session_id, "message": "switch to match agent"})
    client.post("/chat", json={"session_id": session_id, "message": "9"})
    match_resp = client.post("/chat", json={"session_id": session_id, "message": "math"})
    assert match_resp.json()["matches"], "need at least one match for this test to mean anything"

    book_resp = client.post("/chat", json={"session_id": session_id, "message": "book 1"})
    reply = book_resp.json()["reply"]

    if "No slots available" in reply:
        pytest.skip("matched mentor has no configured time slots in this environment")

    # This is the exact request boundary that used to lose pending_slots.
    slot_resp = client.post("/chat", json={"session_id": session_id, "message": "1"})
    assert "No slots available" not in slot_resp.json()["reply"]


@requires_mongodb
def test_admin_rejects_missing_key(client):
    r = client.get("/admin/mentors")
    assert r.status_code == 401


@pytest.mark.skipif(not HAS_ADMIN_KEY, reason="ADMIN_API_KEY not configured")
@requires_mongodb
def test_admin_accepts_configured_key(client):
    import os

    r = client.get("/admin/mentors", headers={"X-Admin-Key": os.environ["ADMIN_API_KEY"]})
    assert r.status_code == 200
    assert isinstance(r.json(), list)
