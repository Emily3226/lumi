"""Shared pytest fixtures.

Two kinds of tests live under tests/:

- Unit tests (the default): use the `mongo_db` fixture below, which points
  every module's `api.db.get_db()` at an in-memory mongomock database instead
  of your real Atlas cluster. Fast, no network, safe to run anytime.
- Integration tests (@pytest.mark.integration): exercise real MongoDB Atlas
  and/or Gemini using whatever is in your .env. They're skipped automatically
  if MONGODB_URI / GEMINI_API_KEY aren't set, so the suite still runs cleanly
  on a machine (or CI) with no secrets configured.
"""

from __future__ import annotations

import os

import mongomock
import pytest

import api.db as db_module
from api.env import load_dotenv_once

load_dotenv_once()

HAS_MONGODB = bool(os.environ.get("MONGODB_URI", "").strip())
HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY", "").strip())
HAS_ADMIN_KEY = bool(os.environ.get("ADMIN_API_KEY", "").strip())

requires_mongodb = pytest.mark.skipif(not HAS_MONGODB, reason="MONGODB_URI not configured")
requires_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="GEMINI_API_KEY not configured")
requires_admin_key = pytest.mark.skipif(not HAS_ADMIN_KEY, reason="ADMIN_API_KEY not configured")


@pytest.fixture
def mongo_db(monkeypatch):
    """An isolated in-memory MongoDB (mongomock) - no network, no real Atlas.

    Patches api.db's module-level singleton. Every module that already did
    `from api.db import get_db` (session_store, memory_store, retriever,
    admin, services, ...) transparently gets the fake database too, because
    they all call the *same* function object, which reads api.db's globals.
    """
    client = mongomock.MongoClient()
    fake_db = client["lumi_test"]
    monkeypatch.setattr(db_module, "_client", client)
    monkeypatch.setattr(db_module, "_db", fake_db)
    return fake_db
