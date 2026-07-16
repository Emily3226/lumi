"""
Shared MongoDB Atlas connection helper.

Replaces the old Neon/Postgres connection (api/db.py used to wrap psycopg2).
Configure with the MONGODB_URI environment variable, e.g.:

    mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority

You get this connection string from the Atlas dashboard
(Database -> Connect -> Drivers). Database name defaults to "lumi" and can
be overridden with MONGODB_DB_NAME.

This module also exposes `next_id(name)`, a small atomic counter helper so
collections that used to rely on Postgres SERIAL/RETURNING id (bookings,
mentor_timeslots) keep plain integer ids - nothing downstream (frontend,
admin panel) has to change to deal with ObjectIds.
"""

from __future__ import annotations

import os
from typing import Any

from pymongo import MongoClient
from pymongo.database import Database

from api.env import load_dotenv_once

load_dotenv_once()

MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "lumi")

_client: MongoClient | None = None
_db: Database | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError(
                "MONGODB_URI environment variable is not set. "
                "Set it to your Atlas connection string, e.g. "
                "mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/"
            )
        # A small server-side connection pool is enough for a single small
        # instance; serverSelectionTimeoutMS keeps failures fast instead of
        # hanging a request for 30s if Atlas network access isn't configured.
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=8000,
            maxPoolSize=20,
        )
    return _client


def get_db() -> Database:
    """Return the shared MongoDB database handle."""
    global _db
    if _db is None:
        _db = get_client()[MONGODB_DB_NAME]
    return _db


def next_id(counter_name: str) -> int:
    """Atomically return the next integer id for `counter_name`.

    Mirrors Postgres SERIAL columns so bookings/mentor_timeslots keep small
    int ids instead of ObjectIds.
    """
    doc = get_db()["counters"].find_one_and_update(
        {"_id": counter_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,
    )
    return doc["seq"]


def ensure_indexes() -> None:
    """Create the indexes the app relies on. Safe to call repeatedly."""
    db = get_db()
    db["mentors"].create_index("name", unique=True)
    db["mentees"].create_index("name", unique=True)
    db["mentor_timeslots"].create_index("id", unique=True)
    db["mentor_timeslots"].create_index("mentor_name")
    db["bookings"].create_index("id", unique=True)
    db["bookings"].create_index("created_at")
    db["historical_pairings"].create_index("source_file")
