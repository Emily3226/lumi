"""
scripts/migrate_sqlite_to_mongo.py

One-time migration: copies mentors / mentees / mentor_timeslots / bookings
straight out of the local data/lumi.db SQLite file into MongoDB Atlas.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import get_db, ensure_indexes

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lumi.db")


def _rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(f"SELECT * FROM {table}")
    return [dict(r) for r in cur.fetchall()]


def main() -> None:
    if not os.path.exists(DB_PATH):
        print(f"No SQLite file found at {DB_PATH} - nothing to migrate.")
        return

    conn = sqlite3.connect(DB_PATH)
    db = get_db()
    ensure_indexes()

    mentors = _rows(conn, "mentors")
    for m in mentors:
        db["mentors"].update_one({"name": m["name"]}, {"$set": m}, upsert=True)
    print(f"mentors: migrated {len(mentors)}")

    mentees = _rows(conn, "mentees")
    for m in mentees:
        db["mentees"].update_one({"name": m["name"]}, {"$set": m}, upsert=True)
    print(f"mentees: migrated {len(mentees)}")

    slots = _rows(conn, "mentor_timeslots")
    max_slot_id = 0
    for s in slots:
        max_slot_id = max(max_slot_id, s["id"])
        db["mentor_timeslots"].update_one({"id": s["id"]}, {"$set": s}, upsert=True)
    print(f"mentor_timeslots: migrated {len(slots)}")

    bookings = _rows(conn, "bookings")
    max_booking_id = 0
    for b in bookings:
        max_booking_id = max(max_booking_id, b["id"])
        db["bookings"].update_one({"id": b["id"]}, {"$set": b}, upsert=True)
    print(f"bookings: migrated {len(bookings)}")

    db["counters"].update_one({"_id": "mentor_timeslots"}, {"$max": {"seq": max_slot_id}}, upsert=True)
    db["counters"].update_one({"_id": "bookings"}, {"$max": {"seq": max_booking_id}}, upsert=True)

    conn.close()
    print("\nDone. Refresh the app - mentors and time slots should now show up.")


if __name__ == "__main__":
    main()
