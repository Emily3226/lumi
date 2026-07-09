"""
scripts/migrate_to_neon.py

One-time migration: copies the schema and data from the local SQLite
database (data/lumi.db) into a Neon Postgres database.

Usage:
    1. Make sure DATABASE_URL is set in your .env file at the project root
       (it will be loaded automatically), or export it directly:

         export DATABASE_URL="postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require"

       (On Windows PowerShell: $env:DATABASE_URL = "postgresql://...")

    2. Run from the project root:

         python scripts/migrate_to_neon.py

This is safe to re-run: tables are created with IF NOT EXISTS, and rows
are inserted with ON CONFLICT DO NOTHING for tables with primary keys,
so re-running won't duplicate data. It will NOT overwrite existing rows
in Postgres if they already exist.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
SQLITE_PATH = ROOT / "data" / "lumi.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.env import load_dotenv_once

load_dotenv_once()

DATABASE_URL = os.environ.get("DATABASE_URL")

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS mentors (
        name           TEXT PRIMARY KEY,
        grade          INTEGER,
        qualifications TEXT,
        subject        TEXT,
        available      INTEGER DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mentees (
        name    TEXT PRIMARY KEY,
        grade   INTEGER,
        subject TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bookings (
        id           SERIAL PRIMARY KEY,
        mentor_name  TEXT NOT NULL,
        mentee_name  TEXT NOT NULL,
        subject      TEXT NOT NULL,
        mentor_grade INTEGER,
        mentee_grade INTEGER,
        match_score  REAL,
        explanation  TEXT,
        created_at   TEXT DEFAULT (to_char(now(), 'YYYY-MM-DD HH24:MI:SS')),
        status       TEXT DEFAULT 'active',
        mentee_email TEXT,
        slot_id      INTEGER,
        slot_label   TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mentor_timeslots (
        id          SERIAL PRIMARY KEY,
        mentor_name TEXT NOT NULL,
        day_of_week TEXT NOT NULL,
        start_time  TEXT NOT NULL,
        end_time    TEXT NOT NULL,
        available   INTEGER DEFAULT 1,
        FOREIGN KEY (mentor_name) REFERENCES mentors(name)
    )
    """,
]

# (table, columns, conflict_target) - order matters: mentors/mentees before
# bookings/mentor_timeslots because of the foreign key.
TABLES = [
    ("mentors", ["name", "grade", "qualifications", "subject", "available"], "name"),
    ("mentees", ["name", "grade", "subject"], "name"),
    (
        "bookings",
        [
            "id", "mentor_name", "mentee_name", "subject", "mentor_grade",
            "mentee_grade", "match_score", "explanation", "created_at",
            "status", "mentee_email", "slot_id", "slot_label",
        ],
        "id",
    ),
    (
        "mentor_timeslots",
        ["id", "mentor_name", "day_of_week", "start_time", "end_time", "available"],
        "id",
    ),
]


def main() -> None:
    if not DATABASE_URL:
        raise SystemExit(
            "DATABASE_URL is not set. Export your Neon connection string first, e.g.\n"
            '  export DATABASE_URL="postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require"'
        )
    if not SQLITE_PATH.exists():
        raise SystemExit(f"SQLite database not found at {SQLITE_PATH}")

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    pg_conn = psycopg2.connect(DATABASE_URL)
    pg_cur = pg_conn.cursor()

    print("Creating schema on Neon (if not already present)...")
    for stmt in SCHEMA_STATEMENTS:
        pg_cur.execute(stmt)
    pg_conn.commit()

    for table, columns, conflict_col in TABLES:
        try:
            rows = sqlite_conn.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
        except sqlite3.OperationalError as exc:
            print(f"Skipping {table}: {exc}")
            continue

        if not rows:
            print(f"{table}: no rows to migrate")
            continue

        placeholders = ", ".join(["%s"] * len(columns))
        col_list = ", ".join(columns)
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_col}) DO NOTHING"
        )
        for row in rows:
            pg_cur.execute(sql, tuple(row))
        pg_conn.commit()
        print(f"{table}: migrated {len(rows)} row(s)")

    # Make sure the bookings/mentor_timeslots SERIAL sequences continue
    # from the max id we just inserted, instead of restarting at 1.
    for table in ("bookings", "mentor_timeslots"):
        pg_cur.execute(
            f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
            f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
        )
    pg_conn.commit()

    sqlite_conn.close()
    pg_cur.close()
    pg_conn.close()
    print("Done. Your Neon database now mirrors data/lumi.db.")


if __name__ == "__main__":
    main()