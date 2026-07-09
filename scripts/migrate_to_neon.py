"""
scripts/migrate_training_to_neon.py

One-time migration: copies the `historical_pairings` table from the local
SQLite training database (data/training.db) into the same Neon Postgres
database used by the live app (see api/db.py), as its own table.

After this has been run once, models/train.py, models/clean_training_data.py,
and scripts/import_and_train.py all read/write `historical_pairings`
directly on Neon, so data/training.db is no longer needed.

Usage:
    1. Make sure DATABASE_URL is set in your .env file at the project root
       (it will be loaded automatically), or export it directly:

         export DATABASE_URL="postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require"

    2. Run from the project root:

         python scripts/migrate_training_to_neon.py

This is safe to re-run: the table is created with IF NOT EXISTS, and rows
are inserted with ON CONFLICT (id) DO NOTHING, so re-running won't
duplicate data.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
SQLITE_PATH = ROOT / "data" / "training.db"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.env import load_dotenv_once

load_dotenv_once()

DATABASE_URL = os.environ.get("DATABASE_URL")

SCHEMA_STATEMENT = """
    CREATE TABLE IF NOT EXISTS historical_pairings (
        id                  SERIAL PRIMARY KEY,
        cycle               TEXT,
        source_file         TEXT,
        mentor_name         TEXT,
        mentor_email        TEXT,
        mentor_grade        INTEGER,
        mentor_subjects     TEXT,
        mentor_notes        TEXT,
        mentee_name         TEXT,
        mentee_email        TEXT,
        mentee_grade        INTEGER,
        mentee_subjects     TEXT,
        subjects_satisfied  TEXT,
        subject_count       INTEGER,
        grade_gap           INTEGER,
        match_score         REAL,
        created_at          TEXT DEFAULT to_char(now(), 'YYYY-MM-DD HH24:MI:SS')
    )
"""

COLUMNS = [
    "id", "cycle", "source_file", "mentor_name", "mentor_email", "mentor_grade",
    "mentor_subjects", "mentor_notes", "mentee_name", "mentee_email",
    "mentee_grade", "mentee_subjects", "subjects_satisfied", "subject_count",
    "grade_gap", "match_score", "created_at",
]


def main() -> None:
    if not DATABASE_URL:
        raise SystemExit(
            "DATABASE_URL is not set. Export your Neon connection string first, e.g.\n"
            '  export DATABASE_URL="postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require"'
        )
    if not SQLITE_PATH.exists():
        print(f"No local training database found at {SQLITE_PATH} - nothing to migrate.")
        return

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    pg_conn = psycopg2.connect(DATABASE_URL)
    pg_cur = pg_conn.cursor()

    print("Creating historical_pairings table on Neon (if not already present)...")
    pg_cur.execute(SCHEMA_STATEMENT)
    pg_conn.commit()

    try:
        rows = sqlite_conn.execute(
            f"SELECT {', '.join(COLUMNS)} FROM historical_pairings"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"Skipping historical_pairings: {exc}")
        rows = []

    if not rows:
        print("historical_pairings: no rows to migrate")
    else:
        placeholders = ", ".join(["%s"] * len(COLUMNS))
        col_list = ", ".join(COLUMNS)
        sql = (
            f"INSERT INTO historical_pairings ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT (id) DO NOTHING"
        )
        for row in rows:
            pg_cur.execute(sql, tuple(row))
        pg_conn.commit()
        print(f"historical_pairings: migrated {len(rows)} row(s)")

    pg_cur.execute(
        "SELECT setval(pg_get_serial_sequence('historical_pairings', 'id'), "
        "COALESCE((SELECT MAX(id) FROM historical_pairings), 1))"
    )
    pg_conn.commit()

    sqlite_conn.close()
    pg_cur.close()
    pg_conn.close()

    print("Done. Neon now has your historical_pairings data.")
    print(
        f"Once you've confirmed the data looks right on Neon, {SQLITE_PATH} "
        "is safe to delete (it's already gitignored)."
    )


if __name__ == "__main__":
    main()