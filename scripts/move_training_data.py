"""
scripts/move_training_data.py

Move historical training data out of the live app database and into the
dedicated training database.

This copies `historical_pairings` from `data/lumi.db` to `data/training.db`
and removes the source table from `lumi.db` afterward.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
APP_DB_PATH = DATA_DIR / "lumi.db"
TRAINING_DB_PATH = DATA_DIR / "training.db"


def ensure_training_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_pairings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
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
            created_at          TEXT DEFAULT (datetime('now'))
        )
        """
    )


def move_training_data() -> None:
    if not APP_DB_PATH.exists():
        raise SystemExit(f"Live database not found: {APP_DB_PATH}")

    app_conn = sqlite3.connect(APP_DB_PATH)
    app_conn.row_factory = sqlite3.Row
    training_conn = sqlite3.connect(TRAINING_DB_PATH)
    training_conn.row_factory = sqlite3.Row

    try:
        ensure_training_table(training_conn)
        source_rows = []
        source_table_exists = app_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='historical_pairings'"
        ).fetchone()

        if source_table_exists:
            source_rows = app_conn.execute(
                """
                SELECT cycle, source_file, mentor_name, mentor_email, mentor_grade,
                       mentor_subjects, mentor_notes, mentee_name, mentee_email,
                       mentee_grade, mentee_subjects, subjects_satisfied,
                       subject_count, grade_gap, match_score, created_at
                FROM historical_pairings
                """
            ).fetchall()

        training_conn.execute("DELETE FROM historical_pairings")
        if source_rows:
            training_conn.executemany(
                """
                INSERT INTO historical_pairings
                    (cycle, source_file, mentor_name, mentor_email, mentor_grade,
                     mentor_subjects, mentor_notes, mentee_name, mentee_email,
                     mentee_grade, mentee_subjects, subjects_satisfied,
                     subject_count, grade_gap, match_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [tuple(row) for row in source_rows],
            )

        app_conn.execute("DROP TABLE IF EXISTS historical_pairings")
        app_conn.commit()
        training_conn.commit()

        print(f"Moved {len(source_rows)} historical pairing rows into {TRAINING_DB_PATH.name}")
        print("Removed historical_pairings from the live app database")
    finally:
        app_conn.close()
        training_conn.close()


if __name__ == "__main__":
    move_training_data()