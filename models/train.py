"""
models/train.py
Trains a mentor matcher model from cleaned historical pairings.

Primary training source:
- `historical_pairings` table in `data/training.db`

Fallback:
- `bookings` table, for backwards compatibility with live app bookings.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def get_db_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training.db")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _split_subjects(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [item.strip().lower() for item in value.replace(";", ",").split(",")]
    return [item for item in parts if item]


def _subject_overlap(mentor_subjects: str | None, mentee_subjects: str | None) -> float:
    mentor_set = set(_split_subjects(mentor_subjects))
    mentee_set = set(_split_subjects(mentee_subjects))
    if not mentor_set or not mentee_set:
        return 0.0
    return 1.0 if mentor_set.intersection(mentee_set) else 0.0


def _load_from_historical_pairings(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    if not _table_exists(conn, "historical_pairings"):
        return None, None

    rows = conn.execute(
        """
        SELECT mentor_subjects, mentee_subjects, mentor_grade, mentee_grade,
               grade_gap, match_score
        FROM historical_pairings
        WHERE match_score IS NOT NULL
        """
    ).fetchall()

    if not rows:
        return None, None

    features = []
    targets = []
    for row in rows:
        mentor_grade = float(row["mentor_grade"] or 0)
        mentee_grade = float(row["mentee_grade"] or 0)
        grade_gap = float(row["grade_gap"] if row["grade_gap"] is not None else abs(mentor_grade - mentee_grade))
        subject_match = _subject_overlap(row["mentor_subjects"], row["mentee_subjects"])
        senior_bonus = 1.0 if mentor_grade > mentee_grade else 0.0
        grade_similarity = max(0.0, 1.0 - (grade_gap / 4.0))

        features.append([subject_match, grade_gap, senior_bonus, grade_similarity])
        targets.append(float(row["match_score"]))

    return np.array(features, dtype=float), np.array(targets, dtype=float)


def _load_from_bookings(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    if not _table_exists(conn, "bookings"):
        return None, None

    rows = conn.execute(
        """
        SELECT mentor_name, mentee_name, subject,
               mentor_grade, mentee_grade, match_score
        FROM bookings
        WHERE status = 'active' AND match_score IS NOT NULL
        """
    ).fetchall()

    if not rows:
        return None, None

    features = []
    targets = []
    for row in rows:
        mentor_grade = float(row["mentor_grade"] or 0)
        mentee_grade = float(row["mentee_grade"] or 0)
        grade_gap = abs(mentor_grade - mentee_grade)
        subject_match = 1.0
        senior_bonus = 1.0 if mentor_grade > mentee_grade else 0.0
        grade_similarity = max(0.0, 1.0 - (grade_gap / 4.0))

        features.append([subject_match, grade_gap, senior_bonus, grade_similarity])
        targets.append(float(row["match_score"]))

    return np.array(features, dtype=float), np.array(targets, dtype=float)


def load_training_data() -> Tuple[np.ndarray, np.ndarray]:
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print("⚠ Database not found. Cannot train model.")
        return None, None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    data = _load_from_historical_pairings(conn)
    if data[0] is None:
        data = _load_from_bookings(conn)

    conn.close()
    return data


def train_model() -> Optional[dict]:
    print("Loading training data...")
    X, y = load_training_data()

    if X is None or len(X) < 2:
        print("⚠ Insufficient training data (need at least 2 samples). Using heuristic fallback.")
        return None

    print(f"Training on {len(X)} pairing examples...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.08,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_scaled, y)

    model_path = os.path.join(os.path.dirname(__file__), "mentor_matcher.pkl")
    joblib.dump({"scaler": scaler, "model": model}, model_path)

    train_r2 = model.score(X_scaled, y)
    print("✓ Model trained successfully!")
    print(f"  Training R² score: {train_r2:.3f}")
    print(f"  Model saved to: {model_path}")

    return {"scaler": scaler, "model": model}


if __name__ == "__main__":
    train_model()
