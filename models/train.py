"""
models/train.py
Trains a mentor matcher model from cleaned historical pairings.

Training source:
- `historical_pairings` table in `data/training.db`
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rag.subject_utils import subject_matches


def get_db_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training.db")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _subject_match_feature(mentor_subjects: str | None, mentee_subjects: str | None) -> float:
    """Compute the subject-match feature the same way inference does.

    IMPORTANT: this must stay in sync with `_ml_score` in models/inference.py,
    which computes:

        subject_match = 1.0 if subject_matches(mentor_subject, mentee_subject) else 0.5

    Previously this trained on a strict 0.0/1.0 overlap of raw, uncanonicalized
    subject strings (e.g. "Calculus" vs "Math" never matched), while inference
    fed the StandardScaler values of 1.0 or 0.5 using `subject_matches`, which
    canonicalizes subjects via `subject_key` and recognizes aliases/related
    science subjects. That mismatch meant the scaler/model were trained on a
    feature distribution ({0.0, 1.0}) that almost never occurs at inference
    time ({0.5, 1.0}), which degraded the quality of the trained match scores.
    Using the exact same function here fixes that distribution mismatch.
    """
    return 1.0 if subject_matches(mentor_subjects, mentee_subjects) else 0.5


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
        subject_match = _subject_match_feature(row["mentor_subjects"], row["mentee_subjects"])
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

    conn.close()
    if data[0] is None:
        print("⚠ No training rows found in training.db historical_pairings table.")
    return data


def train_model() -> Optional[dict]:
    print("Loading training data...")
    X, y = load_training_data()

    if X is None or len(X) < 8:
        print("⚠ Insufficient training data (need at least 8 samples for validation). Using heuristic fallback.")
        return None

    print(f"Training on {len(X)} pairing examples...")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.08,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    model_path = os.path.join(os.path.dirname(__file__), "mentor_matcher.pkl")
    candidate_bundle = {
        "scaler": scaler,
        "model": model,
        "metrics": {
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
        },
    }

    # NOTE: the previous "keep existing model if its validation MAE is lower"
    # comparison is no longer a fair comparison once the feature definition
    # changes (the existing model was trained on the OLD subject_match feature
    # distribution). When retraining after this fix, force-replace the
    # existing model so it is trained on the corrected features.
    force_replace = os.environ.get("FORCE_RETRAIN", "1") == "1"

    best_bundle = None
    if not force_replace and os.path.exists(model_path):
        try:
            existing = joblib.load(model_path)
            if isinstance(existing, dict) and existing.get("scaler") is not None and existing.get("model") is not None:
                existing_val_pred = existing["model"].predict(existing["scaler"].transform(X_val))
                existing_val_mae = mean_absolute_error(y_val, existing_val_pred)
                existing_val_r2 = r2_score(y_val, existing_val_pred)
                # Prefer lower validation MAE; use validation R2 as tie-breaker.
                if (existing_val_mae < val_mae) or (abs(existing_val_mae - val_mae) < 1e-9 and existing_val_r2 >= val_r2):
                    best_bundle = existing
                    print("Keeping existing model: better validation performance.")
                    print(f"  Existing val MAE: {existing_val_mae:.4f}, val R²: {existing_val_r2:.4f}")
                    print(f"  Candidate val MAE: {val_mae:.4f}, val R²: {val_r2:.4f}")
        except Exception:
            pass

    if best_bundle is None:
        joblib.dump(candidate_bundle, model_path)
        best_bundle = candidate_bundle
        print("Saved new model based on validation performance.")

    print("✓ Model trained successfully!")
    print(f"  Train R²: {train_r2:.3f} | Val R²: {val_r2:.3f}")
    print(f"  Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f}")
    print(f"  Model saved to: {model_path}")

    return best_bundle


if __name__ == "__main__":
    train_model()