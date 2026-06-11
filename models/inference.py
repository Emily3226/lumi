"""
models/inference.py
Scores mentor candidates for a given mentee using a trained ML model,
with fallback to a heuristic if the model is not available.
"""

from __future__ import annotations
import os
import joblib
import numpy as np

from rag.subject_utils import subject_matches


# Try to load the trained model
_MODEL_CACHE = None
_MODEL_AVAILABLE = False
import logging

logger = logging.getLogger(__name__)
def _load_trained_model():
    """Load the trained mentor matcher model if it exists."""
    global _MODEL_CACHE, _MODEL_AVAILABLE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model_path = os.path.join(os.path.dirname(__file__), "mentor_matcher.pkl")
    if os.path.exists(model_path):
        try:
            _MODEL_CACHE = joblib.load(model_path)
            _MODEL_AVAILABLE = True
            logger.info("Loaded trained mentor matcher model from %s", model_path)
            return _MODEL_CACHE
        except Exception as e:
            logger.warning("Failed to load trained model: %s. Falling back to heuristic.", e)
            _MODEL_AVAILABLE = False
            return None
    return None


def _normalize_score(score: float) -> float:
    return max(0.0, min(1.0, score))


# Tunable weights (adjust to change matching behavior)
SUBJECT_WEIGHT_HINT = 0.60
SUBJECT_WEIGHT_NO_HINT = 0.15
GRADE_WEIGHT = 0.25
SENIOR_BONUS_WEIGHT = 0.10
QUALIFICATION_WEIGHT = 0.05
SIMILARITY_WEIGHT = 0.05
# Post-score adjustments
SUBJECT_MISMATCH_PENALTY = 0.7  # multiply score when subject_hint exists but mentor != subject
SUBJECT_MATCH_BOOST = 0.03      # additive boost when subject matches
BELOW_GRADE_PENALTY = 0.55      # multiply score when mentor grade is below mentee grade


def _ml_score(mentee: dict, mentor: dict, trained_model: dict) -> float:
    """Score using the trained ML model."""
    try:
        scaler = trained_model["scaler"]
        model = trained_model["model"]

        mentor_grade = float(mentor.get("grade", 0))
        mentee_grade = float(mentee.get("grade", 0))

        # Extract same features used in training
        subject_match = 1.0 if subject_matches(mentor.get("subject"), mentee.get("subject_hint") or mentee.get("subject")) else 0.5
        grade_gap = abs(mentor_grade - mentee_grade)
        senior = 1.0 if mentor_grade > mentee_grade else 0.0
        grade_similarity = max(0.0, 1.0 - (grade_gap / 4.0))

        features = np.array([[subject_match, grade_gap, senior, grade_similarity]])
        features_scaled = scaler.transform(features)
        score = float(model.predict(features_scaled)[0])

        return _normalize_score(score)
    except Exception as e:
        logger.warning("ML scoring failed: %s. Using heuristic.", e)
        return _heuristic_score(mentee, mentor)


def _heuristic_score(mentee: dict, mentor: dict) -> float:
    mentee_grade = int(mentee.get("grade", 0) or 0)
    mentor_grade = int(mentor.get("grade", 0) or 0)
    has_grade = mentee_grade > 0
    has_subject_hint = bool((mentee.get("subject_hint") or "").strip())

    subject_match = 1.0 if subject_matches(mentor.get("subject"), mentee.get("subject_hint") or mentee.get("subject")) else 0.5
    grade_gap = abs(mentor_grade - mentee_grade) if has_grade else 0
    senior_bonus = 1.0 if (has_grade and mentor_grade > mentee_grade) else 0.0
    qualification_bonus = 1.0 if mentor.get("qualifications") else 0.0
    similarity_bonus = float(mentor.get("similarity_score", 0.0))

    grade_component = max(0.0, 1.0 - (grade_gap / 3.0)) if has_grade else 0.0
    subject_weight = SUBJECT_WEIGHT_HINT if has_subject_hint else SUBJECT_WEIGHT_NO_HINT

    score = (
        subject_weight * subject_match
        + GRADE_WEIGHT * grade_component
        + SENIOR_BONUS_WEIGHT * senior_bonus
        + QUALIFICATION_WEIGHT * qualification_bonus
        + SIMILARITY_WEIGHT * similarity_bonus
    )
    return _normalize_score(score)


def score_candidates(mentee: dict, candidates: list[dict], strict: bool = False) -> list[dict]:
    """Score each RAG-retrieved mentor candidate using the trained ML model.
    If `strict` is True, raise when the trained model is not available instead
    of falling back to the heuristic scorer.
    Returns candidates sorted by score descending.

    mentee:     { name, grade, subject }
    candidates: list of mentor dicts from the RAG retriever
    """
    trained_model = _load_trained_model()
    if strict and not (trained_model and _MODEL_AVAILABLE):
        raise RuntimeError("Trained mentor matcher model is not available")

    scored = []
    for mentor in candidates:
        # Use trained model if available, otherwise use heuristic
        if trained_model and _MODEL_AVAILABLE:
            score = _ml_score(mentee, mentor, trained_model)
            logger.debug("Scored with trained model: mentee=%s mentor=%s score=%s", mentee.get("name"), mentor.get("name"), score)
        else:
            score = _heuristic_score(mentee, mentor)
            logger.debug("Scored with heuristic: mentee=%s mentor=%s score=%s", mentee.get("name"), mentor.get("name"), score)

        # Prepare explanation reasons and post-process to favor exact subject matches
        reasons = []
        has_subject_hint = bool((mentee.get("subject_hint") or "").strip())
        mentee_grade_val = int(mentee.get("grade", 0) or 0)
        mentor_grade_val = int(mentor.get("grade", 0) or 0)

        if mentee_grade_val > 0 and mentor_grade_val > 0 and mentor_grade_val < mentee_grade_val:
            score = score * BELOW_GRADE_PENALTY
            reasons.append("(lowered because mentor is below mentee grade)")

        if has_subject_hint and not subject_matches(mentor.get("subject"), mentee.get("subject_hint") or mentee.get("subject")):
            score = score * SUBJECT_MISMATCH_PENALTY
            reasons.append("(lowered for not matching mentee subject)")
        elif has_subject_hint and subject_matches(mentor.get("subject"), mentee.get("subject_hint") or mentee.get("subject")):
            score = min(1.0, score + SUBJECT_MATCH_BOOST)
        if subject_matches(mentor.get("subject"), mentee.get("subject_hint") or mentee.get("subject")):
            reasons.append(f"teaches {mentee.get('subject_hint') or mentee.get('subject')}")
        try:
            if int(mentor.get("grade", 0)) > int(mentee.get("grade", 0)):
                diff = int(mentor.get("grade", 0)) - int(mentee.get("grade", 0))
                reasons.append(f"{diff} grade{'s' if diff > 1 else ''} ahead")
        except Exception:
            pass
        if mentor.get("qualifications"):
            reasons.append(mentor.get("qualifications"))

        explanation = f"Good match: {', '.join(reasons)}." if reasons else "Potential match."

        scored.append({
            **mentor,
            "match_score": round(score, 3),
            "explanation": explanation,
        })

    # If the mentee has a subject hint, prioritize mentors who match that subject
    has_subject_hint_final = bool((mentee.get("subject_hint") or "").strip())
    if has_subject_hint_final:
        scored.sort(
            key=lambda x: (
                0 if subject_matches(x.get("subject"), mentee.get("subject_hint") or mentee.get("subject")) else 1,
                -x.get("match_score", 0.0),
            )
        )
    else:
        scored.sort(key=lambda x: x["match_score"], reverse=True)

    return scored


def trained_model_available() -> bool:
    """Return True if the trained mentor matcher model was successfully loaded."""
    # Ensure model load was attempted
    _load_trained_model()
    return bool(_MODEL_AVAILABLE)
