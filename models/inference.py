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
            print("Loaded trained mentor matcher model")
            return _MODEL_CACHE
        except Exception as e:
            print(f"Warning: failed to load trained model: {e}. Falling back to heuristic.")
            _MODEL_AVAILABLE = False
            return None
    return None


def _normalize_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _ml_score(mentee: dict, mentor: dict, trained_model: dict) -> float:
    """
    Score using the trained ML model.
    """
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
        print(f"Warning: ML scoring failed: {e}. Using heuristic.")
        return _heuristic_score(mentee, mentor)



def _heuristic_score(mentee: dict, mentor: dict) -> float:
    mentee_grade = int(mentee.get("grade", 0) or 0)
    mentor_grade = int(mentor.get("grade", 0) or 0)
    has_grade = mentee_grade > 0
    has_subject_hint = bool((mentee.get("subject_hint") or "").strip())

    subject_match = 1.0 if subject_matches(mentor.get("subject"), mentee.get("subject_hint") or mentee.get("subject")) else 0.0
    grade_gap = abs(mentor_grade - mentee_grade) if has_grade else 0
    senior_bonus = 1.0 if (has_grade and mentor_grade > mentee_grade) else 0.0
    qualification_bonus = 1.0 if mentor.get("qualifications") else 0.0
    similarity_bonus = float(mentor.get("similarity_score", 0.0))

    grade_component = max(0.0, 1.0 - (grade_gap / 3.0)) if has_grade else 0.0
    subject_weight = 0.45 if has_subject_hint else 0.10

    score = (
        subject_weight * subject_match
        + 0.20 * grade_component
        + 0.20 * senior_bonus
        + 0.10 * qualification_bonus
        + 0.05 * similarity_bonus
    )
    return _normalize_score(score)


def score_candidates(mentee: dict, candidates: list[dict], strict: bool = False) -> list[dict]:
    """
    Score each RAG-retrieved mentor candidate using the trained ML model.
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
        else:
            score = _heuristic_score(mentee, mentor)

        # Generate a simple rule-based explanation
        reasons = []
        if subject_matches(mentor["subject"], mentee.get("subject_hint") or mentee["subject"]):
            reasons.append(f"teaches {mentee.get('subject_hint') or mentee['subject']}")
        if int(mentor["grade"]) > int(mentee["grade"]):
            diff = int(mentor["grade"]) - int(mentee["grade"])
            reasons.append(f"{diff} grade{'s' if diff > 1 else ''} ahead")
        if mentor["qualifications"]:
            reasons.append(mentor["qualifications"])

        explanation = f"Good match: {', '.join(reasons)}." if reasons else "Potential match."

        scored.append({
            **mentor,
            "match_score": round(score, 3),
            "explanation": explanation,
        })

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored
