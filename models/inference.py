"""
models/inference.py
Loads the trained MentorMatcher and scores mentor candidates for a given mentee.
This is the bridge between RAG (retrieval) and your model (ranking).
"""

import torch
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train import MentorMatcher


def load_model(model_path: str = "models/matcher.pt"):
    checkpoint = torch.load(model_path, map_location="cpu")
    model      = MentorMatcher(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_vocab(vocab_path: str = "models/vocab.json") -> dict:
    with open(vocab_path) as f:
        return json.load(f)


def encode_subject(subject: str, subjects: list) -> list[float]:
    vec = [0.0] * len(subjects)
    if subject in subjects:
        vec[subjects.index(subject)] = 1.0
    return vec


def encode_qual(qual: str, quals: list) -> list[float]:
    vec = [0.0] * len(quals)
    if qual in quals:
        vec[quals.index(qual)] = 1.0
    return vec


def encode_grade(grade: int) -> float:
    return (int(grade) - 9) / 3.0


def build_feature_vector(mentor: dict, mentee: dict, vocab: dict) -> list[float]:
    subjects = vocab["subjects"]
    quals    = vocab["qualifications"]

    mentor_subj_vec = encode_subject(mentor["subject"], subjects)
    mentor_grade    = encode_grade(mentor["grade"])
    mentor_qual_vec = encode_qual(mentor["qualifications"], quals)

    mentee_subj_vec = encode_subject(mentee["subject"], subjects)
    mentee_grade    = encode_grade(mentee["grade"])

    subject_match = 1.0 if mentor["subject"] == mentee["subject"] else 0.0
    grade_diff    = abs(int(mentor["grade"]) - int(mentee["grade"])) / 3.0
    mentor_senior = 1.0 if int(mentor["grade"]) > int(mentee["grade"]) else 0.0

    return (
        mentor_subj_vec +
        [mentor_grade] +
        mentor_qual_vec +
        mentee_subj_vec +
        [mentee_grade] +
        [subject_match, grade_diff, mentor_senior]
    )


def score_candidates(mentee: dict, candidates: list[dict]) -> list[dict]:
    """
    Score each RAG-retrieved mentor candidate using the trained model.
    Returns candidates sorted by model score descending.

    mentee:     { name, grade, subject }
    candidates: list of mentor dicts from the RAG retriever
    """
    model = load_model()
    vocab = load_vocab()

    scored = []
    for mentor in candidates:
        features = build_feature_vector(mentor, mentee, vocab)
        x        = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            score = model(x).item()

        # Generate a simple rule-based explanation
        reasons = []
        if mentor["subject"] == mentee["subject"]:
            reasons.append(f"teaches {mentee['subject']}")
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
