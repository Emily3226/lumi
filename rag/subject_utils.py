"""Shared subject helpers for retrieval and scoring."""

from __future__ import annotations

from typing import Optional


SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "math": (
        "mathematics",
        "algebra",
        "calculus",
        "geometry",
        "trigonometry",
        "statistics",
        "numeracy",
        "number theory",
    ),
    "physics": (
        "mechanics",
        "motion",
        "forces",
        "energy",
        "waves",
        "electromagnetism",
        "relativity",
    ),
    "chemistry": (
        "lab",
        "reaction",
        "reactions",
        "organic chemistry",
        "inorganic chemistry",
        "atoms",
        "molecules",
    ),
    "biology": (
        "life science",
        "living things",
        "cells",
        "genetics",
        "ecology",
        "anatomy",
        "human body",
    ),
    "english": (
        "writing",
        "essay",
        "essays",
        "grammar",
        "reading",
        "literature",
        "comprehension",
        "vocabulary",
    ),
}

# Additional keywords that don't neatly fit into aliases but indicate subjects
SUBJECT_KEYWORDS: dict[str, str] = {
    # math-related terms
    "deriv": "math",
    "derivative": "math",
    "derivatives": "math",
    "differentiat": "math",
    "chain rule": "math",
    "integral": "math",
    "integrals": "math",
    "limit": "math",
    "limits": "math",
    "calculus": "math",
}


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def subject_key(value: str | None) -> Optional[str]:
    """Return the canonical lower-case subject key if one is recognized."""

    haystack = _normalize_text(value)
    if not haystack:
        return None

    for key, aliases in SUBJECT_ALIASES.items():
        if key in haystack:
            return key
        for alias in aliases:
            if alias in haystack:
                return key
    # Check keyword map for smaller tokens or stems
    for kw, k in SUBJECT_KEYWORDS.items():
        if kw in haystack:
            return k
    return None


def subject_label(value: str | None) -> Optional[str]:
    key = subject_key(value)
    return key.capitalize() if key else None


def subject_matches(left: str | None, right: str | None) -> bool:
    left_key = subject_key(left) or _normalize_text(left)
    right_key = subject_key(right) or _normalize_text(right)
    return bool(left_key and right_key and left_key == right_key)


def expand_query_text(value: str | None) -> str:
    """Expand a freeform query with subject aliases to improve semantic retrieval."""

    query = (value or "").strip()
    if not query:
        return ""

    key = subject_key(query)
    if not key:
        return query

    aliases = ", ".join((key, *SUBJECT_ALIASES[key]))
    return f"{query}\nRelated subject area: {key}. Associated topics: {aliases}."
