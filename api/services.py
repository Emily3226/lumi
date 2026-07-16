"""Shared database, matching, and booking helpers (MongoDB-backed)."""

from __future__ import annotations

import csv
import os
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from api.db import get_db, next_id, ensure_indexes
from models.inference import score_candidates
from rag.retriever import MentorRetriever
from rag.langchain_matcher import rank_candidates_langchain
from rag.subject_utils import expand_query_text, subject_key, subject_matches
from models.inference import trained_model_available

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

_retriever: MentorRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> MentorRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:  # re-check inside the lock
                _retriever = MentorRetriever()
    return _retriever


MIN_MATCH_SCORE = 0.35
MAX_ALLOWED_BELOW_GRADE = 1


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    """Ensure indexes exist and seed mentors from data/pairings.csv if empty."""
    db = get_db()
    ensure_indexes()

    if db["mentors"].count_documents({}) == 0:
        csv_path = os.path.join(ROOT_DIR, "data", "pairings.csv")
        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                seen: set[str] = set()
                docs = []
                for row in reader:
                    name = row.get("mentor_name")
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    docs.append({
                        "name": name,
                        "grade": int(row.get("mentor_grade") or 0),
                        "qualifications": row.get("mentor_qualifications") or "",
                        "subject": row.get("mentor_subject") or "",
                        "available": 1,
                    })
                if docs:
                    for d in docs:
                        db["mentors"].update_one(
                            {"name": d["name"]}, {"$setOnInsert": d}, upsert=True
                        )
        except FileNotFoundError:
            pass


def list_available_mentors() -> list[dict]:
    db = get_db()
    rows = list(
        db["mentors"]
        .find({}, {"_id": 0, "name": 1, "grade": 1, "qualifications": 1, "subject": 1, "available": 1})
        .sort([("available", -1), ("name", 1)])
    )
    return rows


def match_mentors(
    mentee_name: str,
    query_text: str,
    mentee_grade: int | None = None,
    top_k: int = 5,
) -> tuple[dict, list[dict]]:
    grade_value = int(mentee_grade or 0)
    query = expand_query_text(query_text)
    subject_hint = subject_key(query_text)

    # Increase retriever recall to give rerankers more options
    candidates = get_retriever().retrieve(query, grade_value, top_k=max(top_k * 3, 10))
    mentee = {
        "name": mentee_name,
        "grade": grade_value,
        "subject": subject_hint or query_text.strip(),
        "subject_hint": subject_hint,
        "query_text": query_text.strip(),
    }
    langchain_ranked = None
    try:
        langchain_ranked = rank_candidates_langchain(mentee, candidates, top_k=top_k)
    except Exception:
        langchain_ranked = None
    if langchain_ranked:
        logger.info("Using LangChain reranker for query: %s", query_text)
        strict = trained_model_available()
        ranked = score_candidates(mentee, langchain_ranked, strict=strict)
    else:
        strict = trained_model_available()
        ranked = score_candidates(mentee, candidates, strict=strict)

    def _passes_hard_filters(item: dict) -> bool:
        mentor_grade = int(item.get("grade") or 0)
        if grade_value > 0 and mentor_grade > 0 and mentor_grade < grade_value - MAX_ALLOWED_BELOW_GRADE:
            return False
        if subject_hint and not subject_matches(item.get("subject"), subject_hint):
            return False
        return True

    hard_filtered = [item for item in ranked if _passes_hard_filters(item)]
    if hard_filtered:
        ranked = hard_filtered

    if subject_hint:
        def _subject_priority(item):
            try:
                from rag.subject_utils import subject_matches as _sm
                mentor_grade = int(item.get("grade") or 0)
                grade_gap = abs(mentor_grade - grade_value) if grade_value > 0 and mentor_grade > 0 else 99
                below_mentee = 1 if grade_value > 0 and mentor_grade > 0 and mentor_grade < grade_value else 0

                return (
                    0 if _sm(item.get("subject"), subject_hint) else 1,
                    below_mentee,
                    grade_gap,
                    -item.get("match_score", 0.0),
                )
            except Exception:
                return (1, -item.get("match_score", 0.0))

        ranked.sort(key=_subject_priority)

    ranked = [item for item in ranked if float(item.get("match_score", 0.0)) >= MIN_MATCH_SCORE]

    return mentee, ranked


def match_mentors_debug(
    mentee_name: str,
    query_text: str,
    mentee_grade: int | None = None,
    top_k: int = 5,
) -> dict:
    """Return detailed matching info for debugging and A/B comparisons."""
    grade_value = int(mentee_grade or 0)
    query = expand_query_text(query_text)
    subject_hint = subject_key(query_text)

    raw_candidates = get_retriever().retrieve(query, grade_value, top_k=max(top_k * 4, 20))

    langchain_ranked = None
    try:
        langchain_ranked = rank_candidates_langchain(
            {"name": mentee_name, "grade": grade_value, "subject": subject_hint or query_text},
            raw_candidates,
            top_k=top_k,
        )
    except Exception:
        langchain_ranked = None

    langchain_used = bool(langchain_ranked)
    used_candidates = langchain_ranked if langchain_ranked else raw_candidates

    strict = trained_model_available()
    mentee_obj = {"name": mentee_name, "grade": grade_value, "subject": subject_hint or query_text, "subject_hint": subject_hint}
    final_ranked = score_candidates(mentee_obj, used_candidates, strict=strict)

    return {
        "mentee": {"name": mentee_name, "grade": grade_value, "subject": subject_hint or query_text, "subject_hint": subject_hint},
        "matches": final_ranked,
        "raw_candidates": raw_candidates,
        "langchain_used": langchain_used,
        "trained_model_loaded": strict,
    }


def book_pairing_in_db(
    mentor_name: str,
    mentee_name: str,
    subject: str,
    mentor_grade: int,
    mentee_grade: int,
    match_score: float,
    explanation: str,
    mentee_email: str = "",
    slot_id: int | None = None,
    slot_label: str = "",
) -> None:
    db = get_db()
    mentor = db["mentors"].find_one({"name": mentor_name})
    if not mentor:
        raise ValueError("Mentor not found")
    if not mentor.get("available"):
        raise ValueError("Mentor already booked")

    if slot_id is not None:
        slot = db["mentor_timeslots"].find_one({"id": slot_id, "mentor_name": mentor_name})
        if not slot:
            raise ValueError("Time slot not found for this mentor")
        if not slot.get("available"):
            raise ValueError("That time slot is already taken")

    booking_id = next_id("bookings")
    db["bookings"].insert_one({
        "id": booking_id,
        "mentor_name": mentor_name,
        "mentee_name": mentee_name,
        "subject": subject,
        "mentor_grade": mentor_grade,
        "mentee_grade": mentee_grade,
        "match_score": match_score,
        "explanation": explanation,
        "created_at": _now_str(),
        "status": "active",
        "mentee_email": mentee_email,
        "slot_id": slot_id,
        "slot_label": slot_label or "",
    })
    db["mentors"].update_one({"name": mentor_name}, {"$set": {"available": 0}})
    if slot_id is not None:
        db["mentor_timeslots"].update_one({"id": slot_id}, {"$set": {"available": 0}})
    db["mentees"].update_one(
        {"name": mentee_name},
        {"$setOnInsert": {"name": mentee_name, "grade": mentee_grade, "subject": subject}},
        upsert=True,
    )


# ── Time slot helpers ─────────────────────────────────────────────────────────

def get_mentor_slots(mentor_name: str, only_available: bool = True) -> list[dict]:
    """Return weekly time slots for a mentor, optionally filtered to only free ones.
    available=1 means the slot is free; available=0 means it is booked.
    """
    DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    db = get_db()
    query: dict = {"mentor_name": mentor_name}
    if only_available:
        query["available"] = 1
    rows = list(
        db["mentor_timeslots"]
        .find(query, {"_id": 0})
        .sort("start_time", 1)
    )
    rows.sort(key=lambda r: (DAYS_ORDER.index(r["day_of_week"]) if r["day_of_week"] in DAYS_ORDER else 99, r["start_time"]))
    return rows


def add_mentor_slot(mentor_name: str, day_of_week: str, start_time: str) -> dict:
    """Add a recurring weekly 1-hour slot for a mentor.
    day_of_week: e.g. 'Monday', start_time: 'HH:MM' (24-hour)."""
    from datetime import timedelta

    VALID_DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    if day_of_week not in VALID_DAYS:
        raise ValueError(f"day_of_week must be one of {sorted(VALID_DAYS)}")

    start_dt = datetime.strptime(start_time, "%H:%M")
    end_dt = start_dt + timedelta(hours=1)
    end_time = end_dt.strftime("%H:%M")

    db = get_db()
    if not db["mentors"].find_one({"name": mentor_name}):
        raise ValueError("Mentor not found")

    slot_id = next_id("mentor_timeslots")
    doc = {
        "id": slot_id,
        "mentor_name": mentor_name,
        "day_of_week": day_of_week,
        "start_time": start_time,
        "end_time": end_time,
        "available": 1,
    }
    db["mentor_timeslots"].insert_one(doc)
    return {"id": slot_id, "mentor_name": mentor_name, "day_of_week": day_of_week, "start_time": start_time, "end_time": end_time, "available": True}


def delete_mentor_slot(slot_id: int) -> bool:
    """Delete a time slot by id. Returns True if deleted, False if not found."""
    db = get_db()
    result = db["mentor_timeslots"].delete_one({"id": slot_id})
    return result.deleted_count > 0
