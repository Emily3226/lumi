"""Shared database, matching, and booking helpers."""

from __future__ import annotations

import csv
import os
import sqlite3
import logging

logger = logging.getLogger(__name__)

from models.inference import score_candidates
from rag.retriever import MentorRetriever
from rag.langchain_matcher import rank_candidates_langchain
from rag.subject_utils import expand_query_text, subject_key
from models.inference import trained_model_available


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(ROOT_DIR, "data", "lumi.db")


retriever = MentorRetriever()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            mentor_name  TEXT NOT NULL,
            mentee_name  TEXT NOT NULL,
            subject      TEXT NOT NULL,
            mentor_grade INTEGER,
            mentee_grade INTEGER,
            match_score  REAL,
            explanation  TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mentors (
            name           TEXT PRIMARY KEY,
            grade          INTEGER,
            qualifications TEXT,
            subject        TEXT,
            available      INTEGER DEFAULT 1
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mentees (
            name    TEXT PRIMARY KEY,
            grade   INTEGER,
            subject TEXT
        )
        """
    )

    cur = conn.execute("SELECT COUNT(1) as c FROM mentors")
    if cur.fetchone()[0] == 0:
        csv_path = os.path.join(ROOT_DIR, "data", "pairings.csv")
        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                seen: set[str] = set()
                for row in reader:
                    name = row.get("mentor_name")
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    conn.execute(
                        "INSERT OR IGNORE INTO mentors (name, grade, qualifications, subject, available) VALUES (?, ?, ?, ?, 1)",
                        (
                            name,
                            int(row.get("mentor_grade") or 0),
                            row.get("mentor_qualifications") or "",
                            row.get("mentor_subject") or "",
                        ),
                    )
        except FileNotFoundError:
            pass

    cols = [r[1] for r in conn.execute("PRAGMA table_info(bookings)").fetchall()]
    if "status" not in cols:
        conn.execute("ALTER TABLE bookings ADD COLUMN status TEXT DEFAULT 'active'")

    conn.commit()
    conn.close()


def list_available_mentors() -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT name, grade, qualifications, subject, available FROM mentors ORDER BY available DESC, name ASC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


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
    candidates = retriever.retrieve(query, grade_value, top_k=max(top_k * 3, 10))
    mentee = {
        "name": mentee_name,
        "grade": grade_value,
        "subject": subject_hint or query_text.strip(),
        "subject_hint": subject_hint,
        "query_text": query_text.strip(),
    }
    # Try LangChain-based reranker first (optional). If unavailable, fall back to internal scorer.
    langchain_ranked = None
    try:
        langchain_ranked = rank_candidates_langchain(mentee, candidates, top_k=top_k)
    except Exception:
        langchain_ranked = None
    if langchain_ranked:
        logger.info("Using LangChain reranker for query: %s", query_text)
        # Even when LangChain produces an initial ranking, use the trained model
        # to compute final match scores so the trained program remains authoritative.
        strict = trained_model_available()
        ranked = score_candidates(mentee, langchain_ranked, strict=strict)
    else:
        strict = trained_model_available()
        ranked = score_candidates(mentee, candidates, strict=strict)

    # If the mentee provided a subject hint, ensure subject-matching mentors are prioritized
    if subject_hint:
        def _subject_priority(item):
            try:
                from rag.subject_utils import subject_matches as _sm

                return (0 if _sm(item.get("subject"), subject_hint) else 1, -item.get("match_score", 0.0))
            except Exception:
                return (1, -item.get("match_score", 0.0))

        ranked.sort(key=_subject_priority)

    return mentee, ranked


def match_mentors_debug(
    mentee_name: str,
    query_text: str,
    mentee_grade: int | None = None,
    top_k: int = 5,
) -> dict:
    """Return detailed matching info for debugging and A/B comparisons.

    Returns a dict with keys: mentee, matches, raw_candidates, langchain_used, trained_model_loaded
    """
    grade_value = int(mentee_grade or 0)
    query = expand_query_text(query_text)
    subject_hint = subject_key(query_text)

    # Pull more candidates to allow reranking
    raw_candidates = retriever.retrieve(query, grade_value, top_k=max(top_k * 4, 20))

    langchain_ranked = None
    try:
        langchain_ranked = rank_candidates_langchain({"name": mentee_name, "grade": grade_value, "subject": subject_hint or query_text}, raw_candidates, top_k=top_k)
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
) -> None:
    conn = get_db()
    cur = conn.execute("SELECT available FROM mentors WHERE name = ?", (mentor_name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError("Mentor not found")
    if row[0] == 0:
        conn.close()
        raise ValueError("Mentor already booked")

    conn.execute(
        """
        INSERT INTO bookings
          (mentor_name, mentee_name, subject, mentor_grade, mentee_grade, match_score, explanation, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
        """,
        (
            mentor_name,
            mentee_name,
            subject,
            mentor_grade,
            mentee_grade,
            match_score,
            explanation,
        ),
    )
    conn.execute("UPDATE mentors SET available = 0 WHERE name = ?", (mentor_name,))
    conn.execute(
        "INSERT OR IGNORE INTO mentees (name, grade, subject) VALUES (?, ?, ?)",
        (mentee_name, mentee_grade, subject),
    )
    conn.commit()
    conn.close()
