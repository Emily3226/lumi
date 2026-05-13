"""
api/chat.py
Conversational layer on top of your existing RAG + PyTorch matching pipeline.

Plug this into your existing main.py by adding:
    from api.chat import router as chat_router
    app.include_router(chat_router)

State machine flow:
    idle → ask_subject → ask_grade → ask_name → showing_results → done
"""

import re
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from rag.retriever import MentorRetriever
from models.inference import score_candidates

router = APIRouter()

# Single shared retriever (loaded once)
retriever = MentorRetriever()

# ── In-memory session store ───────────────────────────────────────────────────
# In production swap this for Redis or a DB table keyed by session_id
sessions: dict[str, dict] = {}

SUBJECTS  = ["math", "physics", "chemistry", "biology", "english"]
GRADES    = {"9": 9, "10": 10, "11": 11, "12": 12,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
             "grade 9": 9, "grade 10": 10, "grade 11": 11, "grade 12": 12}


# ── Intent detection ──────────────────────────────────────────────────────────

def detect_intent(text: str) -> str:
    t = text.lower().strip()
    if re.search(r"\bbook\b\s*(\d)", t):     return "book"
    if re.search(r"\bhistory\b|\bpast\b|\bprevious pairings\b", t): return "history"
    if re.search(r"\bhelp\b|\bwhat can you\b|\bcommands\b", t):     return "help"
    if re.search(r"\brestart\b|\bstart over\b|\bnew\b|\bagain\b", t): return "restart"
    if re.search(r"\bmatch\b|\bfind\b|\bmentor\b|\bpair\b|\blook", t): return "find_match"
    return "unknown"

def extract_subject(text: str) -> Optional[str]:
    t = text.lower()
    for s in SUBJECTS:
        if s in t:
            return s.capitalize()
    return None

def extract_grade(text: str) -> Optional[int]:
    t = text.lower().strip()
    for key, val in GRADES.items():
        if key in t:
            return val
    # bare number
    m = re.search(r"\b(9|10|11|12)\b", t)
    return int(m.group(1)) if m else None

def extract_book_choice(text: str) -> Optional[int]:
    m = re.search(r"\bbook\b\s*(\d)", text.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d)\b", text)
    return int(m.group(1)) if m else None


# ── Response builder ─────────────────────────────────────────────────────────

def format_matches(matches: list[dict]) -> str:
    lines = ["Here are the top matches:\n"]
    for i, m in enumerate(matches[:3], 1):
        pct = round(m["match_score"] * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        lines.append(
            f"{i}. {m['name']}  [{bar}] {pct}%\n"
            f"   Grade {m['grade']} · {m['subject']} · {m['qualifications']}\n"
            f"   {m['explanation']}\n"
        )
    lines.append("Type  book 1 , book 2 , or book 3  to confirm a pairing.")
    return "\n".join(lines)


# ── State machine ─────────────────────────────────────────────────────────────

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "state":   "idle",
            "subject": None,
            "grade":   None,
            "name":    None,
            "matches": [],
        }
    return sessions[session_id]

def reset_session(session_id: str):
    sessions[session_id] = {
        "state":   "idle",
        "subject": None,
        "grade":   None,
        "name":    None,
        "matches": [],
    }


def process_message(session_id: str, text: str) -> str:
    sess   = get_session(session_id)
    state  = sess["state"]
    intent = detect_intent(text)

    # Global commands work from any state
    if intent == "restart":
        reset_session(session_id)
        return "Sure! Let's start over.\n\nWhat subject does the mentee need help with?\n(Math, Physics, Chemistry, Biology, English)"

    if intent == "help":
        return (
            "Here's what I can do:\n\n"
            "• find match  — find the best mentor for a mentee\n"
            "• history     — see past pairings\n"
            "• restart     — start over\n\n"
            "Just tell me what you need!"
        )

    if intent == "history":
        return "__history__"   # signal to the route handler to fetch from DB

    # ── State: idle ───────────────────────────────────────────────────────────
    if state == "idle":
        subj = extract_subject(text)
        if subj:
            sess["subject"] = subj
            sess["state"]   = "ask_grade"
            return f"Got it — {subj}.\n\nWhat grade is the mentee in? (9, 10, 11, or 12)"

        if intent in ("find_match", "unknown"):
            sess["state"] = "ask_subject"
            return "Sure! What subject does the mentee need help with?\n(Math, Physics, Chemistry, Biology, English)"

        return "Hi! I can help you find and book mentor matches.\nType  find match  to get started, or  help  to see what I can do."

    # ── State: ask_subject ────────────────────────────────────────────────────
    if state == "ask_subject":
        subj = extract_subject(text)
        if not subj:
            return "I didn't catch the subject. Please choose one of:\nMath, Physics, Chemistry, Biology, English"
        sess["subject"] = subj
        sess["state"]   = "ask_grade"
        return f"{subj} — great.\n\nWhat grade is the mentee in? (9, 10, 11, or 12)"

    # ── State: ask_grade ──────────────────────────────────────────────────────
    if state == "ask_grade":
        grade = extract_grade(text)
        if not grade:
            return "Please enter a grade between 9 and 12."
        sess["grade"] = grade
        sess["state"] = "ask_name"
        return f"Grade {grade}.\n\nWhat's the mentee's name?"

    # ── State: ask_name ───────────────────────────────────────────────────────
    if state == "ask_name":
        name = text.strip().title()
        if len(name) < 2:
            return "Please enter the mentee's name."
        sess["name"]  = name
        sess["state"] = "searching"

        # Run RAG + model
        mentee     = {"name": name, "grade": sess["grade"], "subject": sess["subject"]}
        candidates = retriever.retrieve(sess["subject"], sess["grade"], top_k=5)
        ranked     = score_candidates(mentee, candidates)
        sess["matches"] = ranked
        sess["state"]   = "showing_results"

        return (
            f"Finding the best {sess['subject']} mentors for {name} (Grade {sess['grade']})...\n\n"
            + format_matches(ranked)
        )

    # ── State: showing_results ────────────────────────────────────────────────
    if state == "showing_results":
        if intent == "book" or re.search(r"\b[123]\b", text):
            choice = extract_book_choice(text)
            if not choice or choice > len(sess["matches"]):
                return f"Please type  book 1 ,  book 2 , or  book 3 ."
            return f"__book__{choice}"   # signal to route handler

        if intent == "find_match":
            reset_session(session_id)
            sess2 = get_session(session_id)
            sess2["state"] = "ask_subject"
            return "Let's find another match!\n\nWhat subject does the mentee need help with?"

        return (
            "Type  book 1 ,  book 2 , or  book 3  to confirm a pairing.\n"
            "Or type  restart  to find a different mentor."
        )

    return "Something went wrong. Type  restart  to begin again."


# ── Route ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str

class ChatResponse(BaseModel):
    reply:   str
    state:   str
    matches: list = []

import sqlite3

def get_db():
    conn = sqlite3.connect("data/auxilizium.db")
    conn.row_factory = sqlite3.Row
    return conn

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    raw = process_message(req.session_id, req.message)
    sess = get_session(req.session_id)

    # Handle special signals from process_message
    if raw == "__history__":
        conn = get_db()
        rows = conn.execute(
            "SELECT mentor_name, mentee_name, subject, match_score, created_at "
            "FROM bookings ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        conn.close()
        if not rows:
            return ChatResponse(reply="No pairings booked yet.", state=sess["state"])
        lines = ["Past pairings:\n"]
        for r in rows:
            pct = round(r["match_score"] * 100)
            lines.append(f"• {r['mentee_name']} → {r['mentor_name']}  ({r['subject']}, {pct}% match)  {r['created_at'][:10]}")
        return ChatResponse(reply="\n".join(lines), state=sess["state"])

    if raw.startswith("__book__"):
        choice  = int(raw.replace("__book__", ""))
        mentor  = sess["matches"][choice - 1]
        conn    = get_db()
        conn.execute(
            "INSERT INTO bookings (mentor_name, mentee_name, subject, mentor_grade, mentee_grade, match_score, explanation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mentor["name"], sess["name"], sess["subject"],
             mentor["grade"], sess["grade"],
             mentor["match_score"], mentor["explanation"])
        )
        conn.commit()
        conn.close()
        reset_session(req.session_id)
        return ChatResponse(
            reply=(
                f"Booked!\n\n"
                f"{sess['name']}  →  {mentor['name']}\n"
                f"{sess['subject']} · Grade {sess['grade']} → Grade {mentor['grade']}\n\n"
                f"Type  find match  to pair another mentee."
            ),
            state="idle"
        )

    return ChatResponse(reply=raw, state=sess["state"], matches=sess.get("matches", []))
