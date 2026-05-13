"""
api/main.py
FastAPI backend — exposes the RAG + model pipeline as a REST API.

Endpoints:
  POST /match   { mentee_name, mentee_grade, mentee_subject } → ranked matches
  POST /book    { mentor_name, mentee_name, ... }             → save booking
  GET  /history                                                → past pairings

Run with:
    uvicorn api.main:app --reload
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from datetime import datetime

from rag.retriever import MentorRetriever
from models.inference import score_candidates
from api.chat import router as chat_router

app = FastAPI(title="Auxilizium Mentor Matcher")
app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG index once at startup
retriever = MentorRetriever()


# ── SQLite setup ──────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("data/auxilizium.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
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
    """)
    conn.commit()
    conn.close()

init_db()


# ── Request / Response models ─────────────────────────────────────────────────
class MenteeRequest(BaseModel):
    mentee_name:    str
    mentee_grade:   int
    mentee_subject: str

class BookingRequest(BaseModel):
    mentor_name:  str
    mentee_name:  str
    subject:      str
    mentor_grade: int
    mentee_grade: int
    match_score:  float
    explanation:  str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/match")
def match_mentee(req: MenteeRequest):
    """
    Full pipeline:
      1. RAG retrieves top mentor candidates by semantic similarity
      2. Trained PyTorch model rescores and ranks them
      3. Returns ranked matches with explanations
    """
    mentee = {
        "name":    req.mentee_name,
        "grade":   req.mentee_grade,
        "subject": req.mentee_subject,
    }

    # Step 1 — RAG retrieval
    candidates = retriever.retrieve(req.mentee_subject, req.mentee_grade, top_k=5)
    if not candidates:
        raise HTTPException(status_code=404, detail="No mentor candidates found")

    # Step 2 — Model scoring
    ranked = score_candidates(mentee, candidates)

    return {
        "mentee":  mentee,
        "matches": ranked,
    }


@app.post("/book")
def book_pairing(req: BookingRequest):
    """Save a confirmed mentor-mentee pairing to SQLite."""
    conn = get_db()
    conn.execute("""
        INSERT INTO bookings
          (mentor_name, mentee_name, subject, mentor_grade, mentee_grade, match_score, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        req.mentor_name, req.mentee_name, req.subject,
        req.mentor_grade, req.mentee_grade,
        req.match_score, req.explanation,
    ))
    conn.commit()
    conn.close()
    return {"status": "booked", "mentor": req.mentor_name, "mentee": req.mentee_name}


@app.get("/history")
def get_history():
    """Return all past pairings from SQLite."""
    conn  = get_db()
    rows  = conn.execute("SELECT * FROM bookings ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/health")
def health():
    return {"status": "ok"}
