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
from api.admin import router as admin_router

app = FastAPI(title="Lumi Mentor Matcher")
app.include_router(chat_router)
app.include_router(admin_router, prefix="/admin")

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
    conn = sqlite3.connect("data/lumi.db")
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

    # Mentors table stores canonical mentor profiles and availability
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentors (
            name           TEXT PRIMARY KEY,
            grade          INTEGER,
            qualifications TEXT,
            subject        TEXT,
            available      INTEGER DEFAULT 1
        )
    """)

    # Mentees table for optionally storing incoming mentee records
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentees (
            name    TEXT PRIMARY KEY,
            grade   INTEGER,
            subject TEXT
        )
    """)

    # Populate mentors from CSV if mentors table is empty
    cur = conn.execute("SELECT COUNT(1) as c FROM mentors")
    if cur.fetchone()[0] == 0:
        import csv
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pairings.csv")
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                seen = set()
                for r in reader:
                    name = r.get('mentor_name')
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    conn.execute(
                        "INSERT OR IGNORE INTO mentors (name, grade, qualifications, subject, available) VALUES (?, ?, ?, ?, 1)",
                        (name, int(r.get('mentor_grade') or 0), r.get('mentor_qualifications') or '', r.get('mentor_subject') or '')
                    )
        except FileNotFoundError:
            # no seed data available; leave mentors empty
            pass
    # Ensure bookings table has a status column for lifecycle management
    cols = [r[1] for r in conn.execute("PRAGMA table_info(bookings)").fetchall()]
    if 'status' not in cols:
        conn.execute("ALTER TABLE bookings ADD COLUMN status TEXT DEFAULT 'active'")

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
    try:
        ranked = score_candidates(mentee, candidates, strict=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "mentee":  mentee,
        "matches": ranked,
    }


@app.post("/book")
def book_pairing(req: BookingRequest):
    """Save a confirmed mentor-mentee pairing to SQLite."""
    conn = get_db()
    # Ensure mentor exists and is available
    cur = conn.execute("SELECT available FROM mentors WHERE name = ?", (req.mentor_name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Mentor not found")
    if row[0] == 0:
        conn.close()
        raise HTTPException(status_code=400, detail="Mentor already booked")

    # Insert booking
    conn.execute("""
        INSERT INTO bookings
          (mentor_name, mentee_name, subject, mentor_grade, mentee_grade, match_score, explanation, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
    """, (
        req.mentor_name, req.mentee_name, req.subject,
        req.mentor_grade, req.mentee_grade,
        req.match_score, req.explanation,
    ))

    # Mark mentor unavailable
    conn.execute("UPDATE mentors SET available = 0 WHERE name = ?", (req.mentor_name,))

    # Ensure mentee recorded
    conn.execute("INSERT OR IGNORE INTO mentees (name, grade, subject) VALUES (?, ?, ?)",
                 (req.mentee_name, req.mentee_grade, req.subject))

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
