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

from __future__ import annotations

import os
import sys
from pathlib import Path
from fastapi.responses import RedirectResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# dev reload marker

from api.admin import router as admin_router
from api.chat import router as chat_router
from api.contest_router import router as contest_router
from api.contest_image_router import router as image_router
from api.services import book_pairing_in_db, get_db, get_mentor_slots, init_db, match_mentors

app = FastAPI(title="Lumi Mentor Matcher")
app.include_router(chat_router)
app.include_router(admin_router, prefix="/admin")
app.include_router(contest_router, prefix="/contest")
app.include_router(image_router, prefix="/contest")

# Serve the frontend directory so the app can be opened via HTTP (avoids file:// CORS issues)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.get("/")
def root():
    return RedirectResponse(url="/frontend/chat.html")


# ── Request / Response models ─────────────────────────────────────────────────
class MenteeRequest(BaseModel):
    mentee_name: str
    mentee_grade: int | None = None
    mentee_subject: str
    mentee_query: str | None = None

class BookingRequest(BaseModel):
    mentor_name: str
    mentee_name: str
    subject: str
    mentor_grade: int
    mentee_grade: int
    match_score: float
    explanation: str
    slot_id: int | None = None
    slot_label: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/match")
def match_mentee(req: MenteeRequest, debug: bool = False):
    """
    Full pipeline:
      1. RAG retrieves top mentor candidates by semantic similarity
      2. Trained model or heuristic rescores and ranks them
      3. Returns ranked matches with explanations
    """
    query_text = req.mentee_query or req.mentee_subject
    if debug:
        from api.services import match_mentors_debug

        result = match_mentors_debug(req.mentee_name, query_text, req.mentee_grade, top_k=5)
        if not result.get("matches"):
            raise HTTPException(status_code=404, detail="No mentor candidates found")
        return result

    mentee, ranked = match_mentors(req.mentee_name, query_text, req.mentee_grade, top_k=5)
    if not ranked:
        raise HTTPException(status_code=404, detail="No mentor candidates found")

    return {
        "mentee":  mentee,
        "matches": ranked,
    }


@app.post("/book")
def book_pairing(req: BookingRequest):
    """Save a confirmed mentor-mentee pairing to SQLite."""
    try:
        book_pairing_in_db(
            mentor_name=req.mentor_name,
            mentee_name=req.mentee_name,
            subject=req.subject,
            mentor_grade=req.mentor_grade,
            mentee_grade=req.mentee_grade,
            match_score=req.match_score,
            explanation=req.explanation,
            slot_id=req.slot_id,
            slot_label=req.slot_label,
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail.lower() else 400
        raise HTTPException(status_code=status_code, detail=detail)

    return {"status": "booked", "mentor": req.mentor_name, "mentee": req.mentee_name}


@app.get("/mentors/{mentor_name}/slots")
def get_slots(mentor_name: str):
    """Return available (unbooked) 1-hour time slots for the given mentor."""
    return get_mentor_slots(mentor_name, only_available=True)


@app.get("/history")
def get_history():
    """Return all past pairings from SQLite."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM bookings ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/health")
def health():
    return {"status": "ok"}
