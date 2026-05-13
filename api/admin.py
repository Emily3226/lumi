from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import os
from typing import Optional

router = APIRouter()


def _get_db():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lumi.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


class MentorIn(BaseModel):
    name: str
    grade: int
    qualifications: Optional[str] = ""
    subject: str


@router.get("/mentors")
def list_mentors():
    conn = _get_db()
    rows = conn.execute("SELECT name, grade, qualifications, subject, available FROM mentors").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/mentors")
def add_mentor(m: MentorIn):
    conn = _get_db()
    conn.execute(
        "INSERT OR REPLACE INTO mentors (name, grade, qualifications, subject, available) VALUES (?, ?, ?, ?, 1)",
        (m.name, m.grade, m.qualifications or "", m.subject),
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "mentor": m.name}


@router.post("/mentors/{name}/availability")
def set_availability(name: str, available: int = 1):
    conn = _get_db()
    cur = conn.execute("SELECT name FROM mentors WHERE name = ?", (name,)).fetchone()
    if not cur:
        conn.close()
        raise HTTPException(status_code=404, detail="Mentor not found")
    conn.execute("UPDATE mentors SET available = ? WHERE name = ?", (1 if available else 0, name))
    conn.commit()
    conn.close()
    return {"status": "ok", "name": name, "available": bool(available)}


@router.get("/mentees")
def list_mentees():
    conn = _get_db()
    rows = conn.execute("SELECT name, grade, subject FROM mentees").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/bookings")
def list_bookings():
    conn = _get_db()
    rows = conn.execute(
        "SELECT id, mentor_name, mentee_name, subject, mentor_grade, mentee_grade, match_score, explanation, created_at, status FROM bookings ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/bookings/{booking_id}/cancel")
def cancel_booking(booking_id: int):
    conn = _get_db()
    cur = conn.execute("SELECT id, mentor_name, status FROM bookings WHERE id = ?", (booking_id,)).fetchone()
    if not cur:
        conn.close()
        raise HTTPException(status_code=404, detail="Booking not found")
    if cur['status'] == 'cancelled':
        conn.close()
        return {"status": "already_cancelled", "id": booking_id}

    # mark booking cancelled and release mentor
    conn.execute("UPDATE bookings SET status = 'cancelled' WHERE id = ?", (booking_id,))
    conn.execute("UPDATE mentors SET available = 1 WHERE name = ?", (cur['mentor_name'],))
    conn.commit()
    conn.close()
    return {"status": "cancelled", "id": booking_id}


@router.post("/bookings/{booking_id}/release-mentor")
def release_mentor(booking_id: int):
    conn = _get_db()
    cur = conn.execute("SELECT mentor_name FROM bookings WHERE id = ?", (booking_id,)).fetchone()
    if not cur:
        conn.close()
        raise HTTPException(status_code=404, detail="Booking not found")
    conn.execute("UPDATE mentors SET available = 1 WHERE name = ?", (cur['mentor_name'],))
    conn.commit()
    conn.close()
    return {"status": "mentor_released", "id": booking_id}


@router.post("/train-model")
def train_mentor_model():
    """
    Trigger training of the ML mentor matcher model using historical booking data.
    This should be called periodically as more bookings accumulate.
    """
    try:
        from models.train import train_model
        result = train_model()
        if result is None:
            return {
                "status": "training_skipped",
                "reason": "Insufficient training data (< 2 bookings)"
            }
        return {
            "status": "training_complete",
            "message": "Mentor matcher model trained successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
