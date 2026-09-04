from dataclasses import asdict

import jinja2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.db import get_db as _get_db
from api.pairing_emails import parse_pairings_csv, render_emails, send_rendered_email
from api.services import add_mentor_slot, delete_mentor_slot, get_mentor_slots

router = APIRouter()


class MentorIn(BaseModel):
    name: str
    grade: int
    qualifications: Optional[str] = ""
    subject: str


@router.get("/mentors")
def list_mentors():
    db = _get_db()
    return list(db["mentors"].find({}, {"_id": 0, "name": 1, "grade": 1, "qualifications": 1, "subject": 1, "available": 1}))


@router.post("/mentors")
def add_mentor(m: MentorIn):
    db = _get_db()
    db["mentors"].update_one(
        {"name": m.name},
        {"$set": {
            "name": m.name,
            "grade": m.grade,
            "qualifications": m.qualifications or "",
            "subject": m.subject,
        }, "$setOnInsert": {"available": 1}},
        upsert=True,
    )
    return {"status": "ok", "mentor": m.name}


@router.post("/mentors/{name}/availability")
def set_availability(name: str, available: int = 1):
    db = _get_db()
    if not db["mentors"].find_one({"name": name}):
        raise HTTPException(status_code=404, detail="Mentor not found")
    db["mentors"].update_one({"name": name}, {"$set": {"available": 1 if available else 0}})
    return {"status": "ok", "name": name, "available": bool(available)}


@router.get("/mentees")
def list_mentees():
    db = _get_db()
    return list(db["mentees"].find({}, {"_id": 0, "name": 1, "grade": 1, "subject": 1}))


@router.get("/bookings")
def list_bookings():
    db = _get_db()
    fields = {
        "_id": 0, "id": 1, "mentor_name": 1, "mentee_name": 1, "subject": 1,
        "mentor_grade": 1, "mentee_grade": 1, "match_score": 1, "explanation": 1,
        "created_at": 1, "status": 1, "slot_id": 1, "slot_label": 1,
    }
    return list(db["bookings"].find({}, fields).sort("created_at", -1))


@router.post("/bookings/{booking_id}/cancel")
def cancel_booking(booking_id: int):
    db = _get_db()
    cur = db["bookings"].find_one({"id": booking_id})
    if not cur:
        raise HTTPException(status_code=404, detail="Booking not found")
    if cur["status"] == "cancelled":
        return {"status": "already_cancelled", "id": booking_id}

    db["bookings"].update_one({"id": booking_id}, {"$set": {"status": "cancelled"}})
    db["mentors"].update_one({"name": cur["mentor_name"]}, {"$set": {"available": 1}})
    if cur.get("slot_id"):
        db["mentor_timeslots"].update_one({"id": cur["slot_id"]}, {"$set": {"available": 1}})
    return {"status": "cancelled", "id": booking_id}


@router.post("/bookings/{booking_id}/release-mentor")
def release_mentor(booking_id: int):
    db = _get_db()
    cur = db["bookings"].find_one({"id": booking_id})
    if not cur:
        raise HTTPException(status_code=404, detail="Booking not found")
    db["mentors"].update_one({"name": cur["mentor_name"]}, {"$set": {"available": 1}})
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


# ── Time slot management ──────────────────────────────────────────────────────

class SlotIn(BaseModel):
    day_of_week: str  # e.g. "Monday", "Tuesday", ...
    start_time: str   # HH:MM (24-hour)


@router.get("/mentors/{name}/slots")
def list_slots(name: str, all: bool = False):
    """List weekly time slots for a mentor. Pass ?all=true to include booked slots."""
    return get_mentor_slots(name, only_available=not all)


@router.post("/mentors/{name}/slots")
def create_slot(name: str, slot: SlotIn):
    """Add a recurring weekly 1-hour slot for a mentor."""
    try:
        result = add_mentor_slot(name, slot.day_of_week, slot.start_time)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/slots/{slot_id}")
def remove_slot(slot_id: int):
    """Delete a time slot by its ID."""
    deleted = delete_mentor_slot(slot_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Slot not found")
    return {"status": "deleted", "slot_id": slot_id}


# ── Pairing announcement emails ───────────────────────────────────────────────
# Bulk mentor/mentee pairing emails rendered from a CSV + Jinja2 templates.
# See api/pairing_emails.py for the CSV format and template variables.

class PairingEmailRenderRequest(BaseModel):
    csv_text: str
    mentor_template: str
    mentee_template: str
    subject_template: str
    extra_vars: dict[str, str] = {}


@router.post("/pairing-emails/preview")
def preview_pairing_emails(req: PairingEmailRenderRequest):
    """Render every mentor/mentee email WITHOUT sending anything - always the
    first step before /pairing-emails/send.
    """
    try:
        mentors = parse_pairings_csv(req.csv_text)
        rendered = render_emails(
            mentors, req.mentor_template, req.mentee_template, req.subject_template, req.extra_vars
        )
    except jinja2.TemplateError as exc:
        raise HTTPException(status_code=400, detail=f"Template error: {exc}")

    return {
        "mentor_count": len(mentors),
        "email_count": len(rendered),
        "emails": [asdict(e) for e in rendered],
    }


class PairingEmailSendRequest(PairingEmailRenderRequest):
    # Must be exactly "SEND" - a deliberate speed bump before a real batch
    # send to real families goes out, on top of whatever the UI already asks.
    confirm: str
    # Optional ISO-8601 datetime. If set, Resend schedules delivery for then
    # instead of sending immediately.
    scheduled_at: Optional[str] = None


@router.post("/pairing-emails/send")
def send_pairing_emails(req: PairingEmailSendRequest):
    if req.confirm != "SEND":
        raise HTTPException(status_code=400, detail="Confirmation text must be exactly 'SEND'.")

    try:
        mentors = parse_pairings_csv(req.csv_text)
        rendered = render_emails(
            mentors, req.mentor_template, req.mentee_template, req.subject_template, req.extra_vars
        )
    except jinja2.TemplateError as exc:
        raise HTTPException(status_code=400, detail=f"Template error: {exc}")

    results = []
    for email in rendered:
        ok, detail = send_rendered_email(email, scheduled_at=req.scheduled_at)
        results.append({
            "to": email.to,
            "recipient_type": email.recipient_type,
            "mentor_name": email.mentor_name,
            "mentee_names": email.mentee_names,
            "ok": ok,
            "detail": detail,
        })

    return {
        "sent": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "results": results,
    }
