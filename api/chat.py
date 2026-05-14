"""Conversational layer on top of the RAG + matching pipeline."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.agents import MentorTaskAgents

router = APIRouter()
agents = MentorTaskAgents()

# In production swap this for Redis or a DB table keyed by session_id.
sessions: dict[str, dict] = {}


def _new_session() -> dict:
    return {
        "state": "idle",
        "subject": None,
        "grade": None,
        "name": None,
        "query_text": None,
        "pending_booking_choice": None,
        "matches": [],
    }


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = _new_session()
    return sessions[session_id]


def reset_session(session_id: str) -> None:
    sessions[session_id] = _new_session()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    forced_agent: str | None = None


class ChatResponse(BaseModel):
    reply: str
    state: str
    matches: list = Field(default_factory=list)
    booking_state: str | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    result = agents.run(req.session_id, req.message, session, forced_agent=req.forced_agent)

    session["state"] = result.state
    if result.matches is not None:
        session["matches"] = result.matches

    return ChatResponse(
        reply=result.reply,
        state=result.state,
        matches=result.matches or [],
        booking_state=result.booking_state,
    )
