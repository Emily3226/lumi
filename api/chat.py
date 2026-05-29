"""Conversational layer on top of the RAG + matching pipeline."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.agents import MentorTaskAgents
from api.memory_store import observe_turn
from api.session_store import get_session, reset_session, save_sessions, sessions

router = APIRouter()
agents = MentorTaskAgents()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    forced_agent: str | None = None


class ChatResponse(BaseModel):
    reply: str
    state: str
    matches: list = Field(default_factory=list)
    booking_state: str | None = None
    active_agent: str | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    session.setdefault("messages", [])
    session["messages"].append({"role": "user", "content": req.message})
    result = agents.run(req.session_id, req.message, session, forced_agent=req.forced_agent)

    session["state"] = result.state
    if result.active_agent:
        session["active_agent"] = result.active_agent
    if result.matches is not None:
        session["matches"] = result.matches
    session["messages"].append({"role": "assistant", "content": result.reply})
    session["messages"] = session["messages"][-40:]
    save_sessions(sessions)
    observe_turn(req.session_id, req.message, result.reply, result.active_agent or session.get("active_agent"))

    return ChatResponse(
        reply=result.reply,
        state=result.state,
        matches=result.matches or [],
        booking_state=result.booking_state,
        active_agent=session.get("active_agent", "general"),
    )
