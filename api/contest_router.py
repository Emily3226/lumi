"""
api/contest_router.py

FastAPI endpoints for the contest knowledge system.
Mount this in main.py:
    from api.contest_router import router as contest_router
    app.include_router(contest_router, prefix="/contest")
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.contest_agent import contest_agent
from api.session_store import get_session, save_sessions, sessions
from rag.contest_retriever import (
    collection_count,
    get_by_contest_year,
    list_available_contests,
    query as chroma_query,
)

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class ContestChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ContestChatResponse(BaseModel):
    reply: str
    problems: list[dict] | None = None
    intent: str
    active_agent: str | None = None


class ContestSearchRequest(BaseModel):
    query: str
    contest: str | None = None
    year: int | None = None
    grade: int | None = None
    topic: str | None = None
    n_results: int = 5


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=ContestChatResponse)
def contest_ask(req: ContestChatRequest):
    """
    Main conversational endpoint.
    Routes to the appropriate handler based on intent detection.
    """
    session_id = req.session_id or "contest_default"
    session = get_session(session_id)
    session.setdefault("messages", [])
    session["messages"].append({"role": "user", "content": req.message})

    result = contest_agent.run(req.message, session)
    if result.active_agent:
        session["active_agent"] = result.active_agent
    if result.problems is not None:
        session["matches"] = result.problems
    session["messages"].append({"role": "assistant", "content": result.reply})
    session["messages"] = session["messages"][-12:]
    save_sessions(sessions)
    return ContestChatResponse(
        reply=result.reply,
        problems=result.problems,
        intent=result.intent,
        active_agent=result.active_agent,
    )


@router.post("/search")
def contest_search(req: ContestSearchRequest):
    """
    Raw semantic search over the contest database.
    Useful for the frontend to fetch problems programmatically.
    """
    if collection_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Contest database not indexed yet. Run scripts/ingest_contests.py first.",
        )

    results = chroma_query(
        text=req.query,
        n_results=req.n_results,
        contest=req.contest,
        year=req.year,
        grade=req.grade,
        topic=req.topic,
    )
    return {"results": results, "count": len(results)}


@router.get("/contests")
def list_contests():
    """List all indexed contests with their year ranges."""
    available = list_available_contests()
    return {"contests": available, "total_problems": collection_count()}


@router.get("/contests/{contest_name}/{year}")
def get_contest_problems(contest_name: str, year: int):
    """Retrieve all problems for a specific contest and year."""
    if collection_count() == 0:
        raise HTTPException(status_code=503, detail="Contest database not indexed yet.")

    problems = get_by_contest_year(contest_name, year)
    if not problems:
        raise HTTPException(
            status_code=404,
            detail=f"No problems found for {contest_name} {year}. Check the contest name or run ingestion.",
        )
    return {"contest": contest_name, "year": year, "problems": problems}


@router.get("/status")
def contest_status():
    """Returns indexing status — useful for the frontend to check before querying."""
    count = collection_count()
    available = list_available_contests() if count > 0 else []
    return {
        "indexed": count > 0,
        "problem_count": count,
        "contests": available,
    }