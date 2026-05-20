"""LangChain-based chat task routing for booking, general questions, and matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any

from langchain_core.runnables import RunnableBranch, RunnableLambda

from api.services import book_pairing_in_db, list_available_mentors, match_mentors
from rag.subject_utils import subject_key


@dataclass
class AgentResult:
    reply: str
    state: str
    matches: list[dict] | None = None
    booking_state: str | None = None


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_grade(text: str) -> int | None:
    t = _normalize(text)
    grade_words = {
        "grade 9": 9,
        "grade 10": 10,
        "grade 11": 11,
        "grade 12": 12,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    for key, value in grade_words.items():
        if key in t:
            return value
    match = re.search(r"\b(9|10|11|12)\b", t)
    return int(match.group(1)) if match else None


def extract_choice(text: str) -> int | None:
    match = re.search(r"\b(?:book\s*)?(1|2|3)\b", text.lower())
    return int(match.group(1)) if match else None


def is_help_request(text: str) -> bool:
    t = _normalize(text)
    return bool(
        re.search(r"^help(\s+me)?(\s+please)?$", t)
        or re.search(r"^help\b", t)
        or re.search(r"what can you do|commands|how do you work", t)
    )


def is_restart_request(text: str) -> bool:
    t = _normalize(text)
    return bool(re.search(r"\brestart\b|start over|new conversation|again", t))


def is_list_request(text: str) -> bool:
    t = _normalize(text)
    return bool(re.search(r"\blist\b|show all mentors|available mentors", t))


FAQ_ENTRIES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bhow (?:does|do) .*match|\bhow does matching work\b|\bhow are mentors matched\b", re.I),
        "I rank mentors in two stages: RAG finds the most relevant mentors for your freeform request, then the matcher rescoring picks the best fit.",
    ),
    (
        re.compile(r"\bwhat can you do\b|\bhelp\b|\bcommands\b|\bhow do you work\b", re.I),
        "I can find mentors from freeform text, list available mentors, explain matches, and book a pairing.",
    ),
    (
        re.compile(r"\bwhat subjects\b|\bwhich subjects\b|\bsupported subjects\b", re.I),
        "Supported subjects include math, physics, chemistry, biology, and english, but freeform requests like 'calculus help' or 'essay writing' also work.",
    ),
    (
        re.compile(r"\bwhat grade|grade levels|\bgrades\b", re.I),
        "The system is tuned for grades 9 through 12.",
    ),
    (
        re.compile(r"\bhow do i book\b|\bbooking\b|\breserve\b|\bconfirm pairing\b", re.I),
        "After I show matches, type 'book 1', 'book 2', or 'book 3'. I may ask for the mentee's name before saving the booking.",
    ),
    (
        re.compile(r"\brestart\b|\bstart over\b|\bnew conversation\b", re.I),
        "Type restart any time and I’ll clear the current match and booking flow.",
    ),
]


def is_booking_request(text: str, session: dict[str, Any]) -> bool:
    if session.get("state") == "awaiting_booking_name":
        return True
    t = _normalize(text)
    return bool(re.search(r"\bbook\b|confirm pairing|reserve mentor", t))


def is_match_request(text: str) -> bool:
    t = _normalize(text)
    return bool(
        re.search(r"\bfind\b.*\bmentor\b|\bmatch\b|\bpair\b", t)
        or re.search(r"\bneed help with\b|\blooking for\b.*\bmentor\b|\btutor\b", t)
    )


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", _normalize(text)))


def _contains_any(tokens: set[str], words: set[str]) -> bool:
    return bool(tokens & words)


MATCH_QUERY_HINT_WORDS = {
    "help",
    "struggling",
    "stuck",
    "practice",
    "learn",
    "study",
    "understand",
    "improve",
    "revision",
    "review",
    "exam",
    "test",
    "homework",
    "assignment",
    "calculus",
    "algebra",
    "geometry",
    "trigonometry",
    "physics",
    "chemistry",
    "biology",
    "english",
    "essay",
    "grammar",
}


class MentorTaskAgents:
    def __init__(self) -> None:
        self.router = RunnableBranch(
            (lambda context: context["intent"] == "search", RunnableLambda(self._search_agent)),
            (lambda context: context["intent"] == "general", RunnableLambda(self._general_agent)),
            RunnableLambda(self._general_agent),
        )

    def run(self, session_id: str, message: str, session: dict[str, Any], forced_agent: str | None = None) -> AgentResult:
        intent = self._intent_for(message, session, forced_agent)
        payload = {"session_id": session_id, "message": message, "session": session, "intent": intent}
        return self.router.invoke(payload)

    def _intent_for(self, message: str, session: dict[str, Any], forced_agent: str | None = None) -> str:
        if forced_agent == "general":
            return "general"
        if forced_agent in {"booking", "match"}:
            return "search"
        if is_restart_request(message):
            return "general"
        if is_help_request(message) or is_list_request(message):
            return "general"
        if is_booking_request(message, session):
            return "search"
        if is_match_request(message):
            return "search"
        return "general"

    def _general_agent(self, context: dict[str, Any]) -> AgentResult:
        message = context["message"]
        session = context["session"]

        if is_restart_request(message):
            session.clear()
            session.update({"state": "idle", "subject": None, "grade": None, "name": None, "matches": []})
            return AgentResult(
                reply=(
                    "Sure. We can start over.\n\n"
                    "Tell me what the mentee needs help with in plain language, like 'I need help with calculus'."
                ),
                state="idle",
            )

        if is_list_request(message):
            mentors = list_available_mentors()
            if not mentors:
                return AgentResult(reply="No mentors are available right now.", state=session.get("state", "idle"), booking_state="no_mentors")
            lines = ["Available mentors:"]
            for mentor in mentors:
                status = "available" if mentor["available"] else "booked"
                lines.append(f"- {mentor['name']} · Grade {mentor['grade']} · {mentor['subject']} · {status}")
            return AgentResult(reply="\n".join(lines), state=session.get("state", "idle"), booking_state="mentor_list")

        general_answer = self._answer_general_question(message, session)
        if general_answer:
            return AgentResult(
                reply=general_answer,
                state=session.get("state", "idle"),
                booking_state=session.get("state", "idle"),
            )

        for pattern, answer in FAQ_ENTRIES:
            if pattern.search(message):
                return AgentResult(
                    reply=answer,
                    state=session.get("state", "idle"),
                    booking_state=session.get("state", "idle"),
                )

        return AgentResult(
            reply=(
                "I can answer questions about this mentor system: matching, booking, availability, and how to use chat modes.\n\n"
                "Try asking: 'How does matching work?', 'What subjects are available?', 'How do I book?', or 'Show all mentors'."
            ),
            state=session.get("state", "idle"),
            booking_state=session.get("state", "idle"),
        )

    def _answer_general_question(self, message: str, session: dict[str, Any]) -> str | None:
        tokens = _token_set(message)
        normalized = _normalize(message)

        if not tokens:
            return "Ask me anything - I can help with mentor matching, general questions, or just chat."

        if _contains_any(tokens, {"day", "date", "today"}):
            day_name = datetime.now().strftime("%A")
            date_str = datetime.now().strftime("%B %d, %Y")
            return f"Today is {day_name}, {date_str}."

        if _contains_any(tokens, {"time"}) and _contains_any(tokens, {"what", "current", "is"}):
            time_str = datetime.now().strftime("%I:%M %p")
            return f"The current time is {time_str}."

        if _contains_any(tokens, {"cycle", "cycles", "grading", "semester", "term", "marking"}) and _contains_any(tokens, {"end", "when", "close"}):
            return "I don't have access to your school's calendar, but grading cycles typically end at the conclusion of each term or semester. Check with your school for specific dates."

        if _contains_any(tokens, {"hi", "hello", "hey", "yo", "sup"}):
            return "Hi! I'm here to help with mentor matching, answer questions, or just chat."

        if _contains_any(tokens, {"thanks", "thank", "thx", "appreciate"}):
            return "You're welcome!"

        if _contains_any(tokens, {"how", "works", "work", "matching", "match", "algorithm", "scoring"}) and _contains_any(tokens, {"match", "matching", "mentor"}):
            return "Matching uses two steps: RAG retrieves relevant mentor profiles from your freeform text, then a matcher rescoring ranks the best fits."

        if _contains_any(tokens, {"subject", "subjects", "topic", "topics", "available"}) and _contains_any(tokens, {"what", "which", "supported", "available", "offer"}):
            mentors = list_available_mentors()
            subjects = sorted({str(m.get("subject") or "").strip() for m in mentors if m.get("subject")})
            if subjects:
                return f"Current subjects from available mentors: {', '.join(subjects)}."
            return "Supported subjects include math, physics, chemistry, biology, and english."

        if _contains_any(tokens, {"grade", "grades", "level", "levels"}):
            return "This system is tuned for grades 9 to 12."

        if _contains_any(tokens, {"state", "status"}) and _contains_any(tokens, {"booking", "book"}):
            current_state = session.get("state", "idle")
            if current_state == "awaiting_booking_name":
                return "Booking is waiting for a mentee name. Reply with the name to finish booking."
            if current_state == "showing_results":
                return "You currently have match results. Type 'book 1', 'book 2', or 'book 3' to confirm one."
            return "No active booking flow right now. Ask me to find a mentor first."

        if _contains_any(tokens, {"book", "booking", "reserve", "confirm"}):
            return "After you get results, type 'book 1', 'book 2', or 'book 3'. If needed, I will ask for the mentee name before saving."

        if _contains_any(tokens, {"cancel", "reset", "restart", "clear"}):
            return "Type 'restart' to clear current matching and booking state."

        if "top match" in normalized or "best match" in normalized:
            matches = session.get("matches") or []
            if matches:
                top = matches[0]
                pct = round(float(top.get("match_score") or 0) * 100)
                return f"Your current top match is {top.get('name', 'Unknown')} at {pct}% for {top.get('subject', 'the selected subject')}."
            return "I do not have match results yet. Ask me to find a mentor first."

        if _contains_any(tokens, {"name"}) and _contains_any(tokens, {"what", "your", "is", "who"}):
            return "I'm Lumi, a mentor matching assistant. I help you find the right tutor and book sessions."

        if _contains_any(tokens, {"study", "tips", "advice", "help", "improve"}) and _contains_any(tokens, {"study", "learning", "grade", "score"}):
            return "Study tips: Take breaks every 25-30 min, find a quiet space, review notes before bed, and teach the material to someone else to check understanding. A good mentor can really accelerate learning."

        if _contains_any(tokens, {"motivation", "motivate", "tired", "lazy", "procrastinate", "don't", "dont"}) and len(tokens) >= 3:
            return "Remember why you started! Breaking work into smaller chunks makes it feel less overwhelming. And having a mentor can provide accountability and personalized guidance that keeps you on track."

        if _contains_any(tokens, {"math", "calculus", "algebra", "geometry"}) and _contains_any(tokens, {"definition", "what", "explain", "mean"}):
            if "calculus" in normalized:
                return "Calculus studies rates of change and accumulation. It has two main branches: differential calculus (derivatives/slopes) and integral calculus (areas/totals). A math mentor can make this intuitive!"
            elif "algebra" in normalized:
                return "Algebra uses symbols and rules to solve equations and understand relationships. The key is balancing both sides of an equation. Start with basics and build from there."
            elif "geometry" in normalized:
                return "Geometry studies shapes, angles, and spatial relationships. It combines visual reasoning with logical proofs. Drawing diagrams often helps unlock solutions."
            return "Math builds on foundations step by step. Don't skip concepts - they usually connect later. A tutor can identify gaps and fill them."

        if _contains_any(tokens, {"physics", "chemistry", "biology"}) and _contains_any(tokens, {"definition", "what", "explain", "hard", "difficult"}):
            if "physics" in normalized:
                return "Physics explains how the natural world works - motion, forces, energy, waves. Start by understanding concepts before diving into equations."
            elif "chemistry" in normalized:
                return "Chemistry is about how substances interact and transform. Understanding atoms, bonds, and reactions is key. Visual models and hands-on labs help a lot."
            elif "biology" in normalized:
                return "Biology studies living systems from cells to ecosystems. Learn the vocabulary and relationships between concepts. It all connects!"
            return "Science subjects build on each other. Understand concepts deeply, not just memorize. A science tutor can make abstract ideas concrete."

        if _contains_any(tokens, {"english", "writing", "essay", "grammar", "reading"}):
            return "For writing: outline first, draft freely, then edit ruthlessly. For grammar: understand the rules, then break them intentionally. Reading widely improves all writing. Practice with feedback helps most."

        if _contains_any(tokens, {"test", "exam", "exam", "prepare", "study"}) and _contains_any(tokens, {"prepare", "ready", "tips", "how"}):
            return "Exam prep: review old tests/practice problems, make study guides, teach concepts aloud, get sleep before the exam, and stay calm. Mentors are great for targeted review of weak areas."

        if _contains_any(tokens, {"homework", "assignment", "due", "deadline"}):
            return "Stay organized with a calendar. Break assignments into steps with mini-deadlines. Start early so you have time for revisions. If stuck, reach out to a tutor or peer for a fresh perspective."

        if _contains_any(tokens, {"sleep", "tired", "focus", "concentrate", "distracted"}):
            return "Sleep is critical for memory and focus. Aim for 7-9 hours. If you're struggling to concentrate, take 5-10 min breaks, hydrate, and study in a quiet space. Consistency matters more than duration."

        if _contains_any(tokens, {"stress", "stressed", "anxious", "anxiety", "overwhelm", "overwhelmed", "pressure", "cope"}) and len(tokens) >= 2:
            return "Stress is normal but manageable. Break tasks down, prioritize, take breaks, and talk to someone you trust. Remember you're not alone - many students feel the same way. Consider reaching out to school counselors too."

        if _contains_any(tokens, {"school", "class", "teacher", "difficult", "hard"}):
            return "Every student struggles sometimes. The key is asking for help early. Teachers, tutors, and peers are all resources. Don't wait until the last minute to address gaps."

        if _contains_any(tokens, {"joke", "funny", "laugh", "entertai"}) and _contains_any(tokens, {"joke", "funny", "me"}):
            jokes = [
                "Why did the student do math in the garden? Because they wanted to improve their roots.",
                "What did the math teacher say to the student? You're a fraction of what you could be!",
                "I tried to do chemistry homework but it was too ionic an experience.",
                "History class is so old... it can't even remember what happened.",
            ]
            return jokes[len(tokens) % len(jokes)]

        if _contains_any(tokens, {"cool", "awesome", "great", "good", "nice"}):
            return "Glad you're in a good mood! That energy is great for learning. If you need help or want to find a mentor, I'm here."

        if _contains_any(tokens, {"sad", "bad", "terrible", "awful", "hate"}):
            return "Sorry to hear you're having a rough time. Remember it's temporary and gets better. Take care of yourself and reach out for support when needed. A good mentor can be encouraging too."

        if _contains_any(tokens, {"love", "like", "best", "favorite", "prefer"}) and _contains_any(tokens, {"subject", "learning", "subject"}):
            return "That's great! Finding your passion in a subject makes learning so much more enjoyable. A mentor in that area can deepen your knowledge even further."

        if _contains_any(tokens, {"question", "ask", "confused", "stuck", "help"}) and len(tokens) >= 3:
            return "What's your question? I can help with mentor matching, or if it's about school subjects, I can point you in the right direction or find a tutor."

        if len(normalized) > 5 and _contains_any(tokens, {"what", "how", "why", "when", "where"}):
            return "That's a great question! I'm specialized in mentor matching and general study help, but I can try to point you in the right direction. What would help most?"

        return None

    def _search_agent(self, context: dict[str, Any]) -> AgentResult:
        message = context["message"].strip()
        session = context["session"]
        matches = session.get("matches", [])
        choice = extract_choice(message)

        if session.get("state") == "awaiting_booking_name":
            mentee_name = message.strip().title()
            if len(mentee_name) < 2:
                return AgentResult(reply="Please enter the mentee's name.", state="awaiting_booking_name", matches=matches, booking_state="awaiting_name")

            booking_choice = int(session.get("pending_booking_choice") or 1)
            if booking_choice < 1 or booking_choice > len(matches):
                return AgentResult(reply="I lost the selected mentor. Please run the search again.", state="idle", matches=[], booking_state="booking_lost")

            mentor = matches[booking_choice - 1]
            self._save_booking(session, mentor, mentee_name)
            self._reset_booking_state(session)
            return AgentResult(
                reply=(
                    f"Booked!\n\n{mentee_name} -> {mentor['name']}\n"
                    f"{session.get('subject') or mentor['subject']} - Grade {session.get('grade') or 0} -> Grade {mentor['grade']}"
                ),
                state="idle",
                matches=[],
                booking_state="booked",
            )

        if matches and choice:
            if choice > len(matches):
                return AgentResult(
                    reply="Tell me which result to book: book 1, book 2, or book 3.",
                    state=session.get("state", "showing_results"),
                    matches=matches,
                    booking_state="awaiting_choice",
                )

            mentor = matches[choice - 1]
            mentee_name = session.get("name")
            if not mentee_name:
                session["pending_booking_choice"] = choice
                session["state"] = "awaiting_booking_name"
                return AgentResult(
                    reply="What is the mentee's name so I can save the booking?",
                    state="awaiting_booking_name",
                    matches=matches,
                    booking_state="awaiting_name",
                )

            self._save_booking(session, mentor, mentee_name)
            self._reset_booking_state(session)
            return AgentResult(
                reply=(
                    f"Booked!\n\n{mentee_name} -> {mentor['name']}\n"
                    f"{session.get('subject') or mentor['subject']} - Grade {session.get('grade') or 0} -> Grade {mentor['grade']}"
                ),
                state="idle",
                matches=[],
                booking_state="booked",
            )

        if not self._is_actionable_match_query(message):
            return AgentResult(
                reply=(
                    "I need a little more detail before I can match mentors.\n\n"
                    "Tell me what the mentee needs help with, for example:\n"
                    "- 'Grade 10 calculus limits'\n"
                    "- 'Need help with physics kinematics'\n"
                    "- 'Essay writing and grammar support'"
                ),
                state=session.get("state", "idle"),
                matches=[],
                booking_state="needs_query",
            )

        grade = extract_grade(message) or session.get("grade")
        name = session.get("name") or self._extract_name(message)
        if name:
            session["name"] = name

        mentee, ranked = match_mentors(name or "Mentee", message, mentee_grade=grade, top_k=5)
        session["matches"] = ranked
        session["grade"] = mentee["grade"] or session.get("grade")
        session["subject"] = mentee.get("subject_hint") or mentee.get("subject")
        session["query_text"] = mentee.get("query_text")
        session["state"] = "showing_results"

        if not ranked:
            return AgentResult(
                reply="I could not find any mentors for that request. Try rephrasing the subject or skill.",
                state="idle",
                matches=[],
                booking_state="search_failed",
            )

        lines = ["Here are the top matches:"]
        for index, mentor in enumerate(ranked[:3], 1):
            pct = round(mentor["match_score"] * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            lines.append(
                f"{index}. {mentor['name']} [{bar}] {pct}%\n"
                f"   Grade {mentor['grade']} - {mentor['subject']} - {mentor['qualifications']}\n"
                f"   {mentor['explanation']}"
            )

        lines.append("\nType 'book 1', 'book 2', or 'book 3' to confirm a pairing.")
        return AgentResult(reply="\n".join(lines), state="showing_results", matches=ranked, booking_state="search_results")

    def _is_actionable_match_query(self, message: str) -> bool:
        normalized = _normalize(message)
        if not normalized:
            return False

        tokens = _token_set(normalized)
        if len(tokens) < 3 and not extract_grade(normalized):
            return False

        if subject_key(normalized):
            return True

        if _contains_any(tokens, MATCH_QUERY_HINT_WORDS):
            return True

        if re.search(r"\b(help|need|looking|tutor)\b", normalized) and len(tokens) >= 4:
            return True

        return False

    def _save_booking(self, session: dict[str, Any], mentor: dict[str, Any], mentee_name: str) -> None:
        subject = session.get("subject") or mentor.get("subject") or "General"
        grade = int(session.get("grade") or 0)
        book_pairing_in_db(
            mentor_name=mentor["name"],
            mentee_name=mentee_name,
            subject=subject,
            mentor_grade=int(mentor.get("grade") or 0),
            mentee_grade=grade,
            match_score=float(mentor.get("match_score") or 0.0),
            explanation=str(mentor.get("explanation") or "Potential match."),
        )

    def _reset_booking_state(self, session: dict[str, Any]) -> None:
        session.update(
            {
                "state": "idle",
                "subject": None,
                "grade": None,
                "name": None,
                "query_text": None,
                "pending_booking_choice": None,
                "matches": [],
            }
        )

    def _extract_name(self, message: str) -> str | None:
        match = re.search(r"\b(?:my name is|i am|i'm|im)\s+([a-z][a-z\-']+(?:\s+[a-z][a-z\-']+)*)", message, re.I)
        if match:
            return match.group(1).strip().title()
        return None
