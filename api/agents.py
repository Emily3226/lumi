"""LangChain-based chat task routing for booking, general questions, and matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from langchain_core.runnables import RunnableBranch, RunnableLambda
import requests

from api.services import book_pairing_in_db, list_available_mentors, match_mentors
from rag.subject_utils import subject_key


GENERAL_KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "general_knowledge.md"
MENTOR_LIST_LIMIT = 3
logger = logging.getLogger(__name__)


def _load_dotenv_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("export "):
            if line.startswith("export "):
                line = line[len("export "):].strip()
            else:
                continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        current_value = os.environ.get(key, "")
        if current_value.strip():
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value


_load_dotenv_file()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
UNKNOWN_REQUEST_MESSAGE = (
    "I’m not sure how to help with that yet. Try asking for a mentor match, a contest problem, or a question from the knowledge file."
)

GROQ_NOT_CONFIGURED_MESSAGE = (
    "I can hand off to the AI fallback, but `GROQ_API_KEY` is not set yet. Put it in the repo root `.env` file and restart the backend."
)


@dataclass
class AgentResult:
    reply: str
    state: str
    matches: list[dict] | None = None
    booking_state: str | None = None
    active_agent: str | None = None


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



def is_agent_switch_request(text: str) -> bool:
    t = _normalize(text)
    return bool(
        (
            re.search(r"\bswitch\b|\bchange\b|\bgo to\b|\bopen\b|\buse\b", t)
            and re.search(r"\bgeneral\b|\bmatch\b|\bcontest\b", t)
        )
        or re.search(r"\bgeneral\s+agent\b|\bmatch\s+agent\b|\bcontest\s+agent\b", t)
    )


def _target_agent(text: str) -> str | None:
    t = _normalize(text)
    if re.search(r"\bcontest\b", t):
        return "contest"
    if re.search(r"\bmatch\b", t):
        return "match"
    if re.search(r"\bgeneral\b", t):
        return "general"
    return None


def is_booking_request(text: str, session: dict[str, Any]) -> bool:
    if session.get("state") == "awaiting_booking_name":
        return True
    t = _normalize(text)
    return bool(re.search(r"\bbook\b|confirm pairing|reserve mentor", t))


def is_match_request(text: str) -> bool:
    t = _normalize(text)
    return bool(
        re.search(r"\bfind\b.*\bmentor\b|\bmake\b.*\bmatch\b|\bmatch\b|\bpair\b|\bwant\b.*\bmentor\b|\bneed\b.*\bmentor\b", t)
        or re.search(r"\bneed help with\b|\blooking for\b.*\bmentor\b|\btutor\b|\bneed\b.*\btutor\b", t)
    )


def is_contest_request(text: str) -> bool:
    t = _normalize(text)
    return bool(re.search(r"\bcontest\b|\bolympiad\b|\bcompetition\b|\bAMC\b|\bAIME\b|contest math|contest problems|practice contest", t))


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", _normalize(text)))


def _is_date_or_time_question(text: str) -> bool:
    t = _normalize(text)
    # Match common date/time question patterns
    return bool(
        re.search(r"\b(today'?s date|current date|what(?:'s| is) the date|what(?:'s| is) today|what day is it|date today|day of the week|what day of the week|what day is it today|what time is it|current time|time now)\b", t)
    )


def _is_transformation_request(text: str) -> bool:
    t = _normalize(text)
    # User asks to summarize, explain, paraphrase, translate, analyze, or rewrite
    return bool(re.search(r"\b(summariz|summarise|explain|paraphrase|translate|rewrite|shorten|lengthen|expand|condense|analy(s|ze)|summation|tl;dr)\w*\b", t))


def _compact_text(text: str) -> str:
    return " ".join(sorted(_token_set(text)))


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


def _load_general_knowledge_entries() -> list[dict[str, str]]:
    if not GENERAL_KNOWLEDGE_PATH.exists():
        return []

    try:
        content = GENERAL_KNOWLEDGE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return []

    if not content:
        return []

    entries: list[dict[str, str]] = []
    for block in re.split(r"\n\s*\n+", content):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if len(lines) == 1 and lines[0].startswith("#"):
            continue

        has_structured_answer = any(line.lower().startswith(("a:", "answer:")) for line in lines)
        if has_structured_answer:
            question_lines: list[str] = []
            answer_lines: list[str] = []
            reading_answer = False
            for line in lines:
                lower = line.lower()
                if lower.startswith(("q:", "question:")):
                    question_lines.append(line.split(":", 1)[1].strip())
                    reading_answer = False
                elif lower.startswith(("a:", "answer:")):
                    answer_lines.append(line.split(":", 1)[1].strip())
                    reading_answer = True
                elif reading_answer:
                    answer_lines.append(line.strip())
                else:
                    question_lines.append(line.strip())

            answer = " ".join(answer_lines).strip()
            question_text = " ".join(question_lines).strip()
            if answer:
                entries.append({"content": question_text or answer, "answer": answer})
            continue

        text = " ".join(lines).strip()
        if text:
            entries.append({"content": text, "answer": text})

    return entries


def _score_general_knowledge(message: str, text: str) -> int:
    normalized_message = _normalize(message)
    normalized_text = _normalize(text)
    if not normalized_message or not normalized_text:
        return 0
    # Ignore common stopwords to avoid spurious matches from short queries
    STOPWORDS = {
        "the",
        "is",
        "a",
        "an",
        "in",
        "on",
        "for",
        "of",
        "to",
        "and",
        "or",
        "what",
        "how",
        "when",
        "where",
        "who",
        "which",
        "it",
        "this",
        "that",
    }

    score = 0
    # exact-substring match is a strong signal
    if normalized_message in normalized_text:
        score += 12

    message_tokens = _token_set(normalized_message)
    text_tokens = _token_set(normalized_text)

    # Consider only meaningful tokens for overlap scoring
    meaningful_msg_tokens = {t for t in message_tokens if t not in STOPWORDS}
    meaningful_text_tokens = {t for t in text_tokens if t not in STOPWORDS}

    # Require at least one meaningful token overlap to avoid stopword-only matches
    overlap = meaningful_msg_tokens & meaningful_text_tokens
    score += len(overlap) * 4

    # Keep phrase matching for multi-word phrases (still useful)
    words = [w for w in normalized_message.split() if w not in STOPWORDS]
    for size in (2, 3):
        for index in range(len(words) - size + 1):
            phrase = " ".join(words[index : index + size])
            if phrase and phrase in normalized_text:
                score += 4

    return score


def _groq_api_key() -> str:
    _load_dotenv_file()
    return os.getenv("GROQ_API_KEY", GROQ_API_KEY).strip()


class MentorTaskAgents:
    def __init__(self) -> None:
        self.router = RunnableBranch(
            (lambda context: context["intent"] == "search", RunnableLambda(self._search_agent)),
            (lambda context: context["intent"] == "general", RunnableLambda(self._general_agent)),
            RunnableLambda(self._general_agent),
        )

    def run(self, session_id: str, message: str, session: dict[str, Any], forced_agent: str | None = None) -> AgentResult:
        if forced_agent in {"general", "match", "contest"} and not is_agent_switch_request(message):
            self._set_active_agent(session, forced_agent)
        # Natural-language auto-switch: set active agent based on intent phrases
        if not is_agent_switch_request(message):
            if is_match_request(message):
                self._set_active_agent(session, "match")
            elif is_contest_request(message):
                self._set_active_agent(session, "contest")
        intent = self._intent_for(message, session, forced_agent)
        payload = {"session_id": session_id, "message": message, "session": session, "intent": intent}
        return self.router.invoke(payload)

    def _set_active_agent(self, session: dict[str, Any], agent: str) -> None:
        session["active_agent"] = agent
        if agent == "general":
            session["pending_match_step"] = None
            session["matches"] = []
            session["state"] = "idle"
        elif agent == "match" and session.get("state") not in {"awaiting_match_details", "showing_results", "awaiting_booking_name"}:
            session["pending_match_step"] = None
            session["state"] = "idle"

    def _intent_for(self, message: str, session: dict[str, Any], forced_agent: str | None = None) -> str:
        if is_agent_switch_request(message):
            return "general"
        if forced_agent == "general":
            return "general"
        if forced_agent in {"booking", "match"}:
            return "search"
        if session.get("active_agent") == "match":
            return "search"
        if is_restart_request(message):
            return "general"
        # If user asks "how do I book" or similar, treat it as a general question, not an actionable booking command
        if re.search(r"\bhow\b", message, re.I) and re.search(r"\bbook\b", message, re.I):
            return "general"
        if is_help_request(message) or is_list_request(message):
            return "general"
        if is_booking_request(message, session):
            return "search"
        if is_match_request(message):
            return "search"
        if extract_grade(message) and subject_key(message):
            return "search"
        return "general"

    def _general_agent(self, context: dict[str, Any]) -> AgentResult:
        message = context["message"]
        session = context["session"]

        if is_agent_switch_request(message):
            target = _target_agent(message)
            if target:
                session["active_agent"] = target
                if target == "contest":
                    reply = (
                        "Switched to the Contest agent.\n\n"
                        "Ask me for a specific contest problem, a solution explanation, or a practice set."
                    )
                elif target == "match":
                    reply = (
                        "Switched to the Match agent.\n\n"
                        "What grade is the mentee in?"
                    )
                    session["pending_match_step"] = "grade"
                else:
                    reply = (
                        "Switched to the General agent.\n\n"
                        "You can ask me questions from the knowledge file or request another mode anytime."
                    )
                return AgentResult(
                    reply=reply,
                    state="awaiting_match_details" if target == "match" else session.get("state", "idle"),
                    active_agent=target,
                    booking_state="needs_grade_and_subject" if target == "match" else session.get("state", "idle"),
                )

        if is_restart_request(message):
            session.clear()
            session.update({"state": "idle", "subject": None, "grade": None, "name": None, "matches": [], "active_agent": "general", "pending_match_step": None})
            return AgentResult(
                reply=(
                    "Sure. We can start over.\n\n"
                    "Tell me what the mentee needs help with in plain language, like 'I need help with calculus'."
                ),
                state="idle",
                active_agent="general",
            )

        if is_list_request(message):
            mentors = list_available_mentors()
            if not mentors:
                return AgentResult(reply="No mentors are available right now.", state=session.get("state", "idle"), booking_state="no_mentors", active_agent=session.get("active_agent", "general"))
            lines = ["Top 3 suitable mentors:"]
            for mentor in mentors[:MENTOR_LIST_LIMIT]:
                status = "available" if mentor["available"] else "booked"
                lines.append(f"- {mentor['name']} · Grade {mentor['grade']} · {mentor['subject']} · {status}")
            return AgentResult(reply="\n".join(lines), state=session.get("state", "idle"), booking_state="mentor_list", active_agent=session.get("active_agent", "general"))

        # Natural-language intent switches: if the user asks for a mentor or contest help, switch modes
        if is_match_request(message):
            session["active_agent"] = "match"
            session["pending_match_step"] = "grade"
            session["state"] = "awaiting_match_details"
            return AgentResult(
                reply=("Switched to the Match agent.\n\nWhat grade is the mentee in?"),
                state="awaiting_match_details",
                active_agent="match",
                booking_state="needs_grade_and_subject",
            )

        if is_contest_request(message):
            session["active_agent"] = "contest"
            session["state"] = "idle"
            return AgentResult(
                reply=(
                    "Switched to the Contest agent.\n\nAsk for a contest problem, a solution explanation, or say which contest you want to practice."
                ),
                state="idle",
                active_agent="contest",
            )

        general_answer = self._answer_general_question(message, session)
        if general_answer:
            return AgentResult(
                reply=general_answer,
                state=session.get("state", "idle"),
                booking_state=session.get("state", "idle"),
                active_agent=session.get("active_agent", "general"),
            )

        return AgentResult(
            reply=UNKNOWN_REQUEST_MESSAGE,
            state=session.get("state", "idle"),
            booking_state="unknown_request",
            active_agent=session.get("active_agent", "general"),
        )

    def _answer_general_question(self, message: str, session: dict[str, Any]) -> str | None:
        # Allow explicit user requests to switch agents (e.g., "switch to match agent")
        if is_agent_switch_request(message):
            target = _target_agent(message)
            if target:
                session["active_agent"] = target
                session["state"] = "idle"
                return f"Switched to the {target} agent. What would you like to do next?"

        # If it's a date/time question, bypass the KB and force the fallback LLM
        if _is_date_or_time_question(message):
            chat_answer = self._answer_from_free_chat(message, session, force_date=True)
            if chat_answer:
                return chat_answer

        # If the user requests a transformation (summarize/explain/translate/etc.),
        # bypass the KB and send directly to Groq for best-effort processing.
        if _is_transformation_request(message):
            chat_answer = self._answer_from_free_chat(message, session, force_date=False)
            if chat_answer:
                return chat_answer

        file_answer = self._answer_from_general_knowledge(message)
        if file_answer:
            return file_answer

        chat_answer = self._answer_from_free_chat(message, session)
        if chat_answer:
            return chat_answer

        if not _groq_api_key():
            return GROQ_NOT_CONFIGURED_MESSAGE

        return UNKNOWN_REQUEST_MESSAGE

    def _answer_from_general_knowledge(self, message: str) -> str | None:


        entries = _load_general_knowledge_entries()
        if not entries:
            return None

        # Avoid matching very short inputs (greetings, one-word queries) against the KB
        tokens = _token_set(message)
        if len(tokens) < 3:
            return None

        best_answer: str | None = None
        best_score = 0
        for entry in entries:
            score = _score_general_knowledge(message, entry["content"])
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

        # Return the best match when the overlap is strong enough to be a likely KB hit.
        # This lets paraphrases like "when does this cycle start" resolve to the stored answer.
        if best_answer and best_score >= 8:
            return best_answer
        return None

    def _answer_from_free_chat(self, message: str, session: dict[str, Any], force_date: bool = False) -> str | None:
        date_context = ""
        if force_date:
            today = datetime.now().strftime("%B %d, %Y").replace(" 0", " ")
            date_context = f"The current date is {today}. Use it when answering date and time questions. "

        prompt = (
            f"You are Lumi's general fallback agent. {date_context}"
            "The local knowledge file did not contain an answer for this question."
            " Provide a concise, helpful reply to the user's question using any recent conversation context supplied. "
            "Do not refuse to answer; if uncertain, give a best-effort response and indicate uncertainty clearly. "
            "Do not perform mentor-matching or contest problem solving unless the user explicitly requests switching modes. "
            "For casual messages like greetings, respond briefly and ask how you can help."
        )
        groq_api_key = _groq_api_key()
        if not groq_api_key:
            return None

        history = session.get("messages", [])
        recent_history = [
            item
            for item in history[-10:]
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        messages = [{"role": "system", "content": prompt}]
        messages.extend({"role": item["role"], "content": str(item["content"])} for item in recent_history)
        messages.append({"role": "user", "content": message})

        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.6,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {groq_api_key}"},
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.warning("Groq fallback request failed: %s", exc)
            return None
        except ValueError as exc:
            logger.warning("Groq fallback returned invalid JSON: %s", exc)
            return None

        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                choice_message = first_choice.get("message")
                if isinstance(choice_message, dict):
                    content = choice_message.get("content")
                    if content:
                        return str(content).strip()
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
            subject = session.get("subject") or mentor["subject"]
            grade = int(session.get("grade") or 0)
            self._save_booking(session, mentor, mentee_name)
            self._reset_booking_state(session)
            return AgentResult(
                reply=(
                    f"Booked!\n\n{mentee_name} -> {mentor['name']}\n"
                    f"{subject} - Grade {grade} -> Grade {mentor['grade']}"
                ),
                state="idle",
                matches=[],
                booking_state="booked",
                active_agent=session.get("active_agent", "general"),
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

            subject = session.get("subject") or mentor["subject"]
            grade = int(session.get("grade") or 0)
            self._save_booking(session, mentor, mentee_name)
            self._reset_booking_state(session)
            return AgentResult(
                reply=(
                    f"Booked!\n\n{mentee_name} -> {mentor['name']}\n"
                    f"{subject} - Grade {grade} -> Grade {mentor['grade']}"
                ),
                state="idle",
                matches=[],
                booking_state="booked",
                active_agent=session.get("active_agent", "general"),
            )

        current_grade = extract_grade(message)
        current_subject = subject_key(message)
        current_name = self._extract_name(message)
        if current_grade is not None:
            session["grade"] = current_grade
        if current_subject:
            session["subject"] = current_subject
        if current_name:
            session["name"] = current_name

        grade = current_grade or session.get("grade")
        subject = current_subject or session.get("subject") or session.get("query_text")

        pending_step = session.get("pending_match_step")
        if pending_step == "grade" and not grade:
            return AgentResult(
                reply="What grade is the mentee in?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_grade",
                active_agent=session.get("active_agent", "general"),
            )

        if pending_step == "grade" and grade and not subject:
            session["pending_match_step"] = "subject"
            return AgentResult(
                reply="What subject or skill do you want help with?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_subject",
                active_agent=session.get("active_agent", "general"),
            )

        if pending_step == "subject" and not subject:
            return AgentResult(
                reply="What subject or skill do you want help with?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_subject",
                active_agent=session.get("active_agent", "general"),
            )

        if not subject and not grade:
            session["pending_match_step"] = "grade"
            return AgentResult(
                reply="What grade is the mentee in?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_grade",
                active_agent=session.get("active_agent", "general"),
            )

        if not grade:
            session["pending_match_step"] = "grade"
            return AgentResult(
                reply="What grade is the mentee in?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_grade",
                active_agent=session.get("active_agent", "general"),
            )

        if not subject:
            session["pending_match_step"] = "subject"
            return AgentResult(
                reply="What subject or skill do you want help with?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_subject",
                active_agent=session.get("active_agent", "general"),
            )

        session["pending_match_step"] = None

        name = session.get("name") or current_name

        mentee, ranked = match_mentors(name or "Mentee", message, mentee_grade=grade, top_k=3)
        ranked = ranked[:3]
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
                active_agent=session.get("active_agent", "general"),
            )

        top_count = min(3, len(ranked))
        reply = (
            f"I found {top_count} match{'es' if top_count != 1 else ''}. "
            "The ranked mentors are shown below. Type 'book 1', 'book 2', or 'book 3' to confirm a pairing."
        )
        return AgentResult(
            reply=reply,
            state="showing_results",
            matches=ranked,
            active_agent=session.get("active_agent", "general"),
        )

    def _is_actionable_match_query(self, message: str) -> bool:
        normalized = _normalize(message)
        if not normalized:
            return False

        tokens = _token_set(normalized)
        # Very short queries are unlikely to be actionable matches unless they include a grade
        if len(tokens) < 3 and not extract_grade(normalized):
            return False

        # Hint words that indicate a match/search intent
        if _contains_any(tokens, MATCH_QUERY_HINT_WORDS):
            return True

        # Explicitly look for longer help-like queries
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
