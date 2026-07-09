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

from api.services import book_pairing_in_db, get_mentor_slots, list_available_mentors, match_mentors
from api.memory_store import get_memory_context, clear_session_memory
from api.email_service import send_booking_confirmation
from api.llm_provider import call_cerebras, get_llm_config
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

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b").strip() or "llama3.1-8b"
AUXILIUM_SUPPORT_EMAIL = "auxilium.mentorship@gmail.com"
SUPPORT_HANDOFF_MESSAGE = (
    f"I’m not sure how to help with that yet. If you want to reach the Auxilium coordinators, email {AUXILIUM_SUPPORT_EMAIL}. "
    "You can also try asking for a mentor match, a contest problem, or a question from the knowledge file."
)

UNKNOWN_REQUEST_MESSAGE = SUPPORT_HANDOFF_MESSAGE

LLM_NOT_CONFIGURED_MESSAGE = (
    f"I can hand off to the AI fallback, but `CEREBRAS_API_KEY` is not set yet. If you need help now, email the Auxilium coordinators at {AUXILIUM_SUPPORT_EMAIL}. "
    "Put `CEREBRAS_API_KEY` in the repo root `.env` file and restart the backend to re-enable AI fallback."
)

# Session states involved in the booking confirmation flow.
BOOKING_FLOW_STATES = {
    "awaiting_booking_name",
    "awaiting_booking_email",
    "awaiting_booking_confirmation",
    "awaiting_slot_selection",
}


@dataclass
class AgentResult:
    reply: str
    state: str
    matches: list[dict] | None = None
    booking_state: str | None = None
    active_agent: str | None = None


@dataclass
class PromptRewriteResult:
    target_agent: str
    cleaned_message: str
    formatted_prompt: str


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
        "universiy": 13
    }
    for key, value in grade_words.items():
        if re.search(rf"\b{re.escape(key)}\b", t):
            return value
    match = re.search(r"\b(9|10|11|12)\b", t)
    return int(match.group(1)) if match else None


def extract_choice(text: str) -> int | None:
    match = re.search(r"\b(?:book\s*)?(1|2|3)\b", text.lower())
    return int(match.group(1)) if match else None


def _extract_email(text: str) -> str | None:
    match = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def is_affirmative(text: str) -> bool:
    t = _normalize(text)
    return bool(re.search(r"^(y|yes|yeah|yep|yup|correct|confirm|confirmed|right|sure|ok|okay|sounds good|that's right|thats right)\b", t))


def is_negative(text: str) -> bool:
    t = _normalize(text)
    return bool(re.search(r"^(n|no|nope|nah|incorrect|wrong|that's wrong|thats wrong)\b", t))


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
    if session.get("state") in BOOKING_FLOW_STATES:
        return True
    t = _normalize(text)
    return bool(re.search(r"\bbook\b|confirm pairing|reserve mentor", t))


def is_match_request(text: str) -> bool:
    t = _normalize(text)
    return bool(
        re.search(r"\bfind\b.*\bmentor\b|\bmake\b.*\bmatch\b|\bmatch\b|\bpair\b|\bwant\b.*\bmentor\b|\bneed\b.*\bmentor\b", t)
        or re.search(r"\bneed help with\b|\blooking for\b.*\bmentor\b|\btutor\b|\bneed\b.*\btutor\b", t)
    )


def is_negative_match_request(text: str) -> bool:
    t = _normalize(text)
    return bool(
        re.search(r"\b(don'?t|do not|dont|no|not)\b.*\b(mentor|mentors|match|tutor|tutors|pair|pairing)\b", t)
        or re.search(r"\b(no|not)\b.*\b(match|matching)\b", t)
        or re.search(r"\b(i\s*do\s*not\s*want|i\s*don't\s*want|i\s*dont\s*want)\b.*\b(mentor|match|tutor|pair)\b", t)
    )


def is_contest_request(text: str) -> bool:
    t = _normalize(text)
    return bool(re.search(r"\bcontest(s)?\b|\bolympiad\b|\bcompetition\b|\bAMC\b|\bAIME\b|contest math|contest problems|practice contest", t))


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


def _extract_llm_text(data: dict[str, Any]) -> str | None:
    if not isinstance(data, dict):
        return None

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            choice_message = first_choice.get("message")
            if isinstance(choice_message, dict):
                content = choice_message.get("content")
                if isinstance(content, str) and content:
                    return content.strip()
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text") or part.get("content")
                            if isinstance(text, str) and text:
                                text_parts.append(text)
                    if text_parts:
                        return "".join(text_parts).strip()

    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        first_candidate = candidates[0]
        if isinstance(first_candidate, dict):
            candidate_text = first_candidate.get("content") or first_candidate.get("output") or first_candidate.get("text")
            if isinstance(candidate_text, str) and candidate_text:
                return candidate_text.strip()
            if isinstance(candidate_text, dict):
                inner_text = candidate_text.get("text") or candidate_text.get("output")
                if isinstance(inner_text, str) and inner_text:
                    return inner_text.strip()

    return None


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```(?:json)?\s*", "", trimmed, flags=re.I)
        trimmed = re.sub(r"\s*```$", "", trimmed)
    return trimmed.strip()


def _rewrite_user_message(message: str, session: dict[str, Any], forced_agent: str | None = None, intent: str | None = None) -> PromptRewriteResult:
    clean_message = message.strip()
    base_target = forced_agent or session.get("active_agent") or "general"
    fallback_prompt = (
        f"Internal agent request.\n"
        f"Target agent: {base_target}\n"
        f"User request: {clean_message}\n"
        f"Use the agent's normal rules to handle it."
    )

    api_key = _llm_api_key()
    if not api_key:
        return PromptRewriteResult(
            target_agent=base_target if base_target in {"general", "match", "contest"} else "unknown",
            cleaned_message=clean_message,
            formatted_prompt=fallback_prompt,
        )

    session_summary: list[str] = []
    for key in ("active_agent", "state", "grade", "subject", "name", "query_text"):
        value = session.get(key)
        if value not in (None, "", [], {}):
            session_summary.append(f"{key}: {value}")

    prompt = (
        "You rewrite raw user messages into clean internal prompts for Lumi's agents. "
        "Return JSON only with keys: target_agent, cleaned_message, formatted_prompt. "
        "target_agent must be one of general, match, contest, or unknown. "
        "If the user explicitly rejects mentors, matches, pairings, or tutoring, set target_agent to general. "
        "Do not answer the user. Preserve names, grades, contest names, dates, and constraints. "
        "Keep the cleaned message concise and explicit."
    )
    request = {
        "message": clean_message,
        "forced_agent": forced_agent,
        "intent_hint": intent,
        "session": session_summary,
    }

    try:
        data = call_cerebras(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            max_tokens=250,
            temperature=0.0,
        )
    except Exception:
        return PromptRewriteResult(
            target_agent=base_target if base_target in {"general", "match", "contest"} else "unknown",
            cleaned_message=clean_message,
            formatted_prompt=fallback_prompt,
        )

    content = _extract_llm_text(data)
    if not content:
        return PromptRewriteResult(
            target_agent=base_target if base_target in {"general", "match", "contest"} else "unknown",
            cleaned_message=clean_message,
            formatted_prompt=fallback_prompt,
        )

    try:
        payload = json.loads(_strip_code_fences(content))
    except json.JSONDecodeError:
        return PromptRewriteResult(
            target_agent=base_target if base_target in {"general", "match", "contest"} else "unknown",
            cleaned_message=clean_message,
            formatted_prompt=fallback_prompt,
        )

    target_agent = str(payload.get("target_agent") or base_target).strip().lower()
    if target_agent not in {"general", "match", "contest", "unknown"}:
        target_agent = "unknown"

    cleaned_message = str(payload.get("cleaned_message") or clean_message).strip() or clean_message
    formatted_prompt = str(payload.get("formatted_prompt") or fallback_prompt).strip() or fallback_prompt

    return PromptRewriteResult(
        target_agent=target_agent,
        cleaned_message=cleaned_message,
        formatted_prompt=formatted_prompt,
    )


def _contains_any(tokens: set[str], words: set[str]) -> bool:
    return bool(tokens & words)


def reset_session(session: dict[str, Any]) -> None:
    """Wipe per-conversation session state and the persistent memory store.

    Call this whenever a brand-new session/conversation begins (e.g. when the
    frontend opens a fresh chat) so leftover state - mentee details, active
    agent, pending bookings, persisted facts/examples from a previous session
    - doesn't bleed into the new one.
    """
    session.clear()
    session.update(
        {
            "state": "idle",
            "subject": None,
            "grade": None,
            "name": None,
            "query_text": None,
            "matches": [],
            "active_agent": "general",
            "pending_match_step": None,
            "pending_booking_choice": None,
            "pending_mentee_email": None,
            "pending_slot_id": None,
            "messages": [],
        }
    )
    try:
        clear_session_memory()
    except Exception:
        logger.exception("Failed to clear persistent session memory")


def _build_memory_context(session: dict[str, Any], limit: int = 24, max_chars: int = 4000) -> str:
    facts: list[str] = []
    name = session.get("name")
    grade = session.get("grade")
    subject = session.get("subject")
    active_agent = session.get("active_agent")
    if name:
        facts.append(f"known name: {name}")
    if grade:
        facts.append(f"known grade: {grade}")
    if subject:
        facts.append(f"known subject: {subject}")
    if active_agent:
        facts.append(f"active agent: {active_agent}")

    history = session.get("messages", [])
    recent_history = [
        item
        for item in history[-limit:]
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    history_lines = [f"{item['role']}: {str(item['content']).strip()}" for item in recent_history]

    blocks: list[str] = []
    if facts:
        blocks.append("Session facts:\n- " + "\n- ".join(facts))
    if history_lines:
        memory_text = "\n".join(history_lines)
        if len(memory_text) > max_chars:
            memory_text = memory_text[-max_chars:]
        blocks.append("Conversation memory:\n" + memory_text)

    return "\n\n".join(blocks)


SIMPLE_SMALL_TALK_HINTS = {
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank",
    "thank you",
    "ok",
    "okay",
    "cool",
    "great",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "what can you do",
    "help",
    "bye",
    "see you",
}

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


def _is_simple_small_talk(message: str) -> bool:
    normalized = _normalize(message)
    if not normalized:
        return False
    if normalized in SIMPLE_SMALL_TALK_HINTS:
        return True

    tokens = _token_set(normalized)
    if len(tokens) <= 4 and any(hint in normalized for hint in SIMPLE_SMALL_TALK_HINTS):
        return True

    return False


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


def _llm_api_key() -> str:
    api_key, _, _ = get_llm_config()
    return api_key.strip()


def _format_slot_list(slots: list[dict]) -> str:
    """Format a list of weekly time slots as a numbered list for chat display."""
    if not slots:
        return "(No slots available)"
    lines = []
    for i, slot in enumerate(slots, 1):
        lines.append(f"{i}. {slot['day_of_week']}s  {slot['start_time']}–{slot['end_time']}")
    return "\n".join(lines)


class MentorTaskAgents:
    def __init__(self) -> None:
        self.router = RunnableBranch(
            (lambda context: context["intent"] == "search", RunnableLambda(self._search_agent)),
            (lambda context: context["intent"] == "general", RunnableLambda(self._general_agent)),
            RunnableLambda(self._general_agent),
        )

    def run(self, session_id: str, message: str, session: dict[str, Any], forced_agent: str | None = None) -> AgentResult:
        intent = self._intent_for(message, session, forced_agent)
        rewrite = _rewrite_user_message(message, session, forced_agent, intent)
        session["last_agent_prompt"] = rewrite.formatted_prompt

        if forced_agent in {"general", "match", "contest"} and not is_agent_switch_request(message) and not is_negative_match_request(message):
            self._set_active_agent(session, forced_agent)

        # Natural-language auto-switch: set active agent based on the rewritten intent when it is explicit.
        if not is_agent_switch_request(message) and not is_negative_match_request(message):
            if rewrite.target_agent == "match" or (rewrite.target_agent == "unknown" and is_match_request(rewrite.cleaned_message)):
                self._set_active_agent(session, "match")
            elif rewrite.target_agent == "contest" or (rewrite.target_agent == "unknown" and is_contest_request(rewrite.cleaned_message)):
                self._set_active_agent(session, "contest")
            elif rewrite.target_agent == "general":
                self._set_active_agent(session, "general")

        payload = {
            "session_id": session_id,
            "message": message,
            "prompt_message": rewrite.formatted_prompt,
            "cleaned_message": rewrite.cleaned_message,
            "session": session,
            "intent": intent,
        }
        return self.router.invoke(payload)

    def _set_active_agent(self, session: dict[str, Any], agent: str) -> None:
        session["active_agent"] = agent
        if agent == "general":
            session["pending_match_step"] = None
            session["matches"] = []
            session["state"] = "idle"
        elif agent == "match" and session.get("state") not in {
            "awaiting_match_details",
            "showing_results",
            *BOOKING_FLOW_STATES,
        }:
            session["pending_match_step"] = None
            session["state"] = "idle"

    def _reset_match_flow(self, session: dict[str, Any]) -> None:
        session["subject"] = None
        session["grade"] = None
        session["name"] = None
        session["query_text"] = None
        session["matches"] = []
        session["pending_booking_choice"] = None
        session["pending_mentee_email"] = None
        session["pending_match_step"] = "grade"
        session["state"] = "awaiting_match_details"

    def _message_has_match_details(self, message: str) -> bool:
        return bool(extract_grade(message) or subject_key(message) or self._extract_name(message) or extract_choice(message))

    def _intent_for(self, message: str, session: dict[str, Any], forced_agent: str | None = None) -> str:
        if session.get("state") in BOOKING_FLOW_STATES:
            return "search"
        if is_agent_switch_request(message):
            return "general"
        if is_restart_request(message):
            return "general"
        if is_contest_request(message):
            return "general"
        if is_negative_match_request(message):
            return "general"
        # If user asks "how do I book" or similar, treat it as a general question, not an actionable booking command
        if re.search(r"\bhow\b", message, re.I) and re.search(r"\bbook\b", message, re.I):
            return "general"
        if is_help_request(message) or is_list_request(message):
            return "general"
        if forced_agent == "general":
            return "general"
        if forced_agent in {"booking", "match"}:
            return "search"
        if session.get("active_agent") == "match":
            return "search"
        if session.get("pending_match_step") in {"grade", "subject"}:
            return "search"
        if is_booking_request(message, session):
            return "search"
        if is_match_request(message):
            return "search"
        if extract_grade(message) or subject_key(message):
            return "search"
        return "general"

    def _general_agent(self, context: dict[str, Any]) -> AgentResult:
        message = context["message"]
        session = context["session"]
        prompt_message = context.get("prompt_message") or message

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
                    session["subject"] = None
                    session["grade"] = None
                    session["name"] = None
                    session["query_text"] = None
                    session["matches"] = []
                    session["pending_booking_choice"] = None
                    session["pending_mentee_email"] = None
                    reply = (
                        "Switched to the Match agent.\n\n"
                        "What grade is the mentee in?"
                    )
                    session["pending_match_step"] = "grade"
                else:
                    session["pending_match_step"] = None
                    session["matches"] = []
                    session["state"] = "idle"
                    reply = (
                        "Switched to the General agent.\n\n"
                        "You can ask me questions from the knowledge file or request another mode anytime."
                    )
                return AgentResult(
                    reply=reply,
                    state="awaiting_match_details" if target == "match" else "idle",
                    active_agent=target,
                    booking_state="needs_grade_and_subject" if target == "match" else "idle",
                )

        if is_restart_request(message):
            reset_session(session)
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
        if not is_negative_match_request(message) and is_match_request(message):
            session["active_agent"] = "match"
            session["pending_match_step"] = "grade"
            session["state"] = "awaiting_match_details"
            return AgentResult(
                reply=("Switched to the Match agent.\n\nWhat grade is the mentee in?"),
                state="awaiting_match_details",
                active_agent="match",
                booking_state="needs_grade_and_subject",
            )

        if not is_negative_match_request(message) and is_contest_request(message):
            session["active_agent"] = "contest"
            session["state"] = "idle"
            return AgentResult(
                reply=(
                    "Switched to the Contest agent.\n\nAsk for a contest problem, a solution explanation, or say which contest you want to practice."
                ),
                state="idle",
                active_agent="contest",
            )

        general_answer = self._answer_general_question(message, session, prompt_message)
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

    def _answer_general_question(self, message: str, session: dict[str, Any], prompt_message: str | None = None) -> str | None:
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
        # bypass the KB and send directly to the configured LLM for best-effort processing.
        if _is_transformation_request(message):
            chat_answer = self._answer_from_free_chat(prompt_message or message, session, force_date=False)
            if chat_answer:
                return chat_answer

        file_answer = self._answer_from_general_knowledge(message)
        if file_answer:
            return file_answer

        chat_answer = self._answer_from_free_chat(prompt_message or message, session)
        if chat_answer:
            return chat_answer

        if not _llm_api_key():
            return LLM_NOT_CONFIGURED_MESSAGE

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

        memory_context = _build_memory_context(session)
        persistent_memory = get_memory_context()
        if persistent_memory:
            memory_context = (memory_context + "\n\n" if memory_context else "") + persistent_memory
        if memory_context:
            memory_context = f"Use the following conversation memory when answering.\n{memory_context}\n\n"

        prompt = (
            f"You are Lumi's general fallback agent for education and Auxilium support. {date_context}"
            f"{memory_context}"
            "Use this assistant for school subjects, tutoring, contest problems, mentor matching, and Auxilium app help. "
            "If the user is asking for casual conversation, greetings, or profile-related updates, answer briefly and naturally. "
            "If the user asks something outside education or Auxilium, politely decline and redirect them back to tutoring, school, contests, mentor matching, or app support. "
            "The local knowledge file did not contain an answer for this question. Provide a concise, helpful reply using any recent conversation context supplied. "
            "If uncertain, give a best-effort response and say so clearly."
        )
        cerebras_api_key = _llm_api_key()
        if not cerebras_api_key:
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

        try:
            data = call_cerebras(
                messages,
                max_tokens=1024,
                temperature=0.6,
            )
        except requests.RequestException as exc:
            logger.warning("Cerebras fallback request failed: %s", exc)
            return None
        except ValueError as exc:
            logger.warning("Cerebras fallback returned invalid JSON: %s", exc)
            return None

        return _extract_llm_text(data)

    def _build_booking_summary(self, session: dict[str, Any], mentor: dict[str, Any]) -> str:
        mentee_name = session.get("name") or "Mentee"
        mentee_email = session.get("pending_mentee_email") or ""
        subject = session.get("subject") or mentor["subject"]
        grade = int(session.get("grade") or 0)
        slot_label = session.get("pending_slot_label") or "No slot selected"
        return (
            "Here's a summary of the booking:\n\n"
            f"Mentee: {mentee_name}\n"
            f"Mentee email: {mentee_email}\n"
            f"Grade: {grade}\n"
            f"Subject: {subject}\n"
            f"Mentor: {mentor['name']} (Grade {mentor['grade']})\n"
            f"Time slot: {slot_label}\n\n"
            "Is this correct? (yes/no)"
        )

    def _search_agent(self, context: dict[str, Any]) -> AgentResult:
        message = (context.get("cleaned_message") or context["message"]).strip()
        session = context["session"]
        prompt_message = context.get("prompt_message") or message
        matches = session.get("matches", [])
        choice = extract_choice(message)

        if is_negative_match_request(message):
            return AgentResult(
                reply="Okay, I won't switch you to the Match agent. Tell me what you want instead.",
                state="idle",
                matches=[],
                booking_state="general",
                active_agent="general",
            )

        if is_agent_switch_request(message):
            target = _target_agent(message)
            if target:
                session["active_agent"] = target
                session["state"] = "idle"
                if target == "general":
                    return AgentResult(
                        reply="Switched to the General agent. What would you like to do next?",
                        state="idle",
                        matches=[],
                        booking_state="switched_general",
                        active_agent="general",
                    )
                if target == "contest":
                    return AgentResult(
                        reply="Switched to the Contest agent. Ask for a contest problem or an explanation.",
                        state="idle",
                        matches=[],
                        booking_state="switched_contest",
                        active_agent="contest",
                    )

        if _is_simple_small_talk(message) and not matches:
            return AgentResult(
                reply="I can help with mentor matching, school subjects, and contest problems. If you want, say 'switch to general' to chat normally.",
                state=session.get("state", "idle"),
                matches=[],
                booking_state=session.get("state", "idle"),
                active_agent=session.get("active_agent", "match"),
            )

        if session.get("active_agent") == "match" and not is_booking_request(message, session):
            has_match_details = self._message_has_match_details(message)
            if session.get("pending_match_step") is None and not has_match_details:
                self._reset_match_flow(session)
                return AgentResult(
                    reply="What grade is the mentee in?",
                    state="awaiting_match_details",
                    matches=[],
                    booking_state="needs_grade",
                    active_agent="match",
                )

        # ------------------------------------------------------------------
        # Slot selection: user picks a time slot after choosing a mentor
        # ------------------------------------------------------------------
        if session.get("state") == "awaiting_slot_selection":
            pending_slots = session.get("pending_slots", [])
            # Accept a number like "1", "2", ... or "slot 1" etc.
            slot_choice_match = re.search(r"\b(\d+)\b", message)
            if not slot_choice_match or not pending_slots:
                slot_lines = _format_slot_list(pending_slots)
                return AgentResult(
                    reply=f"Please enter the number of the time slot you want.\n\n{slot_lines}",
                    state="awaiting_slot_selection",
                    matches=matches,
                    booking_state="awaiting_slot",
                    active_agent=session.get("active_agent", "general"),
                )

            slot_idx = int(slot_choice_match.group(1)) - 1
            if slot_idx < 0 or slot_idx >= len(pending_slots):
                slot_lines = _format_slot_list(pending_slots)
                return AgentResult(
                    reply=f"That number isn't in the list. Please choose between 1 and {len(pending_slots)}.\n\n{slot_lines}",
                    state="awaiting_slot_selection",
                    matches=matches,
                    booking_state="awaiting_slot",
                    active_agent=session.get("active_agent", "general"),
                )

            chosen_slot = pending_slots[slot_idx]
            session["pending_slot_id"] = chosen_slot["id"]
            session["pending_slot_label"] = f"{chosen_slot['day_of_week']}s  {chosen_slot['start_time']}–{chosen_slot['end_time']}"
            session["state"] = "awaiting_booking_name"
            return AgentResult(
                reply=f"Great — you picked **{chosen_slot['day_of_week']}s {chosen_slot['start_time']}–{chosen_slot['end_time']}**.\n\nWhat is the mentee's name so I can save the booking?",
                state="awaiting_booking_name",
                matches=matches,
                booking_state="awaiting_name",
                active_agent=session.get("active_agent", "general"),
            )

        # ------------------------------------------------------------------
        # Booking confirmation flow: name -> email -> confirmation -> booked
        # ------------------------------------------------------------------
        if session.get("state") == "awaiting_booking_name":
            mentee_name = message.strip().title()
            if len(mentee_name) < 2:
                return AgentResult(
                    reply="Please enter the mentee's name.",
                    state="awaiting_booking_name",
                    matches=matches,
                    booking_state="awaiting_name",
                    active_agent=session.get("active_agent", "general"),
                )

            booking_choice = int(session.get("pending_booking_choice") or 1)
            if booking_choice < 1 or booking_choice > len(matches):
                return AgentResult(reply="I lost the selected mentor. Please run the search again.", state="idle", matches=[], booking_state="booking_lost")

            session["name"] = mentee_name
            session["state"] = "awaiting_booking_email"
            return AgentResult(
                reply=f"Thanks, {mentee_name}. What is the mentee's email address so I can send the booking confirmation?",
                state="awaiting_booking_email",
                matches=matches,
                booking_state="awaiting_email",
                active_agent=session.get("active_agent", "general"),
            )

        if session.get("state") == "awaiting_booking_email":
            booking_choice = int(session.get("pending_booking_choice") or 1)
            if booking_choice < 1 or booking_choice > len(matches):
                return AgentResult(reply="I lost the selected mentor. Please run the search again.", state="idle", matches=[], booking_state="booking_lost")

            mentee_email = _extract_email(message)
            if not mentee_email:
                return AgentResult(
                    reply="That doesn't look like a valid email address. Please enter the mentee's email (e.g. name@example.com).",
                    state="awaiting_booking_email",
                    matches=matches,
                    booking_state="awaiting_email",
                    active_agent=session.get("active_agent", "general"),
                )

            session["pending_mentee_email"] = mentee_email
            session["state"] = "awaiting_booking_confirmation"

            mentor = matches[booking_choice - 1]
            return AgentResult(
                reply=self._build_booking_summary(session, mentor),
                state="awaiting_booking_confirmation",
                matches=matches,
                booking_state="awaiting_confirmation",
                active_agent=session.get("active_agent", "general"),
            )

        if session.get("state") == "awaiting_booking_confirmation":
            booking_choice = int(session.get("pending_booking_choice") or 1)
            if booking_choice < 1 or booking_choice > len(matches):
                return AgentResult(reply="I lost the selected mentor. Please run the search again.", state="idle", matches=[], booking_state="booking_lost")

            if is_negative(message):
                self._reset_match_flow(session)
                session["active_agent"] = "match"
                return AgentResult(
                    reply="No problem - let's try again. What grade is the mentee in?",
                    state="awaiting_match_details",
                    matches=[],
                    booking_state="needs_grade",
                    active_agent="match",
                )

            if not is_affirmative(message):
                mentor = matches[booking_choice - 1]
                return AgentResult(
                    reply="Sorry, I didn't catch that. " + self._build_booking_summary(session, mentor),
                    state="awaiting_booking_confirmation",
                    matches=matches,
                    booking_state="awaiting_confirmation",
                    active_agent=session.get("active_agent", "general"),
                )

            mentor = matches[booking_choice - 1]
            mentee_name = session.get("name") or "Mentee"
            mentee_email = session.get("pending_mentee_email") or ""
            subject = session.get("subject") or mentor["subject"]
            grade = int(session.get("grade") or 0)

            try:
                self._save_booking(session, mentor, mentee_name, mentee_email)
            except ValueError as e:
                # Someone else booked this mentor/slot between the match being
                # shown and the user confirming it. Don't 500 — let them pick again.
                self._reset_booking_state(session)
                session["active_agent"] = "match"
                return AgentResult(
                    reply=(
                        f"Sorry — {mentor['name']} just became unavailable "
                        f"({e}). Let's find you another match. What grade is "
                        "the mentee in?"
                    ),
                    state="awaiting_match_details",
                    matches=[],
                    booking_state="needs_grade",
                    active_agent="match",
                )

            email_sent = False
            try:
                email_sent = send_booking_confirmation(
                    mentee_email=mentee_email,
                    mentee_name=mentee_name,
                    mentor_name=mentor["name"],
                    subject=subject,
                    grade=grade,
                    slot_label=session.get("pending_slot_label") or "",
                )
            except Exception:
                logger.exception("Failed to send booking confirmation email")

            self._reset_booking_state(session)
            session["active_agent"] = "general"

            reply_lines = [
                "Booked!",
                "",
                f"{mentee_name} -> {mentor['name']}",
                f"{subject} - Grade {grade} -> Grade {mentor['grade']}",
            ]
            if mentee_email:
                if email_sent:
                    reply_lines.append("")
                    reply_lines.append(f"A confirmation email has been sent to {mentee_email}.")
                else:
                    reply_lines.append("")
                    reply_lines.append(
                        "I couldn't send the confirmation email automatically, but the booking is saved."
                    )
            reply_lines.append("")
            reply_lines.append("Is there anything else I can help you with?")

            return AgentResult(
                reply="\n".join(reply_lines),
                state="idle",
                matches=[],
                booking_state="booked",
                active_agent="general",
            )

        if matches and choice:
            if choice > len(matches):
                return AgentResult(
                    reply="Tell me which result to book: book 1, book 2, or book 3.",
                    state=session.get("state", "showing_results"),
                    matches=matches,
                    booking_state="awaiting_choice",
                )

            session["pending_booking_choice"] = choice
            mentor = matches[choice - 1]

            # Fetch available time slots for the chosen mentor
            try:
                available_slots = get_mentor_slots(mentor["name"], only_available=True)
            except Exception:
                available_slots = []

            if available_slots:
                session["pending_slots"] = available_slots
                session["state"] = "awaiting_slot_selection"
                slot_lines = _format_slot_list(available_slots)
                return AgentResult(
                    reply=f"Great choice! **{mentor['name']}** has the following available time slots. Enter the number to pick one:\n\n{slot_lines}",
                    state="awaiting_slot_selection",
                    matches=matches,
                    booking_state="awaiting_slot",
                    active_agent=session.get("active_agent", "general"),
                )

            # No slots configured — skip straight to name collection
            session["pending_slot_id"] = None
            session["pending_slot_label"] = "To be arranged"

            if not session.get("name"):
                session["state"] = "awaiting_booking_name"
                return AgentResult(
                    reply="What is the mentee's name so I can save the booking?",
                    state="awaiting_booking_name",
                    matches=matches,
                    booking_state="awaiting_name",
                    active_agent=session.get("active_agent", "general"),
                )

            if not session.get("pending_mentee_email"):
                session["state"] = "awaiting_booking_email"
                return AgentResult(
                    reply=f"What is {session['name']}'s email address so I can send the booking confirmation?",
                    state="awaiting_booking_email",
                    matches=matches,
                    booking_state="awaiting_email",
                    active_agent=session.get("active_agent", "general"),
                )

            session["state"] = "awaiting_booking_confirmation"
            return AgentResult(
                reply=self._build_booking_summary(session, mentor),
                state="awaiting_booking_confirmation",
                matches=matches,
                booking_state="awaiting_confirmation",
                active_agent=session.get("active_agent", "general"),
            )

        current_grade = extract_grade(message)
        current_subject = subject_key(message)
        current_name = self._extract_name(message)
        if current_grade is not None:
            session["grade"] = current_grade
        if current_subject:
            session["subject"] = current_subject
            session["query_text"] = message.strip()
        if current_name:
            session["name"] = current_name

        grade = current_grade or session.get("grade")
        subject = current_subject or session.get("subject") or session.get("query_text")

        pending_step = session.get("pending_match_step")
        if pending_step == "grade" and current_grade is None:
            return AgentResult(
                reply="What grade is the mentee in?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_grade",
                active_agent=session.get("active_agent", "general"),
            )

        if pending_step == "grade" and current_grade is not None and not subject:
            session["pending_match_step"] = "subject"
            return AgentResult(
                reply="What subject or skill do you want help with?",
                state="awaiting_match_details",
                matches=[],
                booking_state="needs_subject",
                active_agent=session.get("active_agent", "general"),
            )

        if pending_step == "subject" and not current_subject:
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
        match_query = (session.get("query_text") or session.get("subject") or message).strip()

        mentee, ranked = match_mentors(name or "Mentee", match_query, mentee_grade=grade, top_k=3)
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

    def _save_booking(self, session: dict[str, Any], mentor: dict[str, Any], mentee_name: str, mentee_email: str = "") -> None:
        subject = session.get("subject") or mentor.get("subject") or "General"
        grade = int(session.get("grade") or 0)
        slot_id = session.get("pending_slot_id")
        slot_label = session.get("pending_slot_label") or ""
        book_pairing_in_db(
            mentor_name=mentor["name"],
            mentee_name=mentee_name,
            subject=subject,
            mentor_grade=int(mentor.get("grade") or 0),
            mentee_grade=grade,
            match_score=float(mentor.get("match_score") or 0.0),
            explanation=str(mentor.get("explanation") or "Potential match."),
            mentee_email=mentee_email,
            slot_id=slot_id if isinstance(slot_id, int) else None,
            slot_label=slot_label,
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
                "pending_mentee_email": None,
                "pending_slot_id": None,
                "pending_slot_label": None,
                "pending_slots": None,
                "matches": [],
            }
        )

    def _extract_name(self, message: str) -> str | None:
        match = re.search(r"\b(?:my name is|i am|i'm|im)\s+([a-z][a-z\-']+(?:\s+[a-z][a-z\-']+)*)", message, re.I)
        if match:
            return match.group(1).strip().title()
        return None