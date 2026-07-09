"""
api/contest_agent.py

LangChain-style agent for Waterloo Math Contest knowledge.

v5 — fixes:
  - solution_text now always fetched via get_by_contest_year (not from query results
    which don't include it)
  - handle_search falls back to Gemini for general questions instead of just RAG
  - system prompt overhauled: Gemini responds naturally like an AI assistant,
    trusts problem/solution text it's given, never says a problem is impossible
  - conversation history passed to every Gemini call for better follow-up handling
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

try:
    import certifi_win32  # noqa: F401
except Exception:
    certifi_win32 = None

from api.llm_provider import call_cerebras
from api.agents import _rewrite_user_message
from api.problem_set_service import build_problem_set_from_text, is_problem_set_request
from rag.contest_retriever import (
    collection_count,
    get_by_contest_year,
    list_available_contests,
    query as chroma_query,
)
from rag.contest_ingestor import (
    TOPIC_KEYWORDS,
    CONTEST_GRADES,
    discover_contest_files,
    pair_contest_files,
)

# ── LLM client ─────────────────────────────────────────────────────────────

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
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if os.environ.get(key, "").strip():
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value


_load_dotenv_file()

_llm_init_error = ""

try:
    from api.llm_provider import get_llm_config
    _, _model, _ = get_llm_config()
    if not _model:
        _llm_init_error = "CEREBRAS_MODEL is not set"
except Exception as e:
    _llm_init_error = str(e)

_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b").strip() or "llama3.1-8b"

_SYSTEM_MATH = """\
You are Lumi, a friendly and knowledgeable AI tutor specialising in Waterloo Math Contests \
(Euclid, Fermat, Cayley, Pascal, Gauss, CIMC, CSMC, Fryer, Galois, Hypatia).

Behave like a capable AI assistant — answer questions naturally and conversationally, \
the same way ChatGPT or Gemini would. You can answer general math questions, explain \
concepts, help with study strategies, and discuss contest problems.

When you are given a contest problem and/or official solution in the prompt:
- TRUST the provided text completely. It is the real, official problem and solution.
- NEVER say a problem is impossible, unsolvable, or lacks information. It is a real \
  competition problem with a real answer. Work with what you are given.
- If only the problem text is given (no solution), reason through it yourself step by step.
- CRITICAL: Only explain the single problem given to you in the PROBLEM/OFFICIAL SOLUTION \
  text above. That text is the complete boundary of what you may discuss.
- Do NOT continue on to the next problem number, even if you recognize this contest and \
  remember what comes next from your training data. For example, if given "Problem 1" \
  with parts (a), (b), (c), your response must stop after part (c) — do NOT add a \
  "Problem 2" or "2 (a)" section under any circumstances.
- If you finish explaining the given problem and are tempted to keep going with "the next \
  problem was...", stop immediately instead. The response should end right after the \
  final "Answer:" line for the problem you were given.

FORMATTING — always follow these rules:
- Do NOT use Markdown headers (no "#", "##", "###"). This chat only renders plain text \
  and **bold** — headers show up as literal hashtag characters. For section titles, use \
  **bold** text instead, e.g. "**Part (a) — Evaluate the expression**" on its own line.
- Do NOT use "---" as a horizontal rule/divider between sections either, since it also \
  renders as literal characters. Use a blank line to separate sections instead.
- Structure explanations with clear sections. Use blank lines between sections.
- Number every step: "1.", "2.", "3." etc.
- Put each step on its own line with a blank line after it.
- Use plain Unicode math only: ×, ÷, √, ², ³, ≤, ≥, ≠, π, θ, ±, °, ∠, △. NEVER use LaTeX \
  commands like \\frac, \\sqrt, \\dfrac, \\left, \\right, \\(, \\), \\[, \\], or any backslash. \
  Write fractions as "(a)/(b)" or "a divided by b", not \\frac{a}{b}. \
  Write square roots as "√(x)", not \\sqrt{x}. \
  Write angles as "∠PQR", degrees as "60°", and trig functions as plain text: sin, cos, tan. \
  Example of WRONG output: \\(\\frac{23-32}{32-23}\\) \
  Example of CORRECT output: (23−32)/(32−23)
- When substituting values into an equation, write the full substituted equation on its \
  own line rather than compressed inline fractions (e.g. write "sin(30°) = 1/2" then \
  "a / (1/2) = b / (4/5)" as a labeled step, not squeezed together). \
  Never add footnote-style markers like a trailing ".1" or ".2" after an equation.
- For the key insight, write it on its own line starting with "Key insight:"
- Keep each step focused on one idea. Do not write walls of text.
- End with the final answer clearly stated on its own line: "Answer: ..."

For concept explanations:
- Definition first, then techniques, then a contest tip.
- Use bullet points for lists of techniques or formulas.

For general conversation:
- Be concise and natural. No need for numbered steps unless explaining math.
- Warm and encouraging tone suitable for a high school student."""


def _delatexify(text: str) -> str:
    """Convert common LaTeX math notation to plain Unicode, and strip LaTeX delimiters."""
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
    text = re.sub(r"\\displaystyle|\\textstyle|\\scriptstyle", "", text)

    sqrt_re = re.compile(r"\\sqrt\{([^{}]*)\}")
    for _ in range(4):
        new_text = sqrt_re.sub(r"√(\1)", text)
        if new_text == text:
            break
        text = new_text

    frac_re = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
    for _ in range(4):
        new_text = frac_re.sub(lambda m: f"({m.group(1)})/({m.group(2)})", text)
        if new_text == text:
            break
        text = new_text

    superscripts = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
    def repl_sup(m):
        exp = m.group(1)
        return "".join(superscripts.get(ch, ch) for ch in exp)
    text = re.sub(r"\^\{([^{}]*)\}", repl_sup, text)
    text = re.sub(r"\^(\d)", lambda m: superscripts.get(m.group(1), m.group(1)), text)
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)

    replacements = {
        r"\times": "×", r"\cdot": "·", r"\div": "÷",
        r"\leq": "≤", r"\geq": "≥", r"\neq": "≠",
        r"\pi": "π", r"\theta": "θ", r"\pm": "±",
        r"\infty": "∞", r"\approx": "≈",
        r"\circ": "°",
        r"\angle": "∠", r"\triangle": "△",
        r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
        r"\cot": "cot", r"\sec": "sec", r"\csc": "csc",
        r"\log": "log", r"\ln": "ln",
        r"\overline": "", r"\underline": "",
        r"\;": " ", r"\,": " ", r"\!": "",
        r"\left": "", r"\right": "",
        r"\dfrac": "", r"\text": "",
    }
    for latex, unicode_char in replacements.items():
        text = text.replace(latex, unicode_char)

    generic_re = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")
    for _ in range(4):
        new_text = generic_re.sub(r"\1", text)
        if new_text == text:
            break
        text = new_text

    # For any OTHER backslash command we didn't explicitly handle, keep the
    # word itself (drop only the backslash) rather than deleting it outright.
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")

    # Strip Markdown headers and horizontal rules the frontend can't render —
    # keep the text but drop the leading "#"s / "---" markers.
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"[ \t]{2,}", " ", text)

    return text


def _grok(messages: list[dict], max_tokens: int = 1200) -> str:
    """Call the Cerebras API and return the assistant text."""
    try:
        payload_messages = [{"role": "system", "content": _SYSTEM_MATH}] + [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in messages
        ]
        data = call_cerebras(payload_messages, max_tokens=max_tokens, temperature=0.2)
    except Exception as e:
        import traceback
        print(f"[DEBUG] Cerebras call failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return f"_(AI explanation unavailable: {e})_"

    if not isinstance(data, dict):
        print(f"[DEBUG] Cerebras returned non-dict data: {type(data)} = {data!r}")
        return "_(AI explanation unavailable: unexpected LLM response.)_"

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    content = content.strip()
                    if _is_garbled(content):
                        print(f"[DEBUG] Content flagged as garbled by _is_garbled(): {content!r}")
                        return "_(AI explanation unavailable: garbled model output)_"
                    return _delatexify(content)
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text") or part.get("content")
                            if isinstance(text, str) and text:
                                text_parts.append(text)
                    if text_parts:
                        content = "".join(text_parts).strip()
                        if content:
                            return _delatexify(content)
                print(f"[DEBUG] message.content was empty/unrecognized: {message!r}")
            else:
                print(f"[DEBUG] first_choice had no 'message' dict: {first_choice!r}")
        else:
            print(f"[DEBUG] choices[0] was not a dict: {first_choice!r}")
    else:
        print(f"[DEBUG] No usable 'choices' in response: {data!r}")

    return "_(AI explanation unavailable: LLM returned no text.)_"


def _local_solution_explanation(problem_text: str, solution_text: str) -> str:
    """Produce a grounded fallback explanation from noisy official solution text."""
    def clean(line: str) -> str:
        raw = line.strip()
        for prefix in ("PROBLEM:", "OFFICIAL SOLUTION:", "SOLUTION:"):
            if raw.upper().startswith(prefix):
                return raw[len(prefix):].strip()
        return raw

    raw_lines = [clean(line) for line in solution_text.splitlines() if clean(line)]

    def is_header_or_footer(line: str) -> bool:
        if re.match(r"^Page\s+\d+$", line, re.I):
            return True
        if re.match(r"^\d{4}\s+.*Contest\s+Solutions?$", line, re.I):
            return True
        return False

    def is_noisy_fragment(line: str) -> bool:
        # Drop heavily fragmented lines like "S ce ... s t e" from broken font maps.
        low = line.lower()
        if any(k in low for k in ("therefore", "thus", "since", "solution", "equation")):
            return False
        if "=" in line:
            return False
        tokens = re.findall(r"[A-Za-z]+", line)
        if not tokens:
            # Pure numeric or symbol-only fragments rarely carry useful explanation text.
            return bool(re.match(r"^[\d\s\-+/=().,]+$", line))
        short = sum(1 for t in tokens if len(t) == 1)
        short_ratio = short / max(1, len(tokens))
        if len(tokens) >= 6 and short_ratio >= 0.30:
            return True
        if len(tokens) >= 8 and (sum(len(t) for t in tokens) / len(tokens)) < 2.5 and max(len(t) for t in tokens) <= 3:
            return True
        if re.search(r"\b(?:[a-z]\s+){4,}[a-z]\b", low):
            return True
        if len(line) < 8 and not re.search(r"[=<>+\-×÷]", line):
            return True
        if len(line) < 12 and short >= 2 and not re.search(r"[=<>+\-×÷]", line):
            return True
        return False

    cleaned_lines: list[str] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        nxt = raw_lines[i + 1] if i + 1 < len(raw_lines) else ""

        # Stop before the next problem starts, e.g. "3." followed by "(a) ..."
        if re.match(r"^\d{1,2}\.$", line) and re.match(r"^\([a-z]\)", nxt, re.I):
            break

        if re.match(r"^\([a-z]\)", line, re.I):
            cleaned_lines.append(line)
            i += 1
            continue

        if not is_header_or_footer(line) and not is_noisy_fragment(line) and not _is_garbled(line):
            cleaned_lines.append(line)
        i += 1

    cleaned_lines = [
        line for line in cleaned_lines
        if not re.match(r"^(Part \([a-z]\)|A central equation used is|The conclusion is|In words|The official solution|Official solution summary)", line, re.I)
    ]
    cleaned_lines = [line for line in cleaned_lines if len(line.strip()) >= 6]

    if not cleaned_lines:
        return (
            "Official solution summary:\n\n"
            "The official solution text looks too noisy to summarize reliably, so here is the safest study approach:\n\n"
            "1. Translate each condition into an equation or inequality.\n"
            "2. Solve the resulting algebraic relationships step by step.\n"
            "3. Check the final answer against the original constraints."
        )

    if cleaned_lines:
        # Split into (a), (b), (c) parts when available.
        parts: dict[str, list[str]] = {}
        current = ""
        for line in cleaned_lines:
            m = re.match(r"^\(([a-z])\)\s*(.*)$", line, re.I)
            if m:
                current = m.group(1).lower()
                rest = m.group(2).strip()
                parts.setdefault(current, [])
                if rest:
                    parts[current].append(rest)
                continue
            if current:
                parts.setdefault(current, []).append(line)

        def summarize_part(label: str, lines: list[str]) -> str:
            # Keep useful full lines only.
            useful = [ln for ln in lines if len(ln) >= 10]
            eq_lines = [ln for ln in useful if "=" in ln]
            conclusion = ""
            for ln in reversed(useful):
                low = ln.lower()
                if any(k in low for k in ("therefore", "thus", "gives", "so ")):
                    conclusion = ln
                    break
            if not conclusion and useful:
                conclusion = useful[-1]

            sentences: list[str] = []
            sentences.append(f"Part ({label}): The official solution sets up the key relationships from the problem conditions and then solves them step by step.")
            if eq_lines:
                first_eq = eq_lines[0]
                sentences.append(f"A central equation used is: {first_eq}")
            if conclusion:
                sentences.append(f"The conclusion is: {conclusion}")
            if len(useful) >= 2:
                sentences.append(f"In words: {useful[0]} {useful[1]}")
            return "\n".join(sentences)

        if parts:
            ordered = [k for k in ("a", "b", "c", "d", "e") if k in parts]
            blocks = [summarize_part(k, parts[k]) for k in ordered]
            return "Official solution summary:\n\n" + "\n\n".join(blocks)

        # No explicit part markers: still return coherent sentences.
        useful = [ln for ln in cleaned_lines if len(ln) >= 10]
        if useful:
            top = useful[:5]
            return (
                "Official solution summary:\n\n"
                "The official solution proceeds by translating the conditions into equations and solving them carefully.\n"
                f"Key step: {top[0]}\n"
                + (f"Supporting step: {top[1]}\n" if len(top) > 1 else "")
                + (f"Conclusion: {top[-1]}" if top else "")
            )

    problem_lines = [line.strip() for line in problem_text.splitlines() if line.strip()]
    if problem_lines:
        return (
            "I could not access the official solution text right now, so here is the exact problem statement:\n\n"
            + "\n".join(problem_lines)
        )

    return "I could not access the solution text right now. Please try again."


def _grok_or_local(messages: list[dict], problem_text: str, solution_text: str, max_tokens: int = 1200) -> str:
    """Use the configured LLM when available; otherwise fall back to a grounded local explanation."""
    reply = _grok(messages, max_tokens=max_tokens)
    if reply.startswith("_(AI explanation unavailable:"):
        return _local_solution_explanation(problem_text, solution_text)
    return reply


def _is_garbled(text: str) -> bool:
    """Return True if `text` looks like garbled/OCR-output (many isolated letters or low letter density).

    Heuristic rules:
    - If >20% of whitespace-separated tokens are single letters (and total tokens>10),
      it's likely broken OCR where letters are split by spaces.
    - If the proportion of "meaningful" characters (letters, digits, and common math
      symbols) to total characters is very low, it's likely noisy. Digits and math
      symbols are included here (not just letters) so that legitimate math-heavy
      answers (lots of numbers, +, -, =, parentheses, etc.) aren't misflagged as
      garbled OCR output.
    - If there are many repeated isolated letters like "e q u a d a t", flag as garbled.
    """
    if not text or len(text) < 20:
        return False
    import re

    tokens = re.findall(r"\S+", text)
    if not tokens:
        return True
    single_letter = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    if len(tokens) > 10 and (single_letter / len(tokens)) > 0.20:
        return True

    meaningful = re.findall(r"[A-Za-z0-9+\-*/=<>().,\\{}^_√×÷≤≥≠π]", text)
    if len(meaningful) / max(1, len(text)) < 0.5:
        return True

    # Detect sequences of spaced single letters (e.g., 'e x a m p l e')
    spaced_letters = re.search(r"(?:\b[A-Za-z]\b[\s\W]+){4,}", text)
    if spaced_letters:
        return True

    return False


def _build_history(session: dict, n: int = 6) -> list[dict]:
    """Return the last n turns of conversation history for Gemini."""
    history = []
    for turn in session.get("messages", [])[-n:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content and role in ("user", "assistant"):
            history.append({"role": role, "content": content})
    return history


def _trim_solution_to_problem(solution_text: str, prob_num: int | None) -> str:
    """Cut solution_text down to just the section for `prob_num`.

    The first problem in a solution PDF often has no leading "N." marker
    (it may have been consumed by ingestion/page headers), so it starts
    directly at "(a)". Later problems are marked with "N.\n(a)" style
    headings. To handle both cases: if an explicit heading for `prob_num`
    is found, start there; otherwise, if `prob_num` is smaller than every
    heading found (i.e. it's the first, unlabeled problem), start at the
    beginning of the text. Either way, stop at the first heading for a
    different, larger problem number.
    """
    if not prob_num or not solution_text:
        return solution_text

    heading_re = re.compile(r"(?<!\d)(\d{1,2})\.?\s*\(a\)")
    matches = list(heading_re.finditer(solution_text))

    start_idx = None
    end_idx = len(solution_text)

    for m in matches:
        num = int(m.group(1))
        if num == prob_num and start_idx is None:
            start_idx = m.start()
            continue
        if start_idx is not None and num != prob_num:
            end_idx = m.start()
            break

    if start_idx is None:
        # No explicit heading for this problem — likely the first, unlabeled
        # problem in the document. Start from the beginning and stop at the
        # first heading belonging to a different (higher) problem number.
        if all(int(m.group(1)) > prob_num for m in matches) if matches else True:
            start_idx = 0
            for m in matches:
                if int(m.group(1)) != prob_num:
                    end_idx = m.start()
                    break
        else:
            return solution_text

    return solution_text[start_idx:end_idx]

def _get_full_problem(contest: str, year: int, prob_num: int) -> dict | None:
    """
    Fetch a single problem with full solution_text via get_by_contest_year.
    This is necessary because chroma_query results don't include solution_text.
    """
    rows = get_by_contest_year(contest, year, n=30)
    for r in rows:
        if r.get("problem_number") == prob_num:
            return r
    return None


@lru_cache(maxsize=1)
def _local_contest_index() -> dict[tuple[str, int], dict[str, str]]:
    """Build a lightweight index of local contest and solution PDFs."""
    pdf_root = Path(__file__).resolve().parent.parent / "contests"
    files = discover_contest_files(pdf_root)
    index: dict[tuple[str, int], dict[str, str]] = {}

    for contest_name, contest_file, solution_file in pair_contest_files(files):
        ref = contest_file or solution_file
        if ref is None:
            continue
        index[(contest_name, ref.year)] = {
            "contest": contest_name,
            "year": str(ref.year),
            "pdf_path": str(contest_file.path.resolve()) if contest_file else "",
            "solution_pdf_path": str(solution_file.path.resolve()) if solution_file else "",
        }

    return index


def _resolve_local_problem(contest: str, year: int, prob_num: int) -> dict | None:
    """Resolve an exact contest/year/problem request from local PDFs."""
    index = _local_contest_index()
    keys = [(contest, year)]
    if contest == "Gauss":
        keys.extend([("Gauss7", year), ("Gauss8", year)])

    for key in keys:
        meta = index.get(key)
        if not meta:
            continue
        pdf_path = meta.get("pdf_path", "")
        if not pdf_path:
            continue
        return {
            "contest": meta["contest"],
            "year": int(meta["year"]),
            "problem_number": prob_num,
            "pdf_path": pdf_path,
            "solution_pdf_path": meta.get("solution_pdf_path", ""),
            "has_solution": bool(meta.get("solution_pdf_path")),
            "page_number": prob_num,
            "solution_page_number": prob_num,
            "topics": [],
            "grades": [],
            "source_file": "",
            "part": None,
            "document": "",
        }

    return None


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ContestResult:
    reply: str
    problems: list[dict] | None = None
    intent: str = "general"
    active_agent: str | None = None
    problem_set_url: str | None = None
    problem_set_label: str | None = None
    solutions_url: str | None = None


# ── Regex helpers ─────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_PROB_NUM_RE = re.compile(r"\b(?:question|problem|q|#)\s*(\d{1,2})\b", re.I)
_CONTEST_RE = re.compile(
    r"\b(euclid|fryer|galois|hypatia|gauss\s*[78]?|pascal|cayley|fermat|cimc|csmc)\b",
    re.I,
)
_TOPIC_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in TOPIC_KEYWORDS) + r")\b", re.I
)
_GRADE_RE = re.compile(r"\bgrade\s*(\d{1,2})\b", re.I)
_COUNT_RE = re.compile(r"\b(\d{1,2})\b", re.I)

_EXPLAIN_WORDS = {
    "explain", "walk through", "work through", "show me how",
    "step by step", "why does", "how does", "break down",
}

# Words that explicitly mean "fetch this problem" — never trigger explain_solution
_FETCH_WORDS = {
    "show", "get", "give", "fetch", "retrieve", "display",
    "find", "look up", "pull up", "pick", "select", "load",
}
_BROWSE_WORDS = {
    "problem", "question", "something", "one", "any", "random", "example",
    "challenge", "exercise", "another", "different",
}
_CONCEPT_WORDS = {"what is", "what are", "define", "definition", "concept",
                  "how does", "tell me about"}
_CONTEST_INFO_WORDS = {
    "what is", "what are", "tell me about", "explain", "describe",
    "how does", "who takes", "when is", "how hard", "what level",
    "what topics", "what's on", "format", "overview",
    "how many problems", "how many questions", "how long", "duration",
    "time limit", "when", "month", "date", "schedule", "cutoff",
    "cutoffs", "honour roll", "honor roll",
}
_PRACTICE_WORDS = {"practice", "struggle", "weak", "bad at", "help with",
                   "drill", "exercises", "problems for", "questions on",
                   "study", "prepare"}
_SWITCH_WORDS = {"general agent", "switch to general", "back to general",
                 "return to general", "go to general", "main agent"}
_SMALL_TALK = {"hello", "hi", "hey", "thanks", "thank you", "ok", "okay",
               "what can you do", "help"}

_CONTEST_MAP = {
    "euclid": "Euclid", "fryer": "Fryer", "galois": "Galois",
    "hypatia": "Hypatia", "gauss7": "Gauss7", "gauss8": "Gauss8",
    "gauss": "Gauss", "pascal": "Pascal", "cayley": "Cayley",
    "fermat": "Fermat", "cimc": "CIMC", "csmc": "CSMC",
}

_CONTEST_FACTS: dict[str, dict[str, str]] = {
    "Euclid": {
        "problems": "typically 10 questions (with multi-part, proof-style components)",
        "duration": "2.5 hours (150 minutes)",
        "timing": "usually in April",
        "level": "senior high school (often grade 12)",
        "cutoffs": "no fixed cutoff; distinctions and honour roll thresholds change year to year",
    },
    "Pascal": {
        "problems": "25 multiple-choice style questions",
        "duration": "1 hour (60 minutes)",
        "timing": "usually in late February",
        "level": "grade 9",
        "cutoffs": "no fixed cutoff; certificates/distinction thresholds vary by year",
    },
    "Cayley": {
        "problems": "25 multiple-choice style questions",
        "duration": "1 hour (60 minutes)",
        "timing": "usually in late February",
        "level": "grade 10",
        "cutoffs": "no fixed cutoff; certificates/distinction thresholds vary by year",
    },
    "Fermat": {
        "problems": "25 multiple-choice style questions",
        "duration": "1 hour (60 minutes)",
        "timing": "usually in late February",
        "level": "grade 11",
        "cutoffs": "no fixed cutoff; certificates/distinction thresholds vary by year",
    },
    "Gauss7": {
        "problems": "25 questions",
        "duration": "1 hour (60 minutes)",
        "timing": "usually in May",
        "level": "grade 7",
        "cutoffs": "no fixed cutoff; school and contest recognition cutoffs vary annually",
    },
    "Gauss8": {
        "problems": "25 questions",
        "duration": "1 hour (60 minutes)",
        "timing": "usually in May",
        "level": "grade 8",
        "cutoffs": "no fixed cutoff; school and contest recognition cutoffs vary annually",
    },
    "Fryer": {
        "problems": "4 proof-style questions",
        "duration": "75 minutes",
        "timing": "usually in April",
        "level": "grade 9",
        "cutoffs": "no fixed cutoff; recognition thresholds vary by year",
    },
    "Galois": {
        "problems": "4 proof-style questions",
        "duration": "75 minutes",
        "timing": "usually in April",
        "level": "grade 10",
        "cutoffs": "no fixed cutoff; recognition thresholds vary by year",
    },
    "Hypatia": {
        "problems": "4 proof-style questions",
        "duration": "75 minutes",
        "timing": "usually in April",
        "level": "grade 11",
        "cutoffs": "no fixed cutoff; recognition thresholds vary by year",
    },
    "CIMC": {
        "problems": "intermediate-level multi-question contest (format can vary by year)",
        "duration": "typically around 2 to 2.5 hours",
        "timing": "usually in the fall",
        "level": "intermediate high school",
        "cutoffs": "no fixed cutoff; distinctions/honour roll vary each year",
    },
    "CSMC": {
        "problems": "senior-level multi-question contest (format can vary by year)",
        "duration": "typically around 2 to 2.5 hours",
        "timing": "usually in the fall",
        "level": "senior high school",
        "cutoffs": "no fixed cutoff; distinctions/honour roll vary each year",
    },
}

_CONTEST_CUTOFFS: dict[tuple[str, int], dict[str, str]] = {
    # Format: (contest_name, year) -> {"distinction": "X/Y", "certificate": "A/B", "note": "..."}
    # Example:
    # ("Euclid", 2024): {"distinction": "70/100", "certificate": "50/100", "note": "top 25% get distinction"},
    # ("Pascal", 2023): {"distinction": "160/200", "certificate": "120/200"},
    # Add verified cutoff data here as it becomes available.
}


def _contest_info_overview_text() -> str:
    return (
        "Waterloo math contests are a set of CEMC competitions for different grade levels, "
        "ranging from junior contests (Gauss) to senior contests (Euclid/CSMC), with both "
        "multiple-choice and proof-style formats depending on the contest. "
        "If you name a specific contest, I can give the usual format, duration, timing, and cutoff guidance."
    )


def _is_format_question(n: str) -> bool:
    return any(k in n for k in ("how many problems", "how many questions", "number of problems", "number of questions"))


def _is_duration_question(n: str) -> bool:
    return any(k in n for k in ("how long", "duration", "time limit", "minutes", "hours"))


def _is_timing_question(n: str) -> bool:
    return any(k in n for k in ("when", "month", "date", "schedule", "around when"))


def _is_cutoff_question(n: str) -> bool:
    return any(k in n for k in ("cutoff", "cutoffs", "honour roll", "honor roll", "distinction"))


def _extract_cutoff_year(text: str) -> int | None:
    """Extract year from cutoff-related questions (e.g., 'cutoffs for Euclid 2024', 'last year cutoffs for Pascal')."""
    # First try explicit year
    m = _YEAR_RE.search(text)
    if m:
        return int(m.group(1))
    
    # Then try relative year references (e.g., "last year", "2 years ago")
    year = _extract_year(text)
    return year


def _build_year_specific_cutoff_reply(contest: str, year: int) -> str | None:
    """Return year-specific cutoff info if available, or None if not in database."""
    cutoff_data = _CONTEST_CUTOFFS.get((contest, year))
    if not cutoff_data:
        return None
    
    lines = [f"**{contest} {year} — Cutoffs:**\n"]
    if "distinction" in cutoff_data:
        lines.append(f"- Distinction: {cutoff_data['distinction']}")
    if "certificate" in cutoff_data:
        lines.append(f"- Certificate: {cutoff_data['certificate']}")
    if "honour_roll" in cutoff_data or "honor_roll" in cutoff_data:
        key = "honour_roll" if "honour_roll" in cutoff_data else "honor_roll"
        lines.append(f"- Honour Roll: {cutoff_data[key]}")
    if "note" in cutoff_data:
        lines.append(f"\n{cutoff_data['note']}")
    
    return "\n".join(lines)


def _build_contest_fact_reply(contest: str, n: str) -> str:
    facts = _CONTEST_FACTS.get(contest)
    if not facts and contest == "Gauss":
        # User may ask for "Gauss" generically without specifying grade 7/8.
        facts = {
            "problems": "25 questions",
            "duration": "1 hour (60 minutes)",
            "timing": "usually in May",
            "level": "grades 7 and 8 versions (Gauss 7 and Gauss 8)",
            "cutoffs": "no fixed cutoff; recognition thresholds vary by year",
        }
    if not facts:
        return "I do not have a verified fact sheet for that contest yet. If you share your preferred wording/data, I can use it directly."

    ask_format = _is_format_question(n)
    ask_duration = _is_duration_question(n)
    ask_timing = _is_timing_question(n)
    ask_cutoff = _is_cutoff_question(n)
    asked_specific = ask_format or ask_duration or ask_timing or ask_cutoff

    lines = [f"Here is the usual info for {contest}:"]
    if ask_format or not asked_specific:
        lines.append(f"- Problems/questions: {facts['problems']}")
    if ask_duration or not asked_specific:
        lines.append(f"- Duration: {facts['duration']}")
    if ask_timing or not asked_specific:
        lines.append(f"- Typical timing: {facts['timing']}")
    if ask_cutoff or not asked_specific:
        lines.append(f"- Cutoffs: {facts['cutoffs']}")
    if not asked_specific:
        lines.append(f"- Intended level: {facts['level']}")

    lines.append("These details are typical patterns; Waterloo can adjust format/timing in a given year.")
    return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def is_negative_contest_request(text: str) -> bool:
    n = _norm(text)
    return bool(
        re.search(r"\b(don'?t|do not|dont|no|not)\b.*\b(contest|contests|contest help|contest problem|contest problems|practice contest|math contest)\b", n)
        or re.search(r"\b(i\s*do\s*not\s*want|i\s*don't\s*want|i\s*dont\s*want)\b.*\b(contest|contests|practice|problem|problems|help)\b", n)
        or re.search(r"\b(no|not)\b.*\b(contest help|contest problem|contest problems|contest practice)\b", n)
    )


def is_match_request(text: str) -> bool:
    n = _norm(text)
    return bool(re.search(r"\b(mentor|mentors|tutor|tutors|match|matching)\b", n))

def _get_current_year() -> int:
    from datetime import datetime
    return datetime.now().year

def _extract_year(text: str) -> int | None:
    # Check for explicit 4-digit year first
    m = _YEAR_RE.search(text)
    if m:
        return int(m.group(1))
    
    # Check for relative year references
    norm = _norm(text)
    current_year = _get_current_year()
    
    # Check for "N years ago" pattern FIRST (most flexible and general)
    match = re.search(r"(\d+)\s*years?\s+ago", norm)
    if match:
        years_back = int(match.group(1))
        return current_year - years_back
    
    # Check for specific phrases
    if "last year" in norm or "last year's" in norm:
        return current_year - 1
    if "this year" in norm or "this year's" in norm or "current year" in norm:
        return current_year
    if "year before last" in norm:
        return current_year - 2
    if "next year" in norm or "next year's" in norm:
        return current_year + 1
    
    return None

def _extract_contest(text: str) -> str | None:
    m = _CONTEST_RE.search(text)
    if not m:
        return None
    return _CONTEST_MAP.get(m.group(1).lower().replace(" ", ""))

def _extract_topic(text: str) -> str | None:
    m = _TOPIC_RE.search(text)
    return m.group(1).lower().replace(" ", "_") if m else None

def _extract_prob_num(text: str) -> int | None:
    m = _PROB_NUM_RE.search(text)
    return int(m.group(1)) if m else None

def _extract_grade(text: str) -> int | None:
    m = _GRADE_RE.search(text)
    return int(m.group(1)) if m else None

def _extract_count(text: str) -> int:
    """Extract number of requested problems from natural phrasing."""
    n = _norm(text)

    # Strip year-reference numbers so they don't get picked up as counts.
    # e.g. "2 years ago" → removed, "last year" → no digits, "2024" → 4-digit, safe
    n_stripped = re.sub(r"\b\d+\s*years?\s+ago\b", "", n)
    n_stripped = re.sub(r"\b(20|19)\d{2}\b", "", n_stripped)

    # Strong explicit patterns first (run on stripped text).
    explicit_patterns = [
        r"\b(\d{1,2})\s+(?:problems?|questions?|ones?)\b",
        r"\bshow\s+me\s+(\d{1,2})\b",
        r"\bgive\s+me\s+(\d{1,2})\b",
        r"\b(\d{1,2})\s+[a-z]+\s+(?:problems?|questions?)\b",  # e.g. "3 euclid problems"
    ]
    for pat in explicit_patterns:
        m = re.search(pat, n_stripped, re.I)
        if m:
            return max(1, min(25, int(m.group(1))))

    # Vague multi-item language.
    if any(w in n for w in ("a few", "few", "several", "couple", "multiple", "some")):
        return 3

    # Last resort: if a bare small number appears on stripped text and looks like a problem request.
    if any(w in n for w in ("problem", "question", "show", "give", "find", "get")):
        m_any = _COUNT_RE.search(n_stripped)
        if m_any:
            return max(1, min(25, int(m_any.group(1))))

    return 1


# ── Intent detection ──────────────────────────────────────────────────────────

def detect_intent(text: str, session: dict) -> str:
    n = _norm(text)

    if is_negative_contest_request(n):
        return "switch_general"

    if is_problem_set_request(n):
        return "problem_set"

    if any(w in n for w in _SWITCH_WORDS):
        return "switch_general"

    words = set(n.split())
    if len(words) <= 4 and words & _SMALL_TALK and not _CONTEST_RE.search(n):
        return "small_talk"

    if any(w in n for w in ("what contests", "which contests",
                             "available contests", "list contests")):
        return "list_contests"

    # Check for contest/math info questions BEFORE followup, so "what are waterloo math
    # contests" doesn't get swallowed by the followup trigger words ("what", "how", etc.)
    _has_concept_word = any(w in n for w in _CONTEST_INFO_WORDS)
    _has_contest_name = bool(_CONTEST_RE.search(n))
    _has_waterloo_ref = any(w in n for w in ("waterloo", "contest", "competition", "math contest"))
    if _has_concept_word and (_has_contest_name or _has_waterloo_ref):
        return "contest_info"

    # Similarly, pure math concept questions ("what is modular arithmetic") aren't followups
    _is_pure_concept = (
        any(n.startswith(w) for w in _CONCEPT_WORDS)
        or any(w in n for w in ("what is", "what are"))
    ) and not _CONTEST_RE.search(n)

    has_active = bool(session.get("active_problem"))
    has_contest_signal = bool(_CONTEST_RE.search(n) or _YEAR_RE.search(n))
    if has_active and not has_contest_signal and _PROB_NUM_RE.search(n) is None and not _is_pure_concept:
        if any(w in n for w in _EXPLAIN_WORDS | {"another", "alternative",
                                                   "different", "why", "how",
                                                   "what", "can you", "more"}):
            return "followup"

    # Explicit fetch requests: "show me a problem from euclid", "give me any problem"
    has_fetch = any(w in n for w in _FETCH_WORDS)
    has_prob_num = bool(_PROB_NUM_RE.search(n))
    has_contest = bool(_CONTEST_RE.search(n))
    if has_fetch and not has_prob_num and not any(w in n for w in _EXPLAIN_WORDS):
        if has_contest:
            return "specific_problem"
        if any(w in n for w in _BROWSE_WORDS):
            return "browse"

    # "give me another problem" / "show me something different" with no contest → browse
    if any(w in n for w in ("another problem", "different problem", "new problem",
                             "random problem", "any problem")):
        return "browse"

    if _PROB_NUM_RE.search(n) and _CONTEST_RE.search(n):
        # If the user is explicitly fetching (show/get/find), never explain
        if any(w in n for w in _FETCH_WORDS):
            return "specific_problem"
        # Only explain if user explicitly used explanation language
        if any(w in n for w in _EXPLAIN_WORDS):
            return "explain_solution"
        return "specific_problem"

    # "explain the solution to Euclid 2022" (no problem number) — still explain
    if any(w in n for w in _EXPLAIN_WORDS) and _CONTEST_RE.search(n):
        return "explain_solution"

    # "solution to X" alone (no explain word) — just fetch the problem
    if "solution" in n and _CONTEST_RE.search(n):
        return "specific_problem"

    if any(w in n for w in _PRACTICE_WORDS):
        return "practice"

    if any(n.startswith(w) for w in _CONCEPT_WORDS) or \
            any(w in n for w in ("what is", "what are")):
        return "concept"

    return "search"


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_problem(result: dict, index: int | None = None) -> str:
    prefix = f"**{index}.** " if index is not None else ""
    prob_label = f"Problem {result['problem_number']}" if result["problem_number"] else "Problem"
    lines = [f"{prefix}**{result['contest']} {result['year']} — {prob_label}**"]
    if result.get("topics"):
        lines.append(f"Topics: {', '.join(result['topics'])}")
    if result.get("has_solution"):
        lines.append("_Solution available — click Show solution below the image._")
    else:
        lines.append("_No solution available for this problem._")
    return "\n".join(lines)


def _unavailable_reply() -> ContestResult:
    return ContestResult(
        reply=(
            "The contest problem database hasn't been indexed yet.\n\n"
            "Run the ingestion script first:\n"
            "  python -m scripts.ingest_contests --pdf-root /path/to/your/contests\n\n"
            "Once indexed, I can answer questions, retrieve problems, and explain solutions."
        ),
        intent="not_indexed",
    )


# ── Intent handlers ───────────────────────────────────────────────────────────

def handle_small_talk(text: str, session: dict) -> ContestResult:
    history = _build_history(session)
    reply = _grok(history + [{"role": "user", "content": text}], max_tokens=200)
    return ContestResult(reply=reply, intent="small_talk", active_agent="contest")


def handle_list_contests() -> ContestResult:
    available = list_available_contests()
    if not available:
        return _unavailable_reply()
    lines = ["Here are the indexed Waterloo contests:\n"]
    for item in available:
        years = item["years"]
        years_str = ", ".join(years[:5])
        if len(years) > 5:
            years_str += f" … (+{len(years) - 5} more)"
        lines.append(f"- **{item['contest']}**: {years_str}")
    lines.append("\nAsk me to retrieve problems, explain solutions, or find practice questions!")
    return ContestResult(reply="\n".join(lines), intent="list_contests")


def handle_specific_problem(text: str, session: dict) -> ContestResult:
    contest = _extract_contest(text)
    year = _extract_year(text)
    prob_num = _extract_prob_num(text)
    topic = _extract_topic(text)
    requested_count = _extract_count(text)

    if not (contest or year or topic):
        return ContestResult(
            reply="Please specify a contest name (e.g. Euclid, Fermat) and/or year.",
            intent="specific_problem",
        )

    if contest and year and prob_num:
        result = _get_full_problem(contest, year, prob_num)
        if result:
            session["active_problem"] = result
            return ContestResult(
                reply=_fmt_problem(result),
                problems=[result],
                intent="specific_problem",
            )
        local = _resolve_local_problem(contest, year, prob_num)
        if local:
            session["active_problem"] = local
            return ContestResult(
                reply=_fmt_problem(local),
                problems=[local],
                intent="specific_problem",
            )
        return ContestResult(
            reply=f"Could not find **{contest} {year} Problem {prob_num}** in the database. "
                  f"Make sure the PDFs have been ingested.",
            intent="specific_problem",
        )

    # Contest-only request (no year, no prob_num): pick multiple random problems from that contest
    if contest and not year and not prob_num:
        av = list_available_contests()
        years_for_contest: list[int] = []
        for item in av:
            if item["contest"] == contest:
                for y in item.get("years", []):
                    try:
                        years_for_contest.append(int(y))
                    except Exception:
                        continue
        if years_for_contest:
            selected: list[dict] = []
            seen: set[tuple[str, int, int]] = set()
            attempts = 0
            max_attempts = requested_count * 5
            while len(selected) < requested_count and attempts < max_attempts:
                chosen_year = random.choice(years_for_contest)
                rows = get_by_contest_year(contest, chosen_year, n=30)
                valid = [r for r in rows if r.get("pdf_path") and r.get("problem_number")]
                if valid:
                    for _ in range(len(valid)):
                        if len(selected) >= requested_count:
                            break
                        candidate = random.choice(valid)
                        key = (contest, chosen_year, candidate["problem_number"])
                        if key not in seen:
                            seen.add(key)
                            full = _get_full_problem(contest, chosen_year, candidate["problem_number"])
                            selected.append(full or candidate)
                attempts += 1
            
            if selected:
                session["active_problem"] = selected[0]
                label = "a problem" if len(selected) == 1 else f"{len(selected)} problems"
                lines = [f"Here's {label} from {contest}:\n"]
                for i, r in enumerate(selected, 1):
                    lines.append(_fmt_problem(r, index=i))
                    lines.append("")
                return ContestResult(
                    reply="\n".join(lines).strip(),
                    problems=selected,
                    intent="specific_problem",
                )

    # Year-only or topic-only: fall through to semantic search
    results = chroma_query(
        text=text,
        n_results=max(8, requested_count),
        contest=contest,
        year=year,
        topic=topic,
    )
    if not results:
        desc = " ".join(filter(None, [str(year) if year else None, contest]))
        return ContestResult(
            reply=f"No problems found for {desc or 'that query'}.",
            intent="specific_problem",
        )
    if prob_num:
        exact = [r for r in results if r["problem_number"] == prob_num]
        if exact:
            results = exact

    # Enrich top result with full solution_text
    top = results[0]
    if top.get("contest") and top.get("year") and top.get("problem_number"):
        full = _get_full_problem(top["contest"], top["year"], top["problem_number"])
        if full:
            results[0] = full

    session["active_problem"] = results[0]
    shown = results[:max(1, requested_count)]
    lines = [f"Here {'is' if len(shown) == 1 else 'are'} the matching problem(s):\n"]
    for i, r in enumerate(shown, 1):
        lines.append(_fmt_problem(r, index=i))
        lines.append("")
    return ContestResult(
        reply="\n".join(lines).strip(),
        problems=shown,
        intent="specific_problem",
    )


def handle_explain_solution(text: str, session: dict) -> ContestResult:
    """
    Fetch full problem + solution text via get_by_contest_year (not chroma_query,
    which doesn't return solution_text), then ask the configured LLM to explain step by step.
    """
    contest = _extract_contest(text)
    year = _extract_year(text)
    prob_num = _extract_prob_num(text)

    # Try exact fetch first
    result = None
    if contest and year and prob_num:
        result = _get_full_problem(contest, year, prob_num)

    # Fallback to semantic search, then re-fetch full record
    if result is None:
        results = chroma_query(text=text, n_results=5, contest=contest, year=year)
        if not results:
            return ContestResult(
                reply="I couldn't find that problem. Try specifying the contest name and year.",
                intent="explain_solution",
            )
        candidates = [r for r in results if r.get("has_solution")] or results
        if prob_num:
            exact = [r for r in candidates if r["problem_number"] == prob_num]
            if exact:
                candidates = exact
        top = candidates[0]
        # Re-fetch to get solution_text
        if top.get("contest") and top.get("year") and top.get("problem_number"):
            result = _get_full_problem(top["contest"], top["year"], top["problem_number"]) or top
        else:
            result = top

    session["active_problem"] = result
    prob_label = f"Problem {result['problem_number']}"
    header = f"**{result['contest']} {result['year']} — {prob_label}**\n\n"

    problem_text = result.get("document", "") or result.get("problem_text", "")
    solution_text = result.get("solution_text", "") or ""
    print(f"[DEBUG-TRIM] About to trim: problem_number={result.get('problem_number')!r} "
          f"(type={type(result.get('problem_number'))}), solution_text length={len(solution_text)}")
    solution_text = _trim_solution_to_problem(solution_text, result.get("problem_number"))

    has_solution_text = bool(solution_text and solution_text.strip()
                             and solution_text.strip() != problem_text.strip())

    if has_solution_text:
        prompt = (
            f"Here is a Waterloo Math Contest problem with its official solution.\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
            f"OFFICIAL SOLUTION:\n{solution_text}\n\n"
            f"Explain this solution clearly. Start with a 'Key insight:' line, "
            f"then walk through each step numbered. Each step on its own line. "
            f"End with 'Answer: ...' on its own line."
        )
    elif result.get("has_solution"):
        prompt = (
            f"Here is a Waterloo Math Contest problem:\n\n"
            f"PROBLEM:\n{problem_text}\n\n"
            f"Solve this step by step. Start with a 'Key insight:' line, "
            f"then number each step. Each step on its own line. "
            f"End with 'Answer: ...' on its own line."
        )
    else:
        return ContestResult(
            reply=header + "No solution is available for this problem in the database.",
            problems=[result],
            intent="explain_solution",
        )

    history = _build_history(session)
    explanation = _grok_or_local(
        history + [{"role": "user", "content": prompt}],
        problem_text=problem_text,
        solution_text=solution_text,
        max_tokens=2500,
    )
    return ContestResult(
        reply=header + explanation,
        problems=[result],
        intent="explain_solution",
    )


def handle_followup(text: str, session: dict) -> ContestResult:
    """Answer a follow-up question in the context of the active problem."""
    active = session.get("active_problem", {})
    prob_text = active.get("document", "") or active.get("problem_text", "")
    sol_text = active.get("solution_text", "") or ""
    sol_text = _trim_solution_to_problem(sol_text, active.get("problem_number"))
    prob_label = (
        f"{active.get('contest', '')} {active.get('year', '')} "
        f"Problem {active.get('problem_number', '')}"
    ).strip()

    context_parts = [f"The student is asking about: {prob_label}",
                     f"\nPROBLEM:\n{prob_text}"]
    if sol_text and sol_text.strip() != prob_text.strip():
        context_parts.append(f"\nOFFICIAL SOLUTION:\n{sol_text}")
    context = "\n".join(context_parts)

    # Inject context as a system-level note before the conversation history
    messages: list[dict] = [
        {"role": "user", "content": f"[Context for this conversation]\n{context}"},
        {"role": "assistant", "content": "Understood, I have the problem and solution context."},
    ]
    messages += _build_history(session)
    messages.append({"role": "user", "content": text})

    reply = _grok_or_local(messages, problem_text=prob_text, solution_text=sol_text, max_tokens=2500)
    return ContestResult(
        reply=reply,
        problems=[active] if active else None,
        intent="followup",
        active_agent="contest",
    )


def handle_practice(text: str, session: dict) -> ContestResult:
    topic = _extract_topic(text)
    grade = _extract_grade(text)
    contest = _extract_contest(text)

    query_text = text
    if topic:
        kws = TOPIC_KEYWORDS.get(topic, [])
        query_text = f"{text} {' '.join(kws[:5])}"

    results = chroma_query(
        text=query_text, n_results=8,
        contest=contest, grade=grade, topic=topic,
    )

    if not results:
        desc = " ".join(filter(None, [topic, f"grade {grade}" if grade else None]))
        return ContestResult(
            reply=f"No practice problems found for {desc or 'that topic'}. "
                  f"Try a different topic or make sure the PDFs are ingested.",
            intent="practice",
        )

    seen: set[tuple] = set()
    deduped = []
    for r in results:
        key = (r["contest"], r["year"], r["problem_number"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    topic_label = topic.replace("_", " ") if topic else "the requested area"
    grade_label = f" for grade {grade}" if grade else ""

    intro_prompt = (
        f"A student wants to practice {topic_label}{grade_label} using Waterloo contest problems. "
        f"Write 1-2 encouraging sentences introducing the practice set. "
        f"Do not list the problems."
    )
    intro = _grok([{"role": "user", "content": intro_prompt}], max_tokens=100)

    lines = [intro, ""]
    for i, r in enumerate(deduped[:5], 1):
        lines.append(_fmt_problem(r, index=i))
        lines.append("")
    lines.append('_Ask me to "explain the solution" for any of these._')

    # Enrich first result with solution_text for follow-ups
    top = deduped[0]
    if top.get("contest") and top.get("year") and top.get("problem_number"):
        full = _get_full_problem(top["contest"], top["year"], top["problem_number"])
        if full:
            deduped[0] = full
    session["active_problem"] = deduped[0]

    return ContestResult(
        reply="\n".join(lines).strip(),
        problems=deduped[:5],
        intent="practice",
    )


def handle_contest_info(text: str, session: dict) -> ContestResult:
    """Answer general questions about contests (what is Euclid, format/timing/cutoff, etc.)."""
    n = _norm(text)
    contest = _extract_contest(text)

    # Check for year-specific cutoff queries first (highest priority for grounded data)
    if contest and _is_cutoff_question(n):
        year = _extract_cutoff_year(text)
        if year:
            cutoff_reply = _build_year_specific_cutoff_reply(contest, year)
            if cutoff_reply:
                return ContestResult(
                    reply=cutoff_reply,
                    intent="contest_info",
                    active_agent="contest",
                )
            # No data for this year; offer to help look it up
            return ContestResult(
                reply=f"I don't have verified cutoff data for {contest} {year} stored yet. "
                      f"If you can share the official cutoffs, I can add them. "
                      f"Alternatively, check the official CEMC website for {year} contest results.",
                intent="contest_info",
                active_agent="contest",
            )

    # Prefer deterministic fact-sheet answers for other operational contest questions.
    if contest and (_is_format_question(n) or _is_duration_question(n) or _is_timing_question(n) or _is_cutoff_question(n)):
        return ContestResult(
            reply=_build_contest_fact_reply(contest, n),
            intent="contest_info",
            active_agent="contest",
        )

    # If a specific contest is named, provide a grounded profile reply first.
    if contest:
        fact_reply = _build_contest_fact_reply(contest, n)
        if not fact_reply.lower().startswith("i do not have a verified fact sheet"):
            return ContestResult(reply=fact_reply, intent="contest_info", active_agent="contest")

    # Generic contest-info question (e.g. "what are waterloo math contests").
    history = _build_history(session)
    grounded_context = (
        "Answer this as a Waterloo contest information question. "
        + _contest_info_overview_text()
    )
    reply = _grok(
        history + [
            {"role": "assistant", "content": grounded_context},
            {"role": "user", "content": text},
        ],
        max_tokens=700,
    )
    return ContestResult(reply=reply, intent="contest_info", active_agent="contest")


def handle_concept(text: str, session: dict) -> ContestResult:
    topic = _extract_topic(text)
    results = chroma_query(text=text, n_results=3, topic=topic)

    history = _build_history(session)
    explanation = _grok(
        history + [{"role": "user", "content": text}],
        max_tokens=700,
    )

    if results:
        explanation += "\n\n**Example contest problems:**\n"
        for i, r in enumerate(results[:2], 1):
            explanation += "\n" + _fmt_problem(r, index=i) + "\n"
        session["active_problem"] = results[0]

    return ContestResult(
        reply=explanation,
        problems=results[:2] if results else None,
        intent="concept",
    )


def handle_search(text: str, session: dict) -> ContestResult:
    """
    For messages that don't match a specific intent:
    - If RAG finds relevant problems, show the cards only (no explanation)
    - Otherwise, answer conversationally with Gemini
    """
    contest = _extract_contest(text)
    year = _extract_year(text)
    topic = _extract_topic(text)
    grade = _extract_grade(text)

    # If the message mentions a contest/year/problem, route to specific_problem
    # instead of doing a search — avoids the bot explaining things unprompted
    if contest or year or _extract_prob_num(text):
        return handle_specific_problem(text, session)

    results = chroma_query(
        text=text, n_results=5,
        contest=contest, year=year, topic=topic, grade=grade,
    )

    if results:
        session["active_problem"] = results[0]
        lines = ["Here are the most relevant problems I found:\n"]
        for i, r in enumerate(results[:4], 1):
            lines.append(_fmt_problem(r, index=i))
            lines.append("")
        return ContestResult(
            reply="\n".join(lines).strip(),
            problems=results[:4],
            intent="search",
        )

    # No RAG results — answer conversationally
    history = _build_history(session)
    reply = _grok(history + [{"role": "user", "content": text}], max_tokens=800)
    return ContestResult(reply=reply, intent="search")


def handle_browse(text: str, session: dict) -> ContestResult:
    """Pick and return a random problem from any indexed contest."""
    available = list_available_contests()
    if not available:
        return _unavailable_reply()

    # Prefer contests that appear in the user's message, otherwise pick at random
    contest_obj = None
    contest = _extract_contest(text)
    if contest:
        for item in available:
            if item["contest"] == contest:
                contest_obj = item
                break
    if contest_obj is None:
        contest_obj = random.choice(available)

    years: list[int] = []
    for y in contest_obj.get("years", []):
        try:
            years.append(int(y))
        except Exception:
            continue

    if not years:
        return handle_search(text, session)

    chosen_year = random.choice(years)
    rows = get_by_contest_year(contest_obj["contest"], chosen_year, n=30)
    valid = [r for r in rows if r.get("pdf_path") and r.get("problem_number")]
    if not valid:
        valid = rows
    if not valid:
        return handle_search(text, session)

    result = random.choice(valid)
    full = _get_full_problem(contest_obj["contest"], chosen_year, result["problem_number"])
    result = full or result
    session["active_problem"] = result
    return ContestResult(
        reply="Here's a problem for you:\n\n" + _fmt_problem(result),
        problems=[result],
        intent="browse",
    )


def handle_problem_set(text: str, session: dict) -> ContestResult:
    result = build_problem_set_from_text(text)
    if not result.ok:
        return ContestResult(
            reply=result.reply,
            intent="problem_set",
            active_agent="contest",
        )

    if result.problems:
        session["active_problem"] = result.problems[0]

    return ContestResult(
        reply=result.reply,
        problems=[],
        intent="problem_set",
        active_agent="contest",
        problem_set_url=result.pdf_url,
        problem_set_label=result.label,
        solutions_url=getattr(result, "solutions_url", None),
    )


# ── Main agent ────────────────────────────────────────────────────────────────

class ContestAgent:

    def run(self, message: str, session: dict[str, Any] | None = None) -> ContestResult:
        if session is None:
            session = {}

        rewrite = _rewrite_user_message(message, session, forced_agent="contest")
        session["last_contest_prompt"] = rewrite.formatted_prompt
        message = rewrite.cleaned_message
        prompt_message = rewrite.formatted_prompt

        n = _norm(message)
        if is_match_request(n):
            session["active_agent"] = "match"
            session["state"] = "awaiting_match_details"
            session["pending_match_step"] = "grade"
            return ContestResult(
                reply="Switched to the Match agent. What grade is the mentee in?",
                intent="switch_match",
                active_agent="match",
            )

        if is_negative_contest_request(n):
            session["active_agent"] = "general"
            session["state"] = "idle"
            return ContestResult(
                reply="Switched back to the General agent. Tell me what you need instead.",
                intent="switch_general",
                active_agent="general",
            )

        if any(w in n for w in _SWITCH_WORDS):
            return ContestResult(
                reply="Switched back to the General agent. Ask me a mentor or booking question next.",
                intent="switch_general",
                active_agent="general",
            )

        intent = detect_intent(message, session)

        if intent == "small_talk":
            return handle_small_talk(message, session)
        if intent == "contest_info":
            return handle_contest_info(message, session)
        if intent == "concept":
            return handle_concept(message, session)

        if collection_count() == 0:
            if intent in {"problem_set", "list_contests", "practice", "search", "followup"}:
                return _unavailable_reply()
            # Still answer general/info questions even without DB
            history = _build_history(session)
            reply = _grok(history + [{"role": "user", "content": prompt_message}], max_tokens=800)
            return ContestResult(reply=reply, intent=intent)

        if intent == "list_contests":
            return handle_list_contests()
        elif intent == "specific_problem":
            return handle_specific_problem(message, session)
        elif intent == "explain_solution":
            return handle_explain_solution(message, session)
        elif intent == "followup":
            return handle_followup(message, session)
        elif intent == "practice":
            return handle_practice(message, session)
        elif intent == "problem_set":
            return handle_problem_set(message, session)
        elif intent == "browse":
            return handle_browse(message, session)
        else:
            return handle_search(message, session)


contest_agent = ContestAgent()