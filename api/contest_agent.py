"""
api/contest_agent.py

LangChain-style agent for Waterloo Math Contest knowledge.

v5 — fixes:
  - solution_text now always fetched via get_by_contest_year (not from query results
    which don't include it)
  - handle_search falls back to Groq for general questions instead of just RAG
  - system prompt overhauled: Groq responds naturally like an AI assistant,
    trusts problem/solution text it's given, never says a problem is impossible
  - conversation history passed to every Groq call for better follow-up handling
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import certifi_win32  # noqa: F401
except Exception:
    certifi_win32 = None

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

# ── Groq client ───────────────────────────────────────────────────────────────

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

_groq_init_error = ""

try:
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if OpenAI is None:
        _client = None
        _groq_init_error = "openai package is not installed"
    elif not groq_api_key:
        _client = None
        _groq_init_error = "GROQ_API_KEY is not set"
    else:
        _client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
except Exception as e:
    _client = None
    _groq_init_error = str(e)

_MODEL = "llama-3.3-70b-versatile"

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

FORMATTING — always follow these rules:
- Structure explanations with clear sections. Use blank lines between sections.
- Number every step: "1.", "2.", "3." etc.
- Put each step on its own line with a blank line after it.
- Use plain Unicode math (×, ÷, √, ², ³, ≤, ≥, ≠, π, θ) — absolutely no LaTeX, \
  no dollar signs, no backslashes.
- For the key insight, write it on its own line starting with "Key insight:"
- Keep each step focused on one idea. Do not write walls of text.
- End with the final answer clearly stated on its own line: "Answer: ..."

For concept explanations:
- Definition first, then techniques, then a contest tip.
- Use bullet points for lists of techniques or formulas.

For general conversation:
- Be concise and natural. No need for numbered steps unless explaining math.
- Warm and encouraging tone suitable for a high school student."""


def _grok(messages: list[dict], max_tokens: int = 1200) -> str:
    """Call the Groq API and return the assistant text."""
    if _client is None:
        reason = _groq_init_error or "Groq client is not configured"
        return f"_(AI explanation unavailable: {reason}.)_"
    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": _SYSTEM_MATH}] + messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"_(AI explanation unavailable: {e})_"


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
        if len(tokens) >= 5 and short_ratio >= 0.6 and max(len(t) for t in tokens) <= 3:
            return True
        if len(tokens) >= 6 and (sum(len(t) for t in tokens) / len(tokens)) < 2.0 and max(len(t) for t in tokens) <= 3:
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

        if not is_header_or_footer(line) and not is_noisy_fragment(line):
            cleaned_lines.append(line)
        i += 1

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
    """Use Groq when available; otherwise fall back to a grounded local explanation."""
    reply = _grok(messages, max_tokens=max_tokens)
    if reply.startswith("_(AI explanation unavailable:"):
        return _local_solution_explanation(problem_text, solution_text)
    return reply


def _build_history(session: dict, n: int = 6) -> list[dict]:
    """Return the last n turns of conversation history for Groq."""
    history = []
    for turn in session.get("messages", [])[-n:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content and role in ("user", "assistant"):
            history.append({"role": role, "content": content})
    return history


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

_EXPLAIN_WORDS = {
    "explain", "walk through", "work through", "show me how",
    "step by step", "why does", "how does", "break down",
}

# Words that explicitly mean "fetch this problem" — never trigger explain_solution
_FETCH_WORDS = {
    "show", "get", "give", "fetch", "retrieve", "display",
    "find", "look up", "pull up",
}
_CONCEPT_WORDS = {"what is", "what are", "define", "definition", "concept",
                  "how does", "tell me about"}
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


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())

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
    
    if "last year" in norm or "last year's" in norm:
        return current_year - 1
    if "this year" in norm or "this year's" in norm or "current year" in norm:
        return current_year
    if "year before last" in norm or "2 years ago" in norm:
        return current_year - 2
    
    # Check for "N years ago" pattern
    match = re.search(r"(\d+)\s*years?\s+ago", norm)
    if match:
        years_back = int(match.group(1))
        return current_year - years_back
    
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


# ── Intent detection ──────────────────────────────────────────────────────────

def detect_intent(text: str, session: dict) -> str:
    n = _norm(text)

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

    has_active = bool(session.get("active_problem"))
    has_contest_signal = bool(_CONTEST_RE.search(n) or _YEAR_RE.search(n))
    if has_active and not has_contest_signal and _PROB_NUM_RE.search(n) is None:
        if any(w in n for w in _EXPLAIN_WORDS | {"another", "alternative",
                                                   "different", "why", "how",
                                                   "what", "can you", "more"}):
            return "followup"

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

    results = chroma_query(text=text, n_results=8, contest=contest, year=year, topic=topic)
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
    lines = [f"Here {'is' if len(results) == 1 else 'are'} the matching problem(s):\n"]
    for i, r in enumerate(results[:3], 1):
        lines.append(_fmt_problem(r, index=i))
        lines.append("")
    return ContestResult(
        reply="\n".join(lines).strip(),
        problems=results[:3],
        intent="specific_problem",
    )


def handle_explain_solution(text: str, session: dict) -> ContestResult:
    """
    Fetch full problem + solution text via get_by_contest_year (not chroma_query,
    which doesn't return solution_text), then ask Groq to explain step by step.
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
        max_tokens=1500,
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

    reply = _grok_or_local(messages, problem_text=prob_text, solution_text=sol_text, max_tokens=1200)
    return ContestResult(reply=reply, intent="followup", active_agent="contest")


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
    - Otherwise, answer conversationally with Groq
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
        problems=result.problems,
        intent="problem_set",
        active_agent="contest",
        problem_set_url=result.pdf_url,
        problem_set_label=result.label,
    )


# ── Main agent ────────────────────────────────────────────────────────────────

class ContestAgent:

    def run(self, message: str, session: dict[str, Any] | None = None) -> ContestResult:
        if session is None:
            session = {}

        n = _norm(message)
        if any(w in n for w in _SWITCH_WORDS):
            return ContestResult(
                reply="Switched back to the General agent. Ask me a mentor or booking question next.",
                intent="switch_general",
                active_agent="general",
            )

        intent = detect_intent(message, session)

        if intent == "small_talk":
            return handle_small_talk(message, session)
        if intent == "concept":
            return handle_concept(message, session)

        if collection_count() == 0:
            if intent in {"problem_set", "list_contests", "practice", "search", "followup"}:
                return _unavailable_reply()
            # Still answer general questions even without DB
            history = _build_history(session)
            reply = _grok(history + [{"role": "user", "content": message}], max_tokens=800)
            return ContestResult(reply=reply, intent="concept")

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
        else:
            return handle_search(message, session)


contest_agent = ContestAgent()