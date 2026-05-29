"""
api/contest_agent.py

LangChain-style agent for Waterloo Math Contest knowledge.
Handles:
  1. Concept questions ("what is a geometric sequence?") — answered with
     relevant contest problems as examples via RAG
  2. Specific problem retrieval ("show me a 2022 Euclid combinatorics problem")
  3. Step-by-step solution explanations ("explain the solution to Euclid 2022 Q5")
  4. Weak-area matching ("I struggle with number theory, give me practice problems")

Integrates with the existing MentorTaskAgents via a new "contest" intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag.contest_retriever import (
    collection_count,
    get_by_contest_year,
    list_available_contests,
    query as chroma_query,
)
from rag.contest_ingestor import TOPIC_KEYWORDS, CONTEST_GRADES


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ContestResult:
    reply: str
    problems: list[dict] | None = None
    intent: str = "general"


# ── Intent detection ──────────────────────────────────────────────────────────

_CONTEST_NAMES = {
    "euclid", "fryer", "galois", "hypatia", "gauss", "pascal", "cayley",
    "fermat", "cimc", "csmc", "gauss7", "gauss8",
}

_EXPLAIN_WORDS = {
    "explain", "walk", "step", "how", "why", "understand", "solution",
    "solve", "work through", "show me how",
}

_CONCEPT_WORDS = {
    "what is", "what are", "define", "definition", "concept", "theory",
    "explain", "how does", "tell me about",
}

_PRACTICE_WORDS = {
    "practice", "struggle", "weak", "bad at", "help with", "drill",
    "exercises", "problems for", "questions on", "study", "prepare",
}

_LIST_WORDS = {
    "list", "show", "give me", "find", "retrieve", "get", "display",
    "available", "what contests",
}

_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_PROB_NUM_RE = re.compile(r"\b(?:question|problem|q|#)\s*(\d{1,2})\b", re.I)
_CONTEST_RE = re.compile(
    r"\b(euclid|fryer|galois|hypatia|gauss\s*[78]?|pascal|cayley|fermat|cimc|csmc)\b",
    re.I,
)
_TOPIC_RE = re.compile(
    r"\b(" + "|".join(k for k in TOPIC_KEYWORDS) + r")\b", re.I
)
_GRADE_RE = re.compile(r"\bgrade\s*(\d{1,2})\b|\b(grade)\s+(\d{1,2})\b", re.I)


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _extract_year(text: str) -> int | None:
    m = _YEAR_RE.search(text)
    return int(m.group(1)) if m else None


def _extract_contest(text: str) -> str | None:
    m = _CONTEST_RE.search(text)
    if not m:
        return None
    raw = m.group(1).lower().replace(" ", "")
    # Normalise to title-case canonical names
    _MAP = {
        "euclid": "Euclid", "fryer": "Fryer", "galois": "Galois",
        "hypatia": "Hypatia", "gauss7": "Gauss7", "gauss8": "Gauss8",
        "gauss": "Gauss", "pascal": "Pascal", "cayley": "Cayley",
        "fermat": "Fermat", "cimc": "CIMC", "csmc": "CSMC",
    }
    return _MAP.get(raw)


def _extract_topic(text: str) -> str | None:
    m = _TOPIC_RE.search(text)
    return m.group(1).lower().replace(" ", "_") if m else None


def _extract_problem_number(text: str) -> int | None:
    m = _PROB_NUM_RE.search(text)
    return int(m.group(1)) if m else None


def _extract_grade(text: str) -> int | None:
    m = _GRADE_RE.search(text)
    if m:
        return int(m.group(1) or m.group(3))
    return None


def detect_intent(text: str) -> str:
    """
    Returns one of:
      "list_contests"   — user wants to know what contests are available
      "specific_problem"— user wants a specific contest+year+problem
      "explain_solution"— user wants a step-by-step explanation
      "practice"        — user wants problems to practice a topic/weakness
      "concept"         — user asks what a concept is (answered with examples)
      "search"          — generic semantic search fallback
    """
    n = _norm(text)
    tokens = set(n.split())

    if any(w in n for w in ("what contests", "which contests", "available contests", "list contests")):
        return "list_contests"
    if _PROB_NUM_RE.search(n) and _CONTEST_RE.search(n):
        if any(w in n for w in _EXPLAIN_WORDS):
            return "explain_solution"
        return "specific_problem"
    if any(w in n for w in _EXPLAIN_WORDS) and _CONTEST_RE.search(n):
        return "explain_solution"
    if any(w in n for w in _PRACTICE_WORDS):
        return "practice"
    if any(n.startswith(w) for w in _CONCEPT_WORDS) or any(w in n for w in ("what is", "what are")):
        return "concept"
    return "search"


# ── Intent handlers ───────────────────────────────────────────────────────────

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


def _format_problem(result: dict, index: int | None = None) -> str:
    """Format a problem reference for the chat reply text (not the image)."""
    prefix = f"**{index}.** " if index is not None else ""
    prob_label = f"Problem {result['problem_number']}" if result["problem_number"] else "Problem"
    lines = [f"{prefix}**{result['contest']} {result['year']} — {prob_label}**"]
    if result.get("topics"):
        lines.append(f"Topics: {', '.join(result['topics'])}")
    has_sol = result.get("has_solution", False)
    if has_sol:
        lines.append("_Solution available — click Show solution below the image._")
    else:
        lines.append("_No solution available for this problem._")
    return "\n".join(lines)


def _format_solution_explanation(result: dict) -> str:
    """Return a reply directing user to the solution image panel."""
    prob_label = f"Problem {result['problem_number']}" if result["problem_number"] else "Problem"
    has_sol = result.get("has_solution", False)
    if not has_sol:
        return (
            f"No solution is available for **{result['contest']} {result['year']} — {prob_label}**."
        )
    return (
        f"Here is **{result['contest']} {result['year']} — {prob_label}**. "
        f"The solution is shown below — click **Show solution** to reveal it."
    )


def handle_list_contests() -> ContestResult:
    available = list_available_contests()
    if not available:
        return _unavailable_reply()
    lines = ["Here are the indexed Waterloo contests:\n"]
    for item in available:
        years_str = ", ".join(item["years"][:5])
        if len(item["years"]) > 5:
            years_str += f" ... (+{len(item['years']) - 5} more)"
        lines.append(f"- **{item['contest']}**: {years_str}")
    lines.append("\nAsk me to retrieve problems, explain solutions, or find practice questions!")
    return ContestResult(reply="\n".join(lines), intent="list_contests")


def handle_specific_problem(text: str) -> ContestResult:
    contest = _extract_contest(text)
    year = _extract_year(text)
    prob_num = _extract_problem_number(text)
    topic = _extract_topic(text)

    if not (contest or year or topic):
        return ContestResult(
            reply="Please specify a contest name (e.g. Euclid, Fermat) and/or year to find a specific problem.",
            intent="specific_problem",
        )

    # If contest + year + problem number all given, do exact lookup first
    if contest and year and prob_num:
        all_problems = get_by_contest_year(contest, year, n=30)
        exact = [r for r in all_problems if r["problem_number"] == prob_num]
        if exact:
            result = exact[0]
            reply = _format_problem(result)
            return ContestResult(
                reply=reply,
                problems=[result],
                intent="specific_problem",
            )
        # Not found in index - fall through to semantic search with helpful message
        desc = f"{contest} {year} Problem {prob_num}"
        return ContestResult(
            reply=f"Could not find **{desc}** in the database. Make sure the PDFs have been ingested and re-run ingestion with --clear.",
            intent="specific_problem",
        )

    # Semantic search fallback (no specific problem number, or partial info)
    results = chroma_query(
        text=text,
        n_results=8,
        contest=contest,
        year=year,
        topic=topic,
    )

    if not results:
        desc = " ".join(filter(None, [str(year) if year else None, contest]))
        return ContestResult(
            reply=f"No problems found for {desc or 'that query'}. Make sure those PDFs have been ingested.",
            intent="specific_problem",
        )

    # Filter by problem number if given
    if prob_num:
        exact = [r for r in results if r["problem_number"] == prob_num]
        if exact:
            results = exact

    lines = [f"Here {'is' if len(results) == 1 else 'are'} the matching problem{'s' if len(results) > 1 else ''}:\n"]
    for i, result in enumerate(results[:3], 1):
        lines.append(_format_problem(result, index=i))
        lines.append("")

    return ContestResult(
        reply="\n".join(lines).strip(),
        problems=results[:3],
        intent="specific_problem",
    )


def handle_explain_solution(text: str) -> ContestResult:
    contest = _extract_contest(text)
    year = _extract_year(text)
    prob_num = _extract_problem_number(text)

    results = chroma_query(
        text=text,
        n_results=5,
        contest=contest,
        year=year,
    )

    if not results:
        return ContestResult(
            reply="I couldn't find that problem in the database. Try specifying the contest name and year.",
            intent="explain_solution",
        )

    # Prefer the one with a solution and closest problem number match
    with_solution = [r for r in results if r["has_solution"]]
    candidates = with_solution or results

    if prob_num:
        exact = [r for r in candidates if r["problem_number"] == prob_num]
        if exact:
            candidates = exact

    best = candidates[0]
    explanation = _format_solution_explanation(best)

    return ContestResult(
        reply=explanation,
        problems=[best],
        intent="explain_solution",
    )


def handle_practice(text: str) -> ContestResult:
    topic = _extract_topic(text)
    grade = _extract_grade(text)
    contest = _extract_contest(text)

    # Build a richer semantic query from the request
    query_text = text
    if topic:
        kws = TOPIC_KEYWORDS.get(topic, [])
        query_text = f"{text} {' '.join(kws[:5])}"

    results = chroma_query(
        text=query_text,
        n_results=8,
        contest=contest,
        grade=grade,
        topic=topic,
    )

    if not results:
        desc = " ".join(filter(None, [topic, f"grade {grade}" if grade else None]))
        return ContestResult(
            reply=f"No practice problems found for {desc or 'that topic'}. Try a different topic or make sure the PDFs are ingested.",
            intent="practice",
        )

    # Deduplicate by contest+year+problem_num, prefer variety
    seen: set[tuple] = set()
    deduped = []
    for r in results:
        key = (r["contest"], r["year"], r["problem_number"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    topic_label = topic.replace("_", " ") if topic else "the requested area"
    grade_label = f" for grade {grade}" if grade else ""
    lines = [f"Here are practice problems for **{topic_label}**{grade_label}:\n"]

    for i, result in enumerate(deduped[:5], 1):
        lines.append(_format_problem(result, index=i))
        lines.append("")

    lines.append('_Ask me to "explain the solution" for any of these to walk through it step by step._')
    return ContestResult(
        reply="\n".join(lines).strip(),
        problems=deduped[:5],
        intent="practice",
    )


def handle_concept(text: str) -> ContestResult:
    topic = _extract_topic(text)
    results = chroma_query(text=text, n_results=3, topic=topic)

    # Build a conceptual answer using Claude's knowledge, then attach examples
    concept_reply = _concept_explanation(text, topic)

    if results:
        concept_reply += "\n\n**Example contest problems:**\n"
        for i, result in enumerate(results[:2], 1):
            concept_reply += "\n" + _format_problem(result, index=i) + "\n"

    return ContestResult(reply=concept_reply, problems=results[:2] if results else None, intent="concept")


def _concept_explanation(text: str, topic: str | None) -> str:
    """Return a built-in explanation for common contest math topics."""
    EXPLANATIONS = {
        "algebra": (
            "**Algebra** involves manipulating equations and expressions with variables. "
            "In contest math, common algebra problems involve solving systems of equations, "
            "working with polynomials, factoring, and finding roots. Key techniques include "
            "substitution, completing the square, and Vieta's formulas."
        ),
        "number_theory": (
            "**Number Theory** is the study of integers and their properties. "
            "Contest problems often involve divisibility, prime factorization, modular arithmetic, "
            "GCD/LCM, and Diophantine equations. Key theorems include Fermat's Little Theorem "
            "and the Chinese Remainder Theorem."
        ),
        "geometry": (
            "**Geometry** in contests covers both Euclidean (triangles, circles, polygons) and "
            "coordinate geometry. Key skills: angle chasing, similarity and congruence, the "
            "Pythagorean theorem, circle theorems (power of a point, inscribed angles), and "
            "area formulas. Coordinate geometry converts geometry to algebra."
        ),
        "combinatorics": (
            "**Combinatorics** is counting without listing everything. Core ideas: the "
            "multiplication and addition principles, permutations, combinations (nCr), "
            "the inclusion-exclusion principle, and the pigeonhole principle. "
            "Probability is closely related — it's counting favourable outcomes over total outcomes."
        ),
        "sequences": (
            "**Sequences & Series**: An arithmetic sequence has a constant difference between terms "
            "(sum = n/2 × (first + last)). A geometric sequence has a constant ratio "
            "(sum = a(rⁿ−1)/(r−1)). Contests also feature recursive sequences and telescoping sums."
        ),
        "calculus": (
            "**Calculus** concepts appear in senior contests (Euclid, CSMC). Key ideas: "
            "limits (what a function approaches), derivatives (instantaneous rate of change, "
            "slope of tangent), and integrals (area under a curve). The chain rule, product rule, "
            "and optimization (critical points where f'=0) are frequently tested."
        ),
        "trigonometry": (
            "**Trigonometry** relates angles to side lengths in triangles. "
            "The unit circle defines sin/cos/tan for all angles. Key identities: "
            "sin²θ + cos²θ = 1, the sine rule (a/sin A = b/sin B), and the cosine rule. "
            "Double-angle and sum formulas appear frequently in senior contests."
        ),
        "inequalities": (
            "**Inequalities**: Common contest techniques include AM-GM (arithmetic mean ≥ geometric mean), "
            "Cauchy-Schwarz, and the triangle inequality. Optimization problems often reduce to "
            "proving or applying an inequality. Absolute value inequalities are also common."
        ),
        "logic": (
            "**Proof & Logic**: Contest proofs use direct proof, proof by contradiction, "
            "mathematical induction, and case analysis. 'If and only if' (iff) requires proving "
            "both directions. Induction: prove a base case, then show if it holds for n it holds for n+1."
        ),
    }

    if topic and topic in EXPLANATIONS:
        return EXPLANATIONS[topic]

    # Try to match key words in the question
    for t, explanation in EXPLANATIONS.items():
        kws = TOPIC_KEYWORDS.get(t, [])
        if any(kw in text.lower() for kw in kws):
            return explanation

    return (
        "That's a great math concept to explore! Here are some related contest problems "
        "that might help illustrate it:"
    )


def handle_search(text: str) -> ContestResult:
    """Generic semantic search fallback."""
    contest = _extract_contest(text)
    year = _extract_year(text)
    topic = _extract_topic(text)
    grade = _extract_grade(text)

    results = chroma_query(
        text=text,
        n_results=5,
        contest=contest,
        year=year,
        topic=topic,
        grade=grade,
    )

    if not results:
        return ContestResult(
            reply=(
                "I couldn't find matching contest problems. Try:\n"
                "- Specifying a contest name (Euclid, Fryer, Pascal, etc.)\n"
                "- Naming a topic (geometry, number theory, combinatorics, etc.)\n"
                "- Asking for practice problems on a subject you find difficult"
            ),
            intent="search",
        )

    lines = ["Here are the most relevant problems I found:\n"]
    for i, result in enumerate(results[:4], 1):
        lines.append(_format_problem(result, index=i))
        lines.append("")

    return ContestResult(
        reply="\n".join(lines).strip(),
        problems=results[:4],
        intent="search",
    )


# ── Main entry point ──────────────────────────────────────────────────────────

class ContestAgent:
    """Drop-in agent for Waterloo contest math knowledge."""

    def run(self, message: str, session: dict[str, Any] | None = None) -> ContestResult:
        if collection_count() == 0:
            # Still allow concept questions even without indexed data
            intent = detect_intent(message)
            if intent == "concept":
                return handle_concept(message)
            return _unavailable_reply()

        intent = detect_intent(message)

        if intent == "list_contests":
            return handle_list_contests()
        elif intent == "specific_problem":
            return handle_specific_problem(message)
        elif intent == "explain_solution":
            return handle_explain_solution(message)
        elif intent == "practice":
            return handle_practice(message)
        elif intent == "concept":
            return handle_concept(message)
        else:
            return handle_search(message)


# Singleton
contest_agent = ContestAgent()