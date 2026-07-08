"""Optional LLM-based reranker for mentor matching.

This module attempts to use the configured LLM (Cerebras) to rerank candidate
mentors for a given mentee query. If the LLM is not available or the call
fails for any reason, the functions return None so callers can fall back to
the existing ranking logic.
"""
from __future__ import annotations

import json
import logging
from typing import List

from api.llm_provider import call_cerebras

logger = logging.getLogger(__name__)


def _extract_text(data: dict) -> str | None:
    """Pull the assistant text out of a Cerebras chat-completions response."""
    if not isinstance(data, dict):
        return None

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text") or part.get("content")
                            if isinstance(text, str) and text:
                                text_parts.append(text)
                    if text_parts:
                        return "".join(text_parts)
    return None


def rank_candidates_langchain(mentee: dict, candidates: List[dict], top_k: int = 5) -> List[dict] | None:
    """Try to rerank `candidates` using the configured LLM (Cerebras).

    Returns a new ranked list if successful, otherwise None.
    """
    try:
        prompt = (
            "You are a mentor-ranking assistant. Given a mentee description and a list of mentor candidates, "
            "rank the mentors by suitability and return a JSON array of objects with keys: name, score (0.0-1.0), explanation.\n\n"
            f"Mentee:\n{json.dumps(mentee, ensure_ascii=False)}\n\n"
            "Candidates:\n"
            + "\n".join(
                json.dumps(
                    {
                        "name": c.get("name"),
                        "grade": c.get("grade"),
                        "subject": c.get("subject"),
                        "qualifications": c.get("qualifications"),
                    },
                    ensure_ascii=False,
                )
                for c in candidates[: top_k * 2]
            )
            + "\n\nReturn only valid JSON. Score should be a number between 0 and 1."
        )

        data = call_cerebras(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        output = _extract_text(data)
        if not output:
            logger.info("LangChain reranker: Cerebras returned no usable text")
            return None

        parsed = None
        try:
            parsed = json.loads(output)
        except Exception:
            start = output.find("[")
            end = output.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(output[start : end + 1])
                except Exception:
                    logger.warning("LangChain reranker returned unparsable JSON: %s", output)
                    return None

        if not isinstance(parsed, list):
            logger.warning("LangChain reranker returned non-list JSON: %s", parsed)
            return None

        score_map = {}
        for item in parsed:
            name = item.get("name")
            try:
                score = float(item.get("score") or 0.0)
            except Exception:
                score = 0.0
            explanation = item.get("explanation") or ""
            if name:
                score_map[str(name)] = {"score": score, "explanation": explanation}

        out = []
        for c in candidates:
            meta = score_map.get(c.get("name"), {"score": 0.0, "explanation": ""})
            newc = dict(c)
            newc["match_score"] = float(meta["score"]) if meta else 0.0
            if meta.get("explanation"):
                newc["explanation"] = meta["explanation"]
            out.append(newc)

        out.sort(key=lambda x: x.get("match_score", 0.0), reverse=True)
        logger.info(
            "LangChain reranker used for mentee %s; returning %d candidates",
            mentee.get("name"),
            len(out[:top_k]),
        )
        return out[:top_k]

    except Exception as exc:  # pragma: no cover - optional integration
        logger.info("LangChain reranker not available or failed: %s", exc)
        return None