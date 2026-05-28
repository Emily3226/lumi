"""Optional LangChain-based reranker for mentor matching.

This module attempts to use LangChain + a configured LLM to rerank candidate mentors
for a given mentee query. If LangChain or a suitable LLM is not available, the
functions return None so callers can fall back to the existing ranking logic.
"""
from __future__ import annotations

import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def rank_candidates_langchain(mentee: dict, candidates: List[dict], top_k: int = 5) -> List[dict] | None:
    """Try to rerank `candidates` using LangChain.

    Returns a new ranked list if successful, otherwise None.
    """
    try:
        from langchain import PromptTemplate, LLMChain

        # Prefer a small Groq-backed LLM wrapper if possible, else fallback to OpenAI adapter
        # Build a minimal Groq-backed LangChain LLM implementation only.
        try:
            from langchain.llms.base import LLM
            import os
            import requests

            class GroqLLM(LLM):
                def __init__(self, model: str | None = None, temperature: float = 0.2):
                    self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
                    self.temperature = temperature

                def _call(self, prompt: str, stop: None | list[str] = None) -> str:
                    api_key = os.getenv("GROQ_API_KEY", "").strip()
                    if not api_key:
                        raise ValueError("GROQ_API_KEY not configured for GroqLLM")
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": float(self.temperature),
                    }
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                        json=payload,
                        timeout=20,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        msg = first.get("message") if isinstance(first, dict) else None
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if content:
                                return str(content)
                    return ""

            LLMClass = GroqLLM
            logger.info("LangChain matcher will use GroqLLM if GROQ_API_KEY is set")
        except Exception as exc:
            logger.info("GroqLLM not available for LangChain reranker: %s", exc)
            return None

        # Build prompt listing candidates and asking for ranking in JSON
        tmpl = (
            "You are a mentor-ranking assistant. Given a mentee description and a list of mentor candidates, "
            "rank the mentors by suitability and return a JSON array of objects with keys: name, score (0.0-1.0), explanation.\n\n"
            "Mentee:\n{mentee}\n\nCandidates:\n{cands}\n\n"
            "Return only valid JSON. Score should be a number between 0 and 1. "
        )

        prompt = PromptTemplate(template=tmpl, input_variables=["mentee", "cands"])  # type: ignore

        # Instantiate LLM (expects environment configuration, e.g., OPENAI_API_KEY)
        llm = LLMClass(temperature=0.2)  # type: ignore
        chain = LLMChain(llm=llm, prompt=prompt)

        mentee_str = json.dumps(mentee, ensure_ascii=False)
        cand_lines = []
        for c in candidates[: top_k * 2]:
            # basic fields; avoid dumping nested objects
            cand_lines.append(json.dumps({
                "name": c.get("name"),
                "grade": c.get("grade"),
                "subject": c.get("subject"),
                "qualifications": c.get("qualifications"),
            }, ensure_ascii=False))

        cands_str = "\n".join(cand_lines)
        output = chain.run(mentee=mentee_str, cands=cands_str)

        # Parse JSON from output
        parsed = None
        try:
            parsed = json.loads(output)
        except Exception:
            # Attempt to find JSON substring
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

        # Build mapping name -> score/explanation
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

        # Attach scores to original candidates and sort
        out = []
        for c in candidates:
            meta = score_map.get(c.get("name"), {"score": 0.0, "explanation": ""})
            newc = dict(c)
            newc["match_score"] = float(meta["score"]) if meta else 0.0
            if meta.get("explanation"):
                newc["explanation"] = meta["explanation"]
            out.append(newc)

        out.sort(key=lambda x: x.get("match_score", 0.0), reverse=True)
        try:
            logger.info("LangChain reranker used for mentee %s; returning %d candidates", mentee.get("name"), len(out[:top_k]))
        except Exception:
            logger.info("LangChain reranker used; returning candidates")
        return out[:top_k]

    except Exception as exc:  # pragma: no cover - optional integration
        logger.info("LangChain reranker not available or failed: %s", exc)
        return None
