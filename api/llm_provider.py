from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


_dotenv_loaded = False


def _load_dotenv_file() -> None:
    """Populate os.environ from .env, WITHOUT overriding real env vars.

    Two fixes over the original:

    1. It used to do an unconditional `os.environ[key] = value`, so a stale
       .env checked out on the server silently overrode variables set in the
       actual deployment environment - backwards from how every other dotenv
       loader behaves (and from api/env.py, which this now matches).
    2. It re-read and re-parsed the file on every _resolve_env() call, which
       is several times per request. Now it runs once.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True

    repo_root = Path(__file__).resolve().parents[1]
    env_paths = [repo_root / ".env", repo_root / ".venv" / ".env"]

    for env_path in env_paths:
        if not env_path.exists():
            continue

        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

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

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            # Real environment variables win over the file.
            if os.environ.get(key, "").strip():
                continue
            os.environ[key] = value


def _windows_env_fallback(name: str) -> str:
    if os.name != "nt":
        return ""

    try:
        import winreg
    except Exception:
        return ""

    registry_paths = [
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ]

    for root, subkey in registry_paths:
        try:
            with winreg.OpenKey(root, subkey) as handle:
                value, _ = winreg.QueryValueEx(handle, name)
                if isinstance(value, str):
                    return value.strip()
        except Exception:
            continue

    return ""


def _resolve_env(name: str, default: str = "") -> str:
    _load_dotenv_file()
    value = os.getenv(name, "").strip()
    if value:
        return value

    value = _windows_env_fallback(name)
    if value:
        return value

    return default


# ── Gemini ────────────────────────────────────────────────────────────────────
# This module used to fan out over four providers (Groq / Cerebras / Cloudflare
# / Gemini) that all spoke the OpenAI chat-completions wire format. It is now
# Gemini-only, and talks to Google's *native* REST API:
#
#   POST {base}/models/{model}:generateContent   with an x-goog-api-key header
#
# Callers still receive an OpenAI-shaped dict (choices[0].message.content),
# because api/agents.py, api/contest_agent.py and rag/langchain_matcher.py all
# parse that shape - see _to_openai_shape() below.

# Deliberately the floating alias, not a pinned id. Pinning is what broke this
# provider before: GEMINI_MODEL was set to gemini-2.0-flash, which returns 429
# (no free quota), and gemini-2.5-flash 404s outright. The alias follows
# whatever the current free Flash model is.
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Flash "thinks" before answering, and those tokens count against
# maxOutputTokens without ever appearing in the reply. Measured overhead is
# ~300-400 tokens, so the small budgets callers use for short tasks (250 for
# the prompt rewrite, for example) get consumed entirely by thinking and come
# back empty with finishReason="MAX_TOKENS". Floor the budget so short prompts
# still leave room for an actual answer.
MIN_OUTPUT_TOKENS = 700

API_KEY_ENV = "GEMINI_API_KEY"


def get_llm_config() -> tuple[str, str, str]:
    """(api_key, model, base_url) for Gemini.

    Kept for callers that only want to know whether an LLM is reachable
    (api/agents.py `_llm_api_key`) or which model is answering.
    """
    api_key = _resolve_env(API_KEY_ENV)
    model = _resolve_env("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    base_url = _resolve_env("GEMINI_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    return api_key, model, base_url


def build_payload(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Translate OpenAI-style `messages` into a Gemini generateContent body.

    - `system` messages become a single top-level systemInstruction (Gemini has
      no system role inside `contents`; several are joined with blank lines).
    - `assistant` becomes `model`; everything else becomes `user`.
    - Consecutive same-role turns are merged, because Gemini expects the
      conversation to alternate.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for item in messages:
        role = str(item.get("role", "user")).strip().lower()
        text = str(item.get("content", ""))
        if not text:
            continue

        if role == "system":
            system_parts.append(text)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

    # A request with empty `contents` is rejected, so fold a system-only
    # prompt into the first user turn instead.
    if not contents and system_parts:
        contents = [{"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]}]
        system_parts = []

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max(max_tokens, MIN_OUTPUT_TOKENS),
            "temperature": temperature,
        },
    }

    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    # Only sent when explicitly configured: accepted values differ between
    # model generations (2.5 Flash takes a token budget, 0 to disable; Gemini 3
    # rejects 0 on some models), so guessing here would break whichever model
    # the floating alias currently points at.
    thinking_budget = _resolve_env("GEMINI_THINKING_BUDGET")
    if thinking_budget:
        try:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": int(thinking_budget)
            }
        except ValueError:
            pass

    return payload


def _to_openai_shape(data: dict[str, Any], model: str) -> dict[str, Any]:
    """Map a generateContent response onto the choices/message/content shape."""
    candidates = data.get("candidates") if isinstance(data, dict) else None
    text_parts: list[str] = []
    finish_reason = None

    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            finish_reason = first.get("finishReason")
            content = first.get("content")
            if isinstance(content, dict):
                for part in content.get("parts") or []:
                    # Skip the model's own thinking - only the answer is text
                    # the callers should see.
                    if isinstance(part, dict) and not part.get("thought"):
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            text_parts.append(text)

    usage = (data.get("usageMetadata") if isinstance(data, dict) else None) or {}

    return {
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        },
        # The untouched response, for anything that needs promptFeedback or the
        # safety ratings.
        "raw": data,
    }


# Free-tier Flash returns 503 UNAVAILABLE ("experiencing high demand") in
# bursts, and 429 when the per-minute quota is hit. The old multi-provider
# chain used to absorb those by failing over; with Gemini as the only provider,
# a short retry on the same model is what is left. Anything else (400 bad
# payload, 404 retired model, 403 bad key) is a config problem and fails fast.
_RETRY_STATUS = {429, 503}
_RETRY_BACKOFF_SECONDS = 1.0


def call_llm(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    timeout: int = 30,
) -> dict[str, Any]:
    """Call Gemini generateContent and return an OpenAI-shaped response dict.

    `timeout` is a *total* budget, not per-attempt: retrying must not multiply
    a caller's worst-case wait (api/agents.py's prompt rewrite passes 12s
    expecting to be back within 12s). Each attempt gets whatever is left.
    """
    import requests

    api_key, model, base_url = get_llm_config()
    if not api_key:
        raise ValueError(f"No LLM provider configured. Set {API_KEY_ENV}.")

    url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
    payload = build_payload(messages, max_tokens=max_tokens, temperature=temperature)
    deadline = time.monotonic() + max(1, timeout)
    last_error: Exception | None = None

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            break

        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=remaining,
            )
        except Exception as exc:  # network error / timeout
            raise exc if last_error is None else last_error

        if response.ok:
            return _to_openai_shape(response.json(), model)

        detail = response.text[:200].replace("\n", " ")
        last_error = ValueError(f"gemini returned HTTP {response.status_code}: {detail}")
        if response.status_code not in _RETRY_STATUS:
            break

        if deadline - time.monotonic() <= _RETRY_BACKOFF_SECONDS + 0.5:
            break
        time.sleep(_RETRY_BACKOFF_SECONDS)

    if last_error is not None:
        raise last_error
    raise ValueError("LLM request failed")
