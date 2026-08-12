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
       is several times per provider per request. Now it runs once.
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




# ── Providers ─────────────────────────────────────────────────────────────────
# Every provider below speaks the OpenAI chat-completions wire format, so a
# single request builder covers all of them - only the API key, model id, and
# base URL differ.
#
# Cerebras is no longer the default: its free tier ends 2026-08-17, after which
# accounts move to a credit-based plan that requires a payment method. Groq
# leads the chain instead - same wire format, no credit card, and this repo
# already carried a GROQ_API_KEY from before the Cerebras migration.
#
# Override the order with LLM_PROVIDER_ORDER (comma-separated), or pin a single
# provider with LLM_PROVIDER=groq.

PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "base_url_env": "GEMINI_BASE_URL",
        # Deliberately the floating alias, not a pinned id. Pinning is what
        # broke this provider: GEMINI_MODEL was set to gemini-2.0-flash, which
        # now returns 429 (no free quota), and gemini-2.5-flash 404s outright.
        # The alias follows whatever the current free Flash model is.
        "default_model": "gemini-flash-latest",
        # Google's OpenAI-compatibility layer, so the same payload works.
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "base_url_env": "GROQ_BASE_URL",
        # NOT llama-3.3-70b-versatile: Groq shuts that down for free and
        # developer tiers on 2026-08-16. gpt-oss-120b is their stated
        # migration target (qwen/qwen3.6-27b is the other option).
        "default_model": "openai/gpt-oss-120b",
        "default_base_url": "https://api.groq.com/openai/v1",
    },
    "cerebras": {
        "key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        "base_url_env": "CEREBRAS_BASE_URL",
        "default_model": "gpt-oss-120b",
        "default_base_url": "https://api.cerebras.ai/v1",
    },
    "cloudflare": {
        # Optional fourth provider in a different failure domain from the
        # others (10k neurons/day free). Needs the account id baked into the
        # URL, so there is no usable default - set CLOUDFLARE_BASE_URL to
        # https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/v1
        # and CLOUDFLARE_API_KEY to a Workers AI token to enable it.
        "key_env": "CLOUDFLARE_API_KEY",
        "model_env": "CLOUDFLARE_MODEL",
        "base_url_env": "CLOUDFLARE_BASE_URL",
        "default_model": "@cf/meta/llama-3.1-8b-instruct",
        "default_base_url": "",
    },
}

# Gemini leads: it is the most durable free tier of the four (1,500 requests/day
# on Flash, no card, no expiry) and it is the only one here with no announced
# end date. Cerebras is last because its free tier ends 2026-08-17 - delete it
# from this tuple once that lands.
_DEFAULT_ORDER = ("gemini", "groq", "cerebras", "cloudflare")


def _provider_config(name: str) -> tuple[str, str, str]:
    """Return (api_key, model, base_url) for `name`. Key is "" if unconfigured."""
    spec = PROVIDERS[name]
    api_key = _resolve_env(spec["key_env"])
    model = _resolve_env(spec["model_env"], spec["default_model"]) or spec["default_model"]
    base_url = _resolve_env(spec["base_url_env"], spec["default_base_url"]) or spec["default_base_url"]
    return api_key, model, base_url


def _provider_chain() -> list[str]:
    """Providers to try, in order, that actually have an API key configured."""
    pinned = _resolve_env("LLM_PROVIDER").strip().lower()
    if pinned and pinned in PROVIDERS:
        order = [pinned]
    else:
        raw = _resolve_env("LLM_PROVIDER_ORDER").strip()
        order = [p.strip().lower() for p in raw.split(",") if p.strip()] if raw else list(_DEFAULT_ORDER)

    chain = []
    for name in order:
        if name not in PROVIDERS or name in chain:
            continue
        api_key, _model, base_url = _provider_config(name)
        # Both are required: a provider with a key but no base URL (Cloudflare
        # without its account-scoped URL) would otherwise burn a failed request
        # on every call before falling through.
        if api_key and base_url:
            chain.append(name)
    return chain


def get_llm_config() -> tuple[str, str, str]:
    """(api_key, model, base_url) of the first configured provider.

    Kept for callers that only want to know whether *some* LLM is reachable
    (api/agents.py `_llm_api_key`) or which model is answering.
    """
    chain = _provider_chain()
    if not chain:
        # Nothing configured - report the head of the default order so error
        # messages name a real provider instead of an empty string.
        return _provider_config(_DEFAULT_ORDER[0])
    return _provider_config(chain[0])


def build_payload(messages: list[dict[str, Any]], model: str, *, max_tokens: int = 1200, temperature: float = 0.2) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in messages
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def build_cerebras_payload(messages: list[dict[str, Any]], *, max_tokens: int = 1200, temperature: float = 0.2) -> dict[str, Any]:
    """Back-compat shim - builds a payload for the active provider's model."""
    _, model, _ = get_llm_config()
    return build_payload(messages, model, max_tokens=max_tokens, temperature=temperature)


def call_llm(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    timeout: int = 30,
) -> dict[str, Any]:
    """POST to the first working provider in the chain.

    `timeout` is a *total* budget across providers, not per-attempt: failing
    over must not multiply a caller's worst-case wait (api/agents.py's prompt
    rewrite passes 12s expecting to be back within 12s). Each attempt gets
    whatever is left, and we stop once the budget is spent.
    """
    import requests

    chain = _provider_chain()
    if not chain:
        raise ValueError(
            "No LLM provider configured. Set one of: "
            + ", ".join(spec["key_env"] for spec in PROVIDERS.values())
        )

    deadline = time.monotonic() + max(1, timeout)
    last_error: Exception | None = None

    for name in chain:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            break

        api_key, model, base_url = _provider_config(name)
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=build_payload(messages, model, max_tokens=max_tokens, temperature=temperature),
                timeout=remaining,
            )
        except Exception as exc:  # network error / timeout - try the next provider
            last_error = exc
            continue

        if response.ok:
            return response.json()

        # 4xx that isn't rate limiting is a config problem (bad key, retired
        # model). Record it and let the next provider try.
        detail = response.text[:200].replace("\n", " ")
        last_error = ValueError(f"{name} returned HTTP {response.status_code}: {detail}")

    if last_error is not None:
        raise last_error
    raise ValueError("LLM request failed")


# Back-compat alias: api/agents.py and api/contest_agent.py import this name.
# It is no longer Cerebras-specific - it dispatches over the provider chain.
call_cerebras = call_llm
