from __future__ import annotations

import os
from typing import Any


def get_llm_config() -> tuple[str, str, str]:
    api_key = os.getenv("CEREBRAS_API_KEY", "").strip()
    model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b").strip() or "llama3.1-8b"
    base_url = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").strip() or "https://api.cerebras.ai/v1"
    return api_key, model, base_url


def build_cerebras_payload(messages: list[dict[str, Any]], *, max_tokens: int = 1200, temperature: float = 0.2) -> dict[str, Any]:
    api_key, model, _ = get_llm_config()
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


def call_cerebras(messages: list[dict[str, Any]], *, max_tokens: int = 1200, temperature: float = 0.2) -> dict[str, Any]:
    import requests

    api_key, model, base_url = get_llm_config()
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is not set")

    payload = build_cerebras_payload(messages, max_tokens=max_tokens, temperature=temperature)
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
