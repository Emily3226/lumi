from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _load_dotenv_file() -> None:
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


def get_llm_config() -> tuple[str, str, str]:
    api_key = _resolve_env("CEREBRAS_API_KEY")
    model = _resolve_env("CEREBRAS_MODEL", "llama3.1-8b") or "llama3.1-8b"
    base_url = _resolve_env("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1") or "https://api.cerebras.ai/v1"
    return api_key, model, base_url


def build_cerebras_payload(messages: list[dict[str, Any]], *, max_tokens: int = 1200, temperature: float = 0.2) -> dict[str, Any]:
    _, model, _ = get_llm_config()
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


def _model_candidates(primary_model: str) -> list[str]:
    candidates: list[str] = []
    for model in [primary_model, "llama3.1-8b", "gpt-oss-120b"]:
        model = model.strip()
        if model and model not in candidates:
            candidates.append(model)
    return candidates


def call_cerebras(messages: list[dict[str, Any]], *, max_tokens: int = 1200, temperature: float = 0.2) -> dict[str, Any]:
    import requests

    api_key, model, base_url = get_llm_config()
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is not set")

    last_error: Exception | None = None
    for candidate_model in _model_candidates(model):
        payload = {
            "model": candidate_model,
            "messages": [
                {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                for item in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        if response.ok:
            return response.json()

        if response.status_code == 404:
            last_error = ValueError(
                f"Cerebras model '{candidate_model}' was not found or is unavailable on this account."
            )
            continue

        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            break

    if last_error is not None:
        raise last_error

    raise ValueError("Cerebras request failed")
