"""Shared .env loader.

Populates os.environ from the project's .env file. Safe to call multiple
times and from multiple modules - only fills in variables that aren't
already set, and does nothing once the file has been read.

This is intentionally dependency-free (no python-dotenv) to match the
original loader that lived in api/agents.py.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def load_dotenv_once() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True

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

        current_value = os.environ.get(key, "")
        if current_value.strip():
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value