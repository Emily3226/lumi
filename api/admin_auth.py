"""Shared-secret auth for the /admin router.

The admin panel (mentor/mentee/booking management) had no auth at all -
anyone who found the URL could list real mentor/mentee names and booking
details. There's no user/session system anywhere else in this app, so this
adds the lightest thing that actually closes the gap: every /admin/* request
must carry a header matching ADMIN_API_KEY.

    X-Admin-Key: <the value of ADMIN_API_KEY>

Set ADMIN_API_KEY in the environment (.env locally, Cloud Run env vars in
prod). If it isn't set, /admin/* refuses every request rather than silently
staying open - a missing key should fail closed, not fall back to no auth.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException

from api.env import load_dotenv_once

load_dotenv_once()

ADMIN_HEADER_NAME = "X-Admin-Key"


def require_admin_key(x_admin_key: str | None = Header(default=None, alias=ADMIN_HEADER_NAME)) -> None:
    configured_key = os.environ.get("ADMIN_API_KEY", "").strip()
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="Admin panel is not configured: ADMIN_API_KEY is not set.",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, configured_key):
        raise HTTPException(status_code=401, detail="Missing or invalid admin key.")
