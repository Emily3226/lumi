"""Email notifications for Auxilium mentor bookings.

Sends a booking confirmation email to the mentee, CC'd to the Auxilium
coordinators, via the Resend HTTP API (https://resend.com).

Why not Gmail SMTP: Render (like most PaaS free/starter tiers) blocks
outbound SMTP ports (25/465/587) to prevent spam abuse. That connection
fails with "Network is unreachable" regardless of IPv4/IPv6 — it's not a
DNS issue, the port itself is closed. Resend sends over plain HTTPS (443),
which is never blocked, so this avoids the problem entirely.

Configure via environment variables:

    RESEND_API_KEY=re_xxxxxxxxxxxx
    RESEND_FROM_EMAIL=Auxilium Mentorship <onboarding@resend.dev>

`RESEND_FROM_EMAIL` can use Resend's shared `onboarding@resend.dev` sender
for testing with no setup, or your own address once you've verified a
domain in the Resend dashboard.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

AUXILIUM_SUPPORT_EMAIL = "ezhan322@gmail.com"
RESEND_API_URL = "https://api.resend.com/emails"
DEFAULT_FROM_EMAIL = "Auxilium Mentorship <onboarding@resend.dev>"


def _load_dotenv_file() -> None:
    """Best-effort .env loader so this module works standalone too."""
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


_load_dotenv_file()


def _credentials() -> tuple[str, str] | None:
    _load_dotenv_file()
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    from_email = os.getenv("RESEND_FROM_EMAIL", DEFAULT_FROM_EMAIL).strip()
    if not api_key:
        return None
    return api_key, from_email


def send_booking_confirmation(
    mentee_email: str,
    mentee_name: str,
    mentor_name: str,
    subject: str,
    grade: int,
    slot_label: str = "",
) -> bool:
    """Send a booking confirmation email to the mentee, CC'd to Auxilium coordinators.

    Returns True on success, False otherwise. Never raises - failures are logged
    so a booking can still complete even if the email could not be sent.
    """
    creds = _credentials()
    if not creds:
        logger.warning(
            "Booking confirmation email not sent: RESEND_API_KEY is not "
            "configured in the .env file / environment."
        )
        return False

    api_key, from_email = creds

    if not mentee_email or "@" not in mentee_email:
        logger.warning("Booking confirmation email not sent: invalid mentee email %r", mentee_email)
        return False

    slot_line = f"Weekly session: {slot_label}\n" if slot_label else ""

    body = (
        f"Hi {mentee_name},\n\n"
        "Your Auxilium mentorship pairing has been confirmed!\n\n"
        f"  Mentee:          {mentee_name}\n"
        f"  Mentor:          {mentor_name}\n"
        f"  Subject:         {subject}\n"
        f"  Mentee grade:    {grade}\n"
        f"  {slot_line}"
        "\n"
        "The Auxilium coordinators have been copied on this email and will follow up "
        "with any additional details.\n\n"
        f"If you have any questions, reach out to {AUXILIUM_SUPPORT_EMAIL}.\n\n"
        "- The Auxilium Mentorship Team"
    )

    payload = {
        "from": from_email,
        "to": [mentee_email],
        "cc": [AUXILIUM_SUPPORT_EMAIL],
        "subject": "Auxilium Mentorship - Booking Confirmation",
        "text": body,
    }

    req = urllib.request.Request(
        RESEND_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if 200 <= resp.status < 300:
                return True
            logger.warning("Resend API returned unexpected status %s", resp.status)
            return False
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.warning("Failed to send booking confirmation email: %s - %s", exc, detail)
        return False
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to send booking confirmation email: %s", exc)
        return False