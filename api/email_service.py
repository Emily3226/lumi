"""Email notifications for Auxilium mentor bookings.

Sends a booking confirmation email to the mentee, CC'd to the Auxilium
coordinators, using a Gmail account configured via environment variables:

    GMAIL_USER=auxilium.mentorship@gmail.com
    GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx

`GMAIL_APP_PASSWORD` must be a Gmail "App Password" (Google Account ->
Security -> 2-Step Verification -> App passwords), not the normal account
password.
"""

from __future__ import annotations

import socket
import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)

AUXILIUM_SUPPORT_EMAIL = "ezhan322@gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


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
    user = os.getenv("GMAIL_USER", "").strip()
    password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    if not user or not password:
        return None
    return user, password


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
            "Booking confirmation email not sent: GMAIL_USER / GMAIL_APP_PASSWORD "
            "are not configured in the .env file."
        )
        return False

    gmail_user, gmail_password = creds

    if not mentee_email or "@" not in mentee_email:
        logger.warning("Booking confirmation email not sent: invalid mentee email %r", mentee_email)
        return False

    message = EmailMessage()
    message["Subject"] = "Auxilium Mentorship - Booking Confirmation"
    message["From"] = gmail_user
    message["To"] = mentee_email

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
    message.set_content(body)

    recipients = [mentee_email, AUXILIUM_SUPPORT_EMAIL]
    
    try:
    # Force IPv4 — Render containers often lack outbound IPv6 routing,
    # and smtplib tries AAAA records first if DNS returns one, causing
    # "Network is unreachable" even though IPv4 works fine.
        addr_info = socket.getaddrinfo(SMTP_HOST, SMTP_PORT, socket.AF_INET, socket.SOCK_STREAM)
        ipv4_host = addr_info[0][4][0]

        with smtplib.SMTP_SSL(ipv4_host, SMTP_PORT, timeout=10) as smtp:
            smtp.login(gmail_user, gmail_password)
            smtp.send_message(message, to_addrs=recipients)
        return True
    except Exception as exc:
        logger.warning("Failed to send booking confirmation email: %s", exc)
        return False