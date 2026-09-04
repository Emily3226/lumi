"""Bulk personalized mentor/mentee pairing-announcement emails.

Takes a pairings CSV (one row per mentor-mentee pair; a mentor with several
mentees appears on several rows) and a pair of Jinja2 templates, renders one
email per mentor (listing all their mentees) and one per mentee, and sends -
or schedules - them via Resend (see api/email_service.py for why Resend and
not Gmail SMTP: Cloud Run blocks outbound SMTP ports the same way Render
does).

Expected CSV columns (case-sensitive, matching Auxilium's pairing sheet):
    Mentor, Mentor email, Mentor email (2), Mentor phone #, Mentor grade,
    Mentee, Mentee email, Mentee email (2), Mentee phone #, Mentee grade,
    Subjects satisfied, Time restrictions

Template variables available:
    Mentor template : mentor (MentorGroup), mentees (list[Mentee]), + extra_vars
    Mentee template : mentor (MentorGroup), mentee (Mentee), + extra_vars
    Subject template: same context as whichever email it's rendered for

MentorGroup fields: name, email, email2, phone, grade, mentees
Mentee fields:      name, email, email2, phone, grade, subjects
"""

from __future__ import annotations

import csv
import io
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import jinja2

from api.email_service import RESEND_API_URL, _credentials

logger = logging.getLogger(__name__)

_JINJA_ENV = jinja2.Environment(
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
)


@dataclass
class Mentee:
    name: str
    email: str
    email2: str | None
    phone: str
    grade: str
    subjects: str


@dataclass
class MentorGroup:
    name: str
    email: str
    email2: str | None
    phone: str
    grade: str
    mentees: list[Mentee] = field(default_factory=list)


def _clean(value: str | None) -> str:
    return (value or "").strip()


def parse_pairings_csv(csv_text: str) -> list[MentorGroup]:
    """Parse the pairings CSV, grouping rows by mentor (name + email) so a
    mentor with several mentees gets one group listing all of them.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    groups: dict[tuple[str, str], MentorGroup] = {}
    order: list[tuple[str, str]] = []

    for row in reader:
        mentor_name = _clean(row.get("Mentor"))
        mentor_email = _clean(row.get("Mentor email"))
        if not mentor_name or not mentor_email:
            continue
        key = (mentor_name, mentor_email.lower())
        if key not in groups:
            groups[key] = MentorGroup(
                name=mentor_name,
                email=mentor_email,
                email2=_clean(row.get("Mentor email (2)")) or None,
                phone=_clean(row.get("Mentor phone #")),
                grade=_clean(row.get("Mentor grade")),
            )
            order.append(key)
        groups[key].mentees.append(Mentee(
            name=_clean(row.get("Mentee")),
            email=_clean(row.get("Mentee email")),
            email2=_clean(row.get("Mentee email (2)")) or None,
            phone=_clean(row.get("Mentee phone #")),
            grade=_clean(row.get("Mentee grade")),
            subjects=_clean(row.get("Subjects satisfied")),
        ))

    return [groups[k] for k in order]


@dataclass
class RenderedEmail:
    recipient_type: str  # "mentor" | "mentee"
    to: str
    subject: str
    body: str
    mentor_name: str
    mentee_names: list[str]


def render_emails(
    mentors: list[MentorGroup],
    mentor_template: str,
    mentee_template: str,
    subject_template: str,
    extra_vars: dict[str, str],
) -> list[RenderedEmail]:
    """Render one email per mentor and one per mentee.

    Raises jinja2.TemplateError (a broken template) or jinja2.UndefinedError
    (a variable the template references that was never provided) - callers
    should surface that to whoever is editing the template, not swallow it,
    since a half-rendered email to a real family is worse than a clear error.
    """
    mentor_tmpl = _JINJA_ENV.from_string(mentor_template)
    mentee_tmpl = _JINJA_ENV.from_string(mentee_template)
    subject_tmpl = _JINJA_ENV.from_string(subject_template)

    rendered: list[RenderedEmail] = []
    for mentor in mentors:
        if mentor.email:
            ctx = {"mentor": mentor, "mentees": mentor.mentees, **extra_vars}
            rendered.append(RenderedEmail(
                recipient_type="mentor",
                to=mentor.email,
                subject=subject_tmpl.render(**ctx),
                body=mentor_tmpl.render(**ctx),
                mentor_name=mentor.name,
                mentee_names=[m.name for m in mentor.mentees],
            ))
        for mentee in mentor.mentees:
            if not mentee.email:
                continue
            mctx = {"mentor": mentor, "mentee": mentee, **extra_vars}
            rendered.append(RenderedEmail(
                recipient_type="mentee",
                to=mentee.email,
                subject=subject_tmpl.render(**mctx),
                body=mentee_tmpl.render(**mctx),
                mentor_name=mentor.name,
                mentee_names=[mentee.name],
            ))
    return rendered


def send_rendered_email(email: RenderedEmail, scheduled_at: str | None = None) -> tuple[bool, str]:
    """Send (or, with `scheduled_at`, schedule) one rendered email via Resend.

    `scheduled_at` is an ISO-8601 datetime string - Resend holds the email and
    delivers it at that time instead of immediately. Returns (ok, detail).
    """
    creds = _credentials()
    if not creds:
        return False, "RESEND_API_KEY is not configured."
    api_key, from_email = creds

    if not email.to or "@" not in email.to:
        return False, f"Invalid recipient address: {email.to!r}"

    payload: dict[str, Any] = {
        "from": from_email,
        "to": [email.to],
        "subject": email.subject,
        "text": email.body,
    }
    if scheduled_at:
        payload["scheduled_at"] = scheduled_at

    req = urllib.request.Request(
        RESEND_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; LumiBackend/1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if 200 <= resp.status < 300:
                return True, ("scheduled" if scheduled_at else "sent")
            return False, f"Resend returned HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {detail[:300]}"
    except Exception as exc:  # pragma: no cover - network errors
        return False, str(exc)
