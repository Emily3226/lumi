"""api/pairing_emails.py - CSV parsing, template rendering, and sending."""

from __future__ import annotations

import io
import urllib.error

import jinja2
import pytest

import api.pairing_emails as pe

SAMPLE_CSV = """Mentor,Mentor email,Mentor email (2),Mentor phone #,Mentor grade,Mentee,Mentee email,Mentee email (2),Mentee phone #,Mentee grade,Subjects satisfied,Time restrictions
Leon Wu,leon@example.com,,555-1111,9,William Wang,will@example.com,,555-2222,3,Math,
Zaina Sheikh,zaina@example.com,zshei@example.com,555-3333,12,Chunrong Zhang,chun@example.com,chun2@example.com,555-4444,11,"Math, Chemistry",
Zaina Sheikh,zaina@example.com,zshei@example.com,555-3333,12,Silvia Yu,silvia@example.com,,555-5555,10,"Math, Chemistry",
"""


def test_parse_groups_rows_by_mentor():
    mentors = pe.parse_pairings_csv(SAMPLE_CSV)
    assert len(mentors) == 2
    zaina = next(m for m in mentors if m.name == "Zaina Sheikh")
    assert len(zaina.mentees) == 2
    assert [m.name for m in zaina.mentees] == ["Chunrong Zhang", "Silvia Yu"]


def test_first_name_is_just_the_first_token():
    mentors = pe.parse_pairings_csv(SAMPLE_CSV)
    leon = next(m for m in mentors if m.name == "Leon Wu")
    assert leon.first_name == "Leon"
    assert leon.mentees[0].first_name == "William"


def test_all_emails_includes_secondary_when_present():
    mentors = pe.parse_pairings_csv(SAMPLE_CSV)
    zaina = next(m for m in mentors if m.name == "Zaina Sheikh")
    assert zaina.all_emails == ["zaina@example.com", "zshei@example.com"]
    # Silvia only has one email on file.
    silvia = next(m for m in zaina.mentees if m.name == "Silvia Yu")
    assert silvia.all_emails == ["silvia@example.com"]


def test_render_emails_sends_to_every_email_on_file():
    """Regression test: rendered emails must go to ALL addresses on file for
    a person (email + email2), not just the first one.
    """
    mentors = pe.parse_pairings_csv(SAMPLE_CSV)
    rendered = pe.render_emails(
        mentors,
        mentor_template="Hi {{ mentor.first_name }}",
        mentee_template="Hi {{ mentee.first_name }}",
        subject_template="Subj",
        extra_vars={},
    )
    zaina_email = next(e for e in rendered if e.recipient_type == "mentor" and e.mentor_name == "Zaina Sheikh")
    assert zaina_email.to == ["zaina@example.com", "zshei@example.com"]

    chunrong_email = next(e for e in rendered if e.recipient_type == "mentee" and "Chunrong Zhang" in e.mentee_names)
    assert chunrong_email.to == ["chun@example.com", "chun2@example.com"]


def test_render_emails_mentee_number_labels_only_when_multiple():
    mentors = pe.parse_pairings_csv(SAMPLE_CSV)
    rendered = pe.render_emails(
        mentors,
        mentor_template="{% for mentee in mentees %}{% if mentees|length > 1 %}Mentee #{{ loop.index }} {% endif %}{{ mentee.name }}\n{% endfor %}",
        mentee_template="ignored",
        subject_template="Subj",
        extra_vars={},
    )
    leon_email = next(e for e in rendered if e.recipient_type == "mentor" and e.mentor_name == "Leon Wu")
    assert "Mentee #" not in leon_email.body  # single mentee, no number

    zaina_email = next(e for e in rendered if e.recipient_type == "mentor" and e.mentor_name == "Zaina Sheikh")
    assert "Mentee #1" in zaina_email.body
    assert "Mentee #2" in zaina_email.body


def test_sandboxed_environment_blocks_ssti():
    """Regression test for a real vulnerability: the template TEXT itself is
    admin-supplied (it's what gets edited in the admin panel), and rendering
    it with a plain jinja2.Environment allows dunder-attribute gadget chains
    (''.__class__.__mro__[1].__subclasses__() and similar) that reach
    arbitrary Python objects - a standard SSTI-to-RCE technique. The
    environment must be a SandboxedEnvironment, which blocks this.
    """
    malicious_template = '{{ "".__class__.__mro__[1].__subclasses__() }}'
    with pytest.raises(jinja2.exceptions.SecurityError):
        pe.render_emails(
            pe.parse_pairings_csv(SAMPLE_CSV),
            mentor_template=malicious_template,
            mentee_template="ignored",
            subject_template="Subj",
            extra_vars={},
        )


def test_legitimate_loops_and_conditionals_still_work_under_sandbox():
    """The sandbox must not be so restrictive it breaks normal template
    features these emails actually rely on (loops, filters, conditionals).
    """
    mentors = pe.parse_pairings_csv(SAMPLE_CSV)
    rendered = pe.render_emails(
        mentors,
        mentor_template=(
            "{% for mentee in mentees %}"
            "{% if mentee.email2 %}{{ mentee.name }} has 2 emails{% else %}{{ mentee.name }} has 1 email{% endif %}\n"
            "{% endfor %}"
        ),
        mentee_template="ignored",
        subject_template="Subj",
        extra_vars={},
    )
    zaina_email = next(e for e in rendered if e.recipient_type == "mentor" and e.mentor_name == "Zaina Sheikh")
    assert "Chunrong Zhang has 2 emails" in zaina_email.body
    assert "Silvia Yu has 1 email" in zaina_email.body


class _FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_send_rendered_email_includes_every_recipient(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    captured = {}

    def fake_urlopen(req, timeout=15):
        import json
        captured["payload"] = json.loads(req.data)
        return _FakeResponse(200)

    monkeypatch.setattr(pe.urllib.request, "urlopen", fake_urlopen)

    email = pe.RenderedEmail(
        recipient_type="mentor",
        to=["one@example.com", "two@example.com"],
        subject="Subj",
        body="Body",
        mentor_name="Test",
        mentee_names=[],
    )
    ok, detail = pe.send_rendered_email(email)
    assert ok is True
    assert captured["payload"]["to"] == ["one@example.com", "two@example.com"]


def test_send_rendered_email_drops_invalid_addresses_but_keeps_valid_ones(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    captured = {}

    def fake_urlopen(req, timeout=15):
        import json
        captured["payload"] = json.loads(req.data)
        return _FakeResponse(200)

    monkeypatch.setattr(pe.urllib.request, "urlopen", fake_urlopen)

    email = pe.RenderedEmail(
        recipient_type="mentee",
        to=["not-an-email", "valid@example.com"],
        subject="Subj",
        body="Body",
        mentor_name="Test",
        mentee_names=["Test Mentee"],
    )
    ok, _ = pe.send_rendered_email(email)
    assert ok is True
    assert captured["payload"]["to"] == ["valid@example.com"]


def test_send_rendered_email_fails_when_no_valid_recipients(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    email = pe.RenderedEmail(
        recipient_type="mentee", to=[], subject="Subj", body="Body", mentor_name="Test", mentee_names=[]
    )
    ok, detail = pe.send_rendered_email(email)
    assert ok is False


def test_send_rendered_email_passes_scheduled_at(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    captured = {}

    def fake_urlopen(req, timeout=15):
        import json
        captured["payload"] = json.loads(req.data)
        return _FakeResponse(200)

    monkeypatch.setattr(pe.urllib.request, "urlopen", fake_urlopen)

    email = pe.RenderedEmail(
        recipient_type="mentor", to=["a@example.com"], subject="s", body="b", mentor_name="m", mentee_names=[]
    )
    ok, detail = pe.send_rendered_email(email, scheduled_at="2026-08-01T09:00:00Z")
    assert ok is True
    assert detail == "scheduled"
    assert captured["payload"]["scheduled_at"] == "2026-08-01T09:00:00Z"
