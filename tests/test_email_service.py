"""api/email_service.py - Resend-backed booking confirmation emails."""

from __future__ import annotations

import io
import urllib.error

import api.email_service as es


class _FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_credentials_does_not_raise_when_unconfigured(monkeypatch):
    """Regression test for a real bug: _credentials() called
    _load_dotenv_file(), a per-module .env loader that had already been
    removed from this file when it was consolidated into api/env.py's
    shared load_dotenv_once(). The call site was never cleaned up, so every
    invocation raised NameError - caught by agents.py's broad `except
    Exception` around the email send, so bookings always silently reported
    "I couldn't send the confirmation email automatically" regardless of
    whether RESEND_API_KEY was actually configured.
    """
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    assert es._credentials() is None  # must return None, not raise


def test_send_booking_confirmation_false_when_not_configured(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    ok = es.send_booking_confirmation(
        mentee_email="mentee@example.com",
        mentee_name="Mentee",
        mentor_name="Mentor",
        subject="math",
        grade=9,
    )
    assert ok is False


def test_send_booking_confirmation_rejects_invalid_email(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    ok = es.send_booking_confirmation(
        mentee_email="not-an-email",
        mentee_name="Mentee",
        mentor_name="Mentor",
        subject="math",
        grade=9,
    )
    assert ok is False


def test_send_booking_confirmation_success(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")

    captured = {}

    def fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        return _FakeResponse(200)

    monkeypatch.setattr(es.urllib.request, "urlopen", fake_urlopen)

    ok = es.send_booking_confirmation(
        mentee_email="mentee@example.com",
        mentee_name="Mentee",
        mentor_name="Mentor",
        subject="math",
        grade=9,
        slot_label="Fridays 14:00-15:00",
    )

    assert ok is True
    assert captured["url"] == es.RESEND_API_URL
    assert "bearer" in captured["headers"]["authorization"].lower()


def test_send_booking_confirmation_handles_http_error(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")

    def fake_urlopen(req, timeout=10):
        raise urllib.error.HTTPError(
            es.RESEND_API_URL, 422, "Unprocessable", {}, io.BytesIO(b'{"message":"invalid"}')
        )

    monkeypatch.setattr(es.urllib.request, "urlopen", fake_urlopen)

    ok = es.send_booking_confirmation(
        mentee_email="mentee@example.com",
        mentee_name="Mentee",
        mentor_name="Mentor",
        subject="math",
        grade=9,
    )
    assert ok is False
