"""api/admin_auth.py - shared-secret protection for the /admin router."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.admin_auth import require_admin_key


def test_fails_closed_when_admin_api_key_not_configured(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_admin_key(x_admin_key="anything")
    assert exc_info.value.status_code == 503


def test_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc_info:
        require_admin_key(x_admin_key=None)
    assert exc_info.value.status_code == 401


def test_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc_info:
        require_admin_key(x_admin_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_accepts_correct_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")
    require_admin_key(x_admin_key="correct-key")  # should not raise


def test_rejects_empty_string_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "correct-key")
    with pytest.raises(HTTPException) as exc_info:
        require_admin_key(x_admin_key="")
    assert exc_info.value.status_code == 401
