"""
Shared Postgres (Neon) connection helper.

Provides a thin sqlite3-like interface (conn.execute(...), row["col"],
row[0], dict(row)) on top of psycopg2 so the rest of the codebase didn't
need to be rewritten call-by-call when we moved off SQLite.

Configure with the DATABASE_URL environment variable, e.g.:

    postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require

You can get this connection string from the Neon dashboard
(Project -> Connect -> Connection string).
"""

from __future__ import annotations

import os

import psycopg2

from api.env import load_dotenv_once

load_dotenv_once()

DATABASE_URL = os.environ.get("DATABASE_URL")


def _to_pg(sql: str) -> str:
    """Convert sqlite-style '?' positional placeholders to psycopg2 '%s'."""
    return sql.replace("?", "%s")


class Row(tuple):
    """A tuple that also supports column-name access, like sqlite3.Row."""

    _columns: tuple = ()

    def __new__(cls, values, columns):
        obj = super().__new__(cls, values)
        obj._columns = columns
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            return tuple.__getitem__(self, self._columns.index(key))
        return tuple.__getitem__(self, key)

    def keys(self):
        return list(self._columns)


class Cursor:
    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql: str, params=()) -> "Cursor":
        self._cur.execute(_to_pg(sql), tuple(params) if params else params)
        return self

    def _wrap(self, row):
        if row is None:
            return None
        cols = tuple(c.name for c in self._cur.description)
        return Row(row, cols)

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        return [self._wrap(r) for r in self._cur.fetchall()]

    @property
    def lastrowid(self):
        # Not used directly - see RETURNING id usage in services.py instead.
        return None


class Connection:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=()) -> Cursor:
        cur = Cursor(self._conn.cursor())
        cur.execute(sql, params)
        return cur

    def cursor(self) -> Cursor:
        return Cursor(self._conn.cursor())

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def get_db() -> Connection:
    """Return a connection to the Neon Postgres database."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it to your Neon connection string, e.g. "
            "postgresql://user:password@ep-xxxx.neon.tech/dbname?sslmode=require"
        )
    raw = psycopg2.connect(DATABASE_URL)
    return Connection(raw)