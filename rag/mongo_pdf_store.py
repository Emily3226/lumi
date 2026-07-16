"""
rag/mongo_pdf_store.py

Stores contest/solution PDFs in MongoDB GridFS instead of on local disk, so
the raw PDFs live in Atlas (durable, backed up, and shared across any
number of app instances) instead of depending on a persistent disk volume
on the host.

Because PyMuPDF (fitz) and the image-cropping code in
api/contest_image_router.py work against a file path, we keep a small
local-disk cache directory (like the ONNX embedding cache in
rag/embeddings.py): the first request for a given PDF downloads it from
GridFS once and writes it to CACHE_DIR; every request after that (including
after a process restart, as long as the disk isn't wiped) is a local file
read with zero network/DB round-trips.

Every PDF is addressed by a stable logical `key` (e.g.
"Euclid/2020Euclid.pdf") rather than an absolute local path, so ingestion
and serving both work the same way regardless of which machine they run on.

This cache has no size cap - a PDF written to CACHE_DIR stays there
forever unless swept. _cleanup_stale_cache_files() below deletes anything
that hasn't been read in PDF_CACHE_TTL_DAYS days, so the directory doesn't
grow without bound as more of the contest corpus gets requested over time.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import gridfs

from api.db import get_db

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "pdf_cache"
)
CACHE_DIR = os.environ.get("PDF_CACHE_DIR", _DEFAULT_CACHE_DIR)

# How long a cached PDF can go unread before it's considered stale and
# deleted (it'll simply be re-downloaded from GridFS on the next request
# for it). Override with the PDF_CACHE_TTL_DAYS env var; 0 disables cleanup.
PDF_CACHE_TTL_DAYS = int(os.environ.get("PDF_CACHE_TTL_DAYS", "30"))

# Cleanup runs once at import time (server startup), and occasionally when
# a new file is written to the cache, so long-running processes that rarely
# restart still get swept periodically instead of only at startup.
_CLEANUP_PROBABILITY_ON_WRITE = 0.02

_fs: gridfs.GridFS | None = None


def _bucket() -> gridfs.GridFS:
    global _fs
    if _fs is None:
        _fs = gridfs.GridFS(get_db(), collection="contest_pdfs")
    return _fs


def _cache_path(key: str) -> Path:
    # keys look like "Euclid/2020Euclid.pdf" - keep the same relative shape
    # on disk so the cache is easy to inspect.
    return Path(CACHE_DIR) / key


def _cleanup_stale_cache_files(ttl_days: int = PDF_CACHE_TTL_DAYS) -> int:
    """Delete cached PDFs under CACHE_DIR that haven't been read in
    `ttl_days` days. Returns the number of files removed.

    Uses max(last-accessed, last-modified) as the staleness signal rather
    than last-modified alone: this is a read-through cache, so a file that
    was downloaded long ago but is still being requested regularly should
    survive, not just recently-downloaded ones. Falling back to
    last-modified also keeps this working even on filesystems mounted with
    noatime/relatime, where access-time updates are suppressed or
    throttled - it just degrades to "time since cached" in that case
    instead of "time since last used".

    Never raises - a failed sweep should never break serving a PDF that's
    already on disk, or downloading one that isn't.
    """
    if ttl_days <= 0:
        return 0
    removed = 0
    try:
        cache_root = Path(CACHE_DIR)
        if not cache_root.exists():
            return 0
        cutoff = time.time() - (ttl_days * 86400)
        for path in cache_root.rglob("*.pdf"):
            try:
                stat = path.stat()
                last_used = max(stat.st_atime, stat.st_mtime)
                if last_used < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        # Clean up any empty contest subdirectories left behind (e.g.
        # CACHE_DIR/Euclid/ after all of its cached PDFs were removed).
        for sub in sorted(cache_root.glob("*"), reverse=True):
            try:
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            except OSError:
                continue
    except OSError:
        pass
    return removed


# Sweep once at startup so long-idle deployments don't accumulate forever
# between restarts.
_cleanup_stale_cache_files()


def upload_pdf(local_path: str | Path, key: str) -> str:
    """Upload a local PDF file into GridFS under `key` (upserts: replaces
    any existing file with the same key so re-running ingestion is safe).
    Returns the key unchanged, for convenient chaining.
    """
    fs = _bucket()
    local_path = Path(local_path)

    # Remove any previous version stored under this key first (GridFS has
    # no native upsert-by-filename).
    for existing in fs.find({"filename": key}):
        fs.delete(existing._id)

    with open(local_path, "rb") as fh:
        fs.put(fh, filename=key)

    return key


def get_local_path(key: str) -> str | None:
    """Return a local filesystem path for `key`, downloading from GridFS
    into CACHE_DIR on first use. Returns None if the key doesn't exist in
    GridFS at all.
    """
    if not key:
        return None

    cached = _cache_path(key)
    if cached.exists():
        return str(cached)

    fs = _bucket()
    grid_out = fs.find_one({"filename": key})
    if grid_out is None:
        return None

    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cached.with_suffix(cached.suffix + ".part")
    with open(tmp_path, "wb") as fh:
        fh.write(grid_out.read())
    tmp_path.replace(cached)
    if random.random() < _CLEANUP_PROBABILITY_ON_WRITE:
        _cleanup_stale_cache_files()
    return str(cached)


def exists(key: str) -> bool:
    return _bucket().find_one({"filename": key}) is not None
