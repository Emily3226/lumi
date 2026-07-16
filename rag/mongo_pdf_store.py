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
"""

from __future__ import annotations

import os
from pathlib import Path

import gridfs

from api.db import get_db

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "pdf_cache"
)
CACHE_DIR = os.environ.get("PDF_CACHE_DIR", _DEFAULT_CACHE_DIR)

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
    return str(cached)


def exists(key: str) -> bool:
    return _bucket().find_one({"filename": key}) is not None
