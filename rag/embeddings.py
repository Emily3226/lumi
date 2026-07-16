"""
rag/embeddings.py

Shared local embedding function for both retrievers (rag/retriever.py and
rag/contest_retriever.py).

Why this exists instead of calling chromadb.utils.embedding_functions.
DefaultEmbeddingFunction() directly everywhere:

DefaultEmbeddingFunction downloads the all-MiniLM-L6-v2 ONNX model to a
hardcoded path (~/.cache/chroma/onnx_models/...) with no built-in way to
redirect it. On platforms with ephemeral/wiped local disks (Render's free
tier, most serverless/container platforms), that means the ~90MB model
gets re-downloaded on every fresh boot. On a persistent-disk VM like an
Oracle Cloud Always Free instance this is much less of an issue since the
disk survives restarts/redeploys, but we still keep this override so the
cache lives in a known, inspectable location rather than a user's home dir.

Important: this module deliberately stays on the ONNX Runtime backend
(onnxruntime + tokenizers), NOT sentence-transformers/PyTorch. Torch alone
adds 300-500MB+ of resident memory just to import and load a small model,
which is enough by itself to OOM a 512MB instance. ONNX Runtime is 5-10x
lighter and is already a transitive dependency of chromadb, so this adds
zero new heavy dependencies.

`ONNXMiniLM_L6_V2.DOWNLOAD_PATH` is a plain class/instance attribute
(see chromadb/utils/embedding_functions/onnx_mini_lm_l6_v2.py), so we can
just override it before the model is lazily downloaded on first use.

CACHE_DIR:
  - defaults to <repo_root>/data/embedding_cache (easy to inspect, and easy
    to point a Render persistent disk at if you have one), and
  - can be overridden entirely via the EMBEDDING_MODEL_CACHE_DIR env var.

To avoid ever downloading it at request time, run
`python scripts/warm_embedding_cache.py` once (or as part of your Render
build command) - the model will then already be on disk before the app
starts serving traffic.
"""

from __future__ import annotations

import os
from typing import Any

MODEL_NAME = "all-MiniLM-L6-v2"

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "embedding_cache"
)
CACHE_DIR = os.environ.get("EMBEDDING_MODEL_CACHE_DIR", _DEFAULT_CACHE_DIR)

_embed_fn = None


def get_embedding_function() -> Any:
    """Return a shared, Chroma-compatible ONNX embedding function instance
    (or None if the onnxruntime/tokenizers deps are unavailable), with its
    model download path redirected to CACHE_DIR instead of ~/.cache.
    """
    global _embed_fn
    if _embed_fn is None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
                ONNXMiniLM_L6_V2,
            )

            fn = ONNXMiniLM_L6_V2()
            # Redirect the hardcoded ~/.cache path to our controlled,
            # env-overridable directory. Must be set before the model is
            # first invoked (download happens lazily on first __call__).
            fn.DOWNLOAD_PATH = CACHE_DIR
            _embed_fn = fn
        except Exception as e:
            print(f"⚠ Local embedding function unavailable: {e}")
            _embed_fn = None
    return _embed_fn


def warm_cache() -> None:
    """Force the model to download/verify into CACHE_DIR right now, so the
    first real request doesn't pay for it. Safe to call multiple times.
    """
    fn = get_embedding_function()
    if fn is None:
        raise RuntimeError("Embedding function unavailable — check onnxruntime/tokenizers install")
    # Any call triggers the lazy download-if-missing + load.
    fn(["warm up"])