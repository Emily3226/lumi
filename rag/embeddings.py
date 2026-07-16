"""
rag/embeddings.py

Shared local embedding function for both retrievers (rag/retriever.py and
rag/contest_retriever.py).

Uses rag/onnx_embedder.py - a standalone, chromadb-free re-implementation
of the all-MiniLM-L6-v2 ONNX embedding model. We deliberately do NOT import
chromadb here (even though we used to, for its embedding-function utility
class): chromadb's package __init__ pulls in its full client/telemetry
surface, which costs real memory and startup time we can't spare on a
1GB-RAM instance (Oracle's free E2.1.Micro shape) - and we don't use
chromadb for storage anymore anyway (MongoDB Atlas does that now).

Important: this module deliberately stays on the ONNX Runtime backend
(onnxruntime + tokenizers), NOT sentence-transformers/PyTorch. Torch alone
adds 300-500MB+ of resident memory just to import and load a small model,
enough by itself to OOM a 1GB instance. ONNX Runtime is 5-10x lighter.

CACHE_DIR:
  - defaults to <repo_root>/data/embedding_cache (easy to inspect), and
  - can be overridden entirely via the EMBEDDING_MODEL_CACHE_DIR env var.

The model (~90MB) downloads once into CACHE_DIR and is never re-fetched
after that as long as the disk persists - true on Oracle Cloud (unlike
Render's free tier, which wiped ephemeral disk on every cold start).

To avoid ever downloading it at request time, run
`python scripts/warm_embedding_cache.py` once before serving traffic.
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
    """Return a shared, callable embedding function instance (or None if
    onnxruntime/tokenizers deps are unavailable): `fn(list[str]) -> list[np.ndarray]`.
    """
    global _embed_fn
    if _embed_fn is None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            from rag.onnx_embedder import ONNXEmbedder

            _embed_fn = ONNXEmbedder(download_path=CACHE_DIR)
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
