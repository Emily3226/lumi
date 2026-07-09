"""
rag/embeddings.py

Shared local embedding function for both retrievers (rag/retriever.py and
rag/contest_retriever.py).

Why this exists instead of chromadb.utils.embedding_functions.DefaultEmbeddingFunction:
Chroma's default embedding function downloads the all-MiniLM-L6-v2 ONNX model
to a HARDCODED path (~/.cache/chroma/onnx_models/...) with no way to
redirect it (see https://github.com/chroma-core/chroma/issues/1962). On
platforms like Render whose local disk is wiped between deploys/restarts
(unless you've attached a persistent disk), that means the ~90MB model gets
re-downloaded every time a fresh instance boots and someone's first message
needs it - which is slow and depends on the model host being reachable.

This module instead loads the same model via `sentence-transformers`
directly, with an explicit, project-local `cache_folder` that:
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

_model = None


def _load_model():
    global _model
    if _model is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
    return _model


class LocalEmbeddingFunction:
    """Chroma-compatible EmbeddingFunction backed by a local SentenceTransformer.

    Implements the same callable-with-list-of-strings protocol Chroma expects
    (see chromadb.api.types.EmbeddingFunction), so it can be passed anywhere
    DefaultEmbeddingFunction() was used, and can also be called directly
    (e.g. `embed_fn(["some text"])`) the way rag/retriever.py does.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = _load_model()
        return model.encode(list(input), convert_to_numpy=True).tolist()

    # chromadb's newer EmbeddingFunction protocol looks for a `name()` method
    # on custom embedding functions; harmless to include for older versions.
    @staticmethod
    def name() -> str:
        return "local-sentence-transformers-all-MiniLM-L6-v2"


def get_embedding_function() -> Any:
    """Return a shared LocalEmbeddingFunction instance, or None if unavailable."""
    try:
        return LocalEmbeddingFunction()
    except Exception as e:
        print(f"⚠ Local embedding function unavailable: {e}")
        return None