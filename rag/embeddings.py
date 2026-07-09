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
    global _embed_fn
    if _embed_fn is None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
            fn = ONNXMiniLM_L6_V2()
            fn.DOWNLOAD_PATH = CACHE_DIR
            _embed_fn = fn
        except Exception as e:
            print(f"⚠ Local embedding function unavailable: {e}")
            _embed_fn = None
    return _embed_fn


def warm_cache() -> None:
    fn = get_embedding_function()
    if fn is None:
        raise RuntimeError("Embedding function unavailable")
    fn(["warm up"])