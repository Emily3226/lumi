"""
scripts/warm_embedding_cache.py

Pre-downloads and caches the local ONNX embedding model (see
rag/embeddings.py) so it's already on disk before the app starts serving
traffic, instead of downloading it lazily the first time a chat message
needs mentor matching or contest retrieval.

Run this once locally, or add it to your deploy's build step (e.g. Render's
"Build Command": `pip install -r requirements.txt && python scripts/warm_embedding_cache.py`)
so the download happens during build, not on a user's first request.

Note: this only avoids *repeated* downloads if EMBEDDING_MODEL_CACHE_DIR (or
the default data/embedding_cache/) actually persists between deploys/restarts
- e.g. baked into a Docker image, or on a Render persistent disk. If your
disk is fully ephemeral, this still helps (moves the download off the
request path and into the build step), but won't eliminate it entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embeddings import CACHE_DIR, MODEL_NAME, warm_cache


def main() -> None:
    print(f"Downloading/verifying {MODEL_NAME} into {CACHE_DIR} ...")
    warm_cache()
    print("Done. The embedding model is cached and ready.")


if __name__ == "__main__":
    main()