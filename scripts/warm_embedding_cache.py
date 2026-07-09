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