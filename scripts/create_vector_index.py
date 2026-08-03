"""
scripts/create_vector_index.py

Create the Atlas Vector Search index that rag/contest_retriever.query() needs.

Without it, every contest search silently falls back to _brute_force_query(),
which streams the whole corpus through the app process to score it. That still
returns correct results, but it is far slower than Atlas doing the search, and
it puts steady memory pressure on a small instance.

Run once against your Atlas cluster:

    python scripts/create_vector_index.py

It is safe to re-run: if the index already exists the script reports its status
and exits without touching it. Index builds are asynchronous - the script polls
until Atlas reports the index queryable, so it may sit for a minute or two.

Requires MONGODB_URI in the environment (or .env), and an Atlas cluster - Atlas
Vector Search is not available on a self-hosted mongod.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo.operations import SearchIndexModel

from api.db import get_db
from rag.contest_retriever import COLLECTION_NAME, VECTOR_INDEX_NAME

POLL_SECONDS = 5
POLL_TIMEOUT = 600


def _detect_dimensions(col) -> int:
    doc = col.find_one({"embedding": {"$exists": True}}, {"embedding": 1})
    if not doc or not doc.get("embedding"):
        raise SystemExit(
            f"No documents with an `embedding` field in `{COLLECTION_NAME}`.\n"
            "Run scripts/ingest_contests.py first - there is nothing to index yet."
        )
    return len(doc["embedding"])


def main() -> None:
    db = get_db()
    col = db[COLLECTION_NAME]

    total = col.estimated_document_count()
    print(f"Collection `{COLLECTION_NAME}`: {total} documents")
    if total == 0:
        raise SystemExit("Collection is empty - run scripts/ingest_contests.py first.")

    try:
        existing = {i["name"]: i for i in col.list_search_indexes()}
    except Exception as e:
        raise SystemExit(
            f"Could not list search indexes: {e}\n"
            "Atlas Vector Search requires an Atlas-hosted cluster (a local or "
            "self-hosted mongod does not support it)."
        )

    if VECTOR_INDEX_NAME in existing:
        info = existing[VECTOR_INDEX_NAME]
        print(
            f"Index '{VECTOR_INDEX_NAME}' already exists "
            f"(status={info.get('status')}, queryable={info.get('queryable')}). Nothing to do."
        )
        return

    dims = _detect_dimensions(col)
    print(f"Detected embedding dimensions: {dims}")

    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dims,
                    # Cosine, to match the similarity _brute_force_query computes,
                    # so results stay consistent whichever path serves the query.
                    "similarity": "cosine",
                }
            ]
        },
        name=VECTOR_INDEX_NAME,
        type="vectorSearch",
    )

    print(f"Creating vector index '{VECTOR_INDEX_NAME}' ...")
    col.create_search_index(model)

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        info = next((i for i in col.list_search_indexes() if i["name"] == VECTOR_INDEX_NAME), None)
        if info and info.get("queryable"):
            print(f"  Index is queryable (status={info.get('status')}).")
            print("\nDone. Contest search will now use Atlas Vector Search instead of")
            print("the in-process brute-force fallback.")
            return
        status = info.get("status") if info else "PENDING"
        print(f"  building... status={status}")
        time.sleep(POLL_SECONDS)

    print(
        f"\nTimed out after {POLL_TIMEOUT}s waiting for the index to become queryable.\n"
        "The build usually finishes on its own - re-run this script to check its status."
    )


if __name__ == "__main__":
    main()
