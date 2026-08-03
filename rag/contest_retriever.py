"""
rag/contest_retriever.py

MongoDB Atlas Vector Search-backed semantic retriever for Waterloo contest
problems. Uses a local ONNX embedding model (all-MiniLM-L6-v2) via
rag/embeddings.py to embed both the corpus (once, at ingest time) and each
query (unavoidable - every RAG query needs one embedding call for the query
text itself, this was already true with ChromaDB), and stores the vectors
in the `contest_chunks` collection so they persist across restarts instead
of living in a local Chroma index that got wiped whenever Render's ephemeral
disk was reset.

Requires an Atlas Vector Search index named VECTOR_INDEX_NAME on the
`embedding` field of the `contest_chunks` collection (see
DEPLOY_ORACLE_CLOUD.md / scripts/migrate_to_mongo.py for how to create it).
If that index isn't set up yet, `query()` transparently falls back to an
in-process brute-force cosine similarity scan so the app still works (just
slower) - you'll see a one-time warning printed when that happens.
"""

from __future__ import annotations

import heapq
import os
import time
from typing import Any, Callable

import numpy as np

from api.db import get_db

COLLECTION_NAME = "contest_chunks"
VECTOR_INDEX_NAME = os.environ.get("CONTEST_VECTOR_INDEX", "contest_vector_index")

_warned_no_index = False

# The contest corpus only changes when scripts/ingest_contests.py runs, but
# collection_count() and list_available_contests() were being hit on nearly
# every request - /contest/status alone fires on each page load, and the
# agent's browse/list/problem-set paths call them again per turn. Each call
# was a round trip to Atlas (and list_available_contests() was a full
# collection scan). Cache both for a few minutes; set CONTEST_META_CACHE_TTL=0
# to disable while re-ingesting.
_META_CACHE_TTL = float(os.environ.get("CONTEST_META_CACHE_TTL", "300"))
_meta_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, produce: Callable[[], Any]) -> Any:
    if _META_CACHE_TTL <= 0:
        return produce()
    now = time.monotonic()
    hit = _meta_cache.get(key)
    if hit is not None and (now - hit[0]) < _META_CACHE_TTL:
        return hit[1]
    value = produce()
    _meta_cache[key] = (now, value)
    return value


def invalidate_meta_cache() -> None:
    """Drop the cached corpus metadata. Call after ingesting new contests."""
    _meta_cache.clear()


def _get_embedding_function():
    from rag.embeddings import get_embedding_function
    return get_embedding_function()


def _collection():
    return get_db()[COLLECTION_NAME]


def collection_count() -> int:
    def _count() -> int:
        try:
            return _collection().estimated_document_count()
        except Exception as e:
            print(f"⚠ MongoDB count error: {e}")
            return 0

    return _cached("count", _count)


def add_chunks(chunks: list[dict]) -> None:
    col = _collection()
    if not chunks:
        return

    seen: dict[str, dict] = {}
    for c in chunks:
        seen[c["chunk_id"]] = c
    unique_chunks = list(seen.values())

    dupes = len(chunks) - len(unique_chunks)
    if dupes:
        print(f"  ℹ Deduplicated {dupes} chunk(s) before writing to MongoDB")

    embed_fn = _get_embedding_function()
    if embed_fn is None:
        raise RuntimeError("Embedding function unavailable - check onnxruntime/tokenizers install")

    documents = [c["document"] for c in unique_chunks]
    embeddings = embed_fn(documents)

    for chunk, vector in zip(unique_chunks, embeddings):
        doc = {"_id": chunk["chunk_id"], "document": chunk["document"], "embedding": list(map(float, vector))}
        doc.update(chunk["metadata"])
        col.replace_one({"_id": chunk["chunk_id"]}, doc, upsert=True)

    # The corpus just changed, so the cached count/contest list are stale.
    # Invalidating here (rather than asking every caller to remember) means
    # ingest scripts and any future writer stay correct for free.
    invalidate_meta_cache()


def _meta_to_result(doc: dict, dist: float | None = None) -> dict:
    """Convert a stored Mongo document into the clean result dict the rest
    of the app expects (same shape the old Chroma-backed version returned).
    """
    from rag.contest_ingestor import tag_topics  # local import to avoid cycles at module load

    raw_topics = [t for t in (doc.get("topics") or "").split(",") if t]
    text = doc.get("document", "")
    result = {
        "document": text,
        "contest": doc.get("contest", ""),
        "year": int(doc.get("year", 0) or 0),
        "problem_number": int(doc.get("problem_number", 0) or 0) or None,
        "part": doc.get("part") or None,
        "topics": raw_topics or tag_topics(text),
        "grades": [int(g) for g in str(doc.get("grades", "")).split(",") if g],
        "has_solution": str(doc.get("has_solution")) == "True",
        "has_diagram": str(doc.get("has_diagram")) == "True",
        "source_file": doc.get("source_file", ""),
        "pdf_path": doc.get("pdf_path", ""),
        "solution_pdf_path": doc.get("solution_pdf_path", ""),
        "page_number": int(doc.get("page_number", 0) or 0),
        "solution_page_number": int(doc.get("solution_page_number", 0) or 0),
        "solution_text": doc.get("solution_text", ""),
    }
    if dist is not None:
        result["similarity"] = round(max(0.0, dist), 3)
    return result


def _build_filter(contest: str | None, year: int | None, topic: str | None, grade: int | None) -> dict:
    conditions = []
    if contest:
        conditions.append({"contest": contest})
    if year:
        conditions.append({"year": str(year)})
    if topic:
        conditions.append({"topics": {"$regex": rf"(^|,){topic}(,|$)"}})
    if grade:
        conditions.append({"grades": {"$regex": rf"(^|,){grade}(,|$)"}})
    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _brute_force_query(query_vec: list[float], n_results: int, where: dict) -> list[dict]:
    global _warned_no_index
    if not _warned_no_index:
        print(
            "⚠ Atlas Vector Search index "
            f"'{VECTOR_INDEX_NAME}' not found - falling back to a brute-force "
            "in-process scan. This still works but is slower; create the "
            "index in Atlas (see DEPLOY_ORACLE_CLOUD.md) to speed it up."
        )
        _warned_no_index = True

    col = _collection()
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q)) or 1.0

    # Stream the vectors and keep only the running top-N. The previous version
    # did list(col.find(where)) - materialising every document, each carrying a
    # 384-float embedding *and* its full problem/solution text, into Python
    # objects at once. On the 1GB instance that is enough to OOM the process on
    # a single query, and this path runs for EVERY contest search whenever the
    # Atlas vector index is missing. Now memory is O(n_results), not O(corpus).
    top: list[tuple[float, int, Any]] = []  # min-heap; tie-break on seq
    seq = 0
    try:
        cursor = col.find(where, {"embedding": 1}).batch_size(200)
        for doc in cursor:
            emb = doc.get("embedding")
            if not emb:
                continue
            v = np.asarray(emb, dtype=np.float32)
            if v.size == 0:
                continue
            sim = float(np.dot(q, v) / (q_norm * (float(np.linalg.norm(v)) or 1.0)))
            seq += 1
            if len(top) < n_results:
                heapq.heappush(top, (sim, seq, doc["_id"]))
            elif sim > top[0][0]:
                heapq.heapreplace(top, (sim, seq, doc["_id"]))
    except Exception as e:
        print(f"⚠ MongoDB brute-force scan error: {e}")
        return []

    if not top:
        return []

    ranked = sorted(top, key=lambda t: t[0], reverse=True)
    ids = [doc_id for _, _, doc_id in ranked]
    by_id = {d["_id"]: d for d in col.find({"_id": {"$in": ids}})}
    return [
        _meta_to_result(by_id[doc_id], dist=sim)
        for sim, _, doc_id in ranked
        if doc_id in by_id
    ]


def query(
    text: str,
    n_results: int = 5,
    contest: str | None = None,
    year: int | None = None,
    grade: int | None = None,
    topic: str | None = None,
) -> list[dict]:
    if collection_count() == 0:
        return []

    embed_fn = _get_embedding_function()
    if embed_fn is None:
        return []
    query_vec = list(map(float, embed_fn([text])[0]))

    where = _build_filter(contest, year, topic, grade)

    try:
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": max(n_results * 20, 200),
                    "limit": n_results,
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]
        if where:
            # Apply the metadata filter as a post-match. (For large corpora,
            # add `where` fields as `filter` type in the Atlas index
            # definition and move this into the $vectorSearch stage instead.)
            pipeline.insert(1, {"$match": where})

        results = list(_collection().aggregate(pipeline))
        if not results and where:
            return []
        return [_meta_to_result(d, dist=d.get("_score")) for d in results]
    except Exception as e:
        if "index not found" in str(e).lower() or "$vectorSearch" in str(e):
            return _brute_force_query(query_vec, n_results, where)
        print(f"⚠ MongoDB vector search error: {e}")
        return []


def get_by_contest_year(contest: str, year: int, n: int = 20) -> list[dict]:
    if collection_count() == 0:
        return []
    try:
        docs = list(_collection().find({"contest": contest, "year": str(year)}).limit(n))
        output = [_meta_to_result(d) for d in docs]
        output.sort(key=lambda x: x["problem_number"] or 0)
        return output
    except Exception as e:
        print(f"⚠ MongoDB get error: {e}")
        return []


def list_available_contests() -> list[dict]:
    if collection_count() == 0:
        return []

    def _list() -> list[dict]:
        try:
            # Group server-side instead of streaming every document back just
            # to read two fields off it. The old version pulled the whole
            # collection over the wire on every call; this returns one row per
            # (contest, year) pair.
            pipeline = [
                {"$group": {"_id": {"contest": "$contest", "year": "$year"}}},
            ]
            seen: dict[str, set] = {}
            for row in _collection().aggregate(pipeline):
                key = row.get("_id") or {}
                c = key.get("contest") or "Unknown"
                y = str(key.get("year", "?"))
                seen.setdefault(c, set()).add(y)
            return [
                {"contest": c, "years": sorted(ys, reverse=True), "count": len(ys)}
                for c, ys in sorted(seen.items())
            ]
        except Exception as e:
            print(f"⚠ MongoDB list error: {e}")
            return []

    return _cached("contests", _list)
