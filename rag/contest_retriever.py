"""
rag/contest_retriever.py

ChromaDB-backed semantic retriever for Waterloo contest problems.
Uses sentence-transformers for embeddings (all-MiniLM-L6-v2 — fast, 
fits in memory, no API key needed).

Falls back to keyword search if ChromaDB is empty or unavailable.
"""

from __future__ import annotations

import os
import re
from typing import Any

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "contest_chroma")
COLLECTION_NAME = "waterloo_contests"

_client = None
_collection = None
_embed_fn = None


def _get_embedding_function():
    """Lazy-load the embedding function (downloads model on first call)."""
    global _embed_fn
    if _embed_fn is None:
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            _embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        except Exception:
            # Fallback: use chromadb's default (no extra deps needed)
            _embed_fn = None
    return _embed_fn


def _get_collection():
    """Lazy-init ChromaDB client + collection."""
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        embed_fn = _get_embedding_function()
        kwargs: dict[str, Any] = {"name": COLLECTION_NAME}
        if embed_fn:
            kwargs["embedding_function"] = embed_fn
        _collection = _client.get_or_create_collection(**kwargs)
        return _collection
    except Exception as e:
        print(f"⚠ ChromaDB unavailable: {e}")
        return None


def collection_count() -> int:
    """Return number of chunks indexed."""
    col = _get_collection()
    return col.count() if col else 0


def add_chunks(chunks: list[dict]) -> None:
    """
    Add ProblemChunk dicts to ChromaDB.
    Each dict must have: chunk_id, document (full text), metadata dict.
    """
    col = _get_collection()
    if not col or not chunks:
        return

    # Deduplicate by chunk_id — last entry wins (handles Gauss shared solutions)
    seen: dict[str, dict] = {}
    for c in chunks:
        seen[c["chunk_id"]] = c
    unique_chunks = list(seen.values())

    dupes = len(chunks) - len(unique_chunks)
    if dupes:
        print(f"  ℹ Deduplicated {dupes} chunk(s) before writing to ChromaDB")

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(unique_chunks), batch_size):
        batch = unique_chunks[i : i + batch_size]
        col.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def query(
    text: str,
    n_results: int = 5,
    contest: str | None = None,
    year: int | None = None,
    grade: int | None = None,
    topic: str | None = None,
) -> list[dict]:
    """
    Semantic search over contest problems.
    Returns list of result dicts with problem metadata + text.

    Optional filters:
      contest  — exact contest name (e.g. "Euclid", "Fryer")
      year     — exact year
      grade    — filter to contests covering this grade level
      topic    — filter to chunks tagged with this topic
    """
    col = _get_collection()
    if not col or col.count() == 0:
        return []

    # Build ChromaDB where clause
    where: dict = {}
    conditions = []
    if contest:
        conditions.append({"contest": {"$eq": contest}})
    if year:
        conditions.append({"year": {"$eq": str(year)}})
    if topic:
        conditions.append({"topics": {"$contains": topic}})
    # Grade filtering: grades stored as "9,10" — check substring
    if grade:
        conditions.append({"grades": {"$contains": str(grade)}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        kwargs: dict[str, Any] = {
            "query_texts": [text],
            "n_results": min(n_results, col.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
    except Exception as e:
        print(f"⚠ ChromaDB query error: {e}")
        return []

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Convert distance to similarity score (cosine distance → similarity)
        similarity = max(0.0, 1.0 - dist)
        output.append({
            "document": doc,
            "contest": meta.get("contest", ""),
            "year": int(meta.get("year", 0)),
            "problem_number": int(meta.get("problem_number", 0)) or None,
            "part": meta.get("part") or None,
            "topics": [t for t in meta.get("topics", "").split(",") if t],
            "grades": [int(g) for g in meta.get("grades", "").split(",") if g],
            "has_solution": meta.get("has_solution") == "True",
            "has_diagram": meta.get("has_diagram") == "True",
            "source_file": meta.get("source_file", ""),
            "pdf_path": meta.get("pdf_path", ""),
            "solution_pdf_path": meta.get("solution_pdf_path", ""),
            "page_number": int(meta.get("page_number", 0)),
            "solution_page_number": int(meta.get("solution_page_number", 0)),
            "similarity": round(similarity, 3),
        })

    return output


def get_by_contest_year(contest: str, year: int, n: int = 20) -> list[dict]:
    """Retrieve all problems for a specific contest year."""
    col = _get_collection()
    if not col or col.count() == 0:
        return []
    try:
        results = col.get(
            where={"$and": [{"contest": {"$eq": contest}}, {"year": {"$eq": str(year)}}]},
            include=["documents", "metadatas"],
            limit=n,
        )
        output = []
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            output.append({
                "document": doc,
                "contest": meta.get("contest", ""),
                "year": int(meta.get("year", 0)),
                "problem_number": int(meta.get("problem_number", 0)) or None,
                "part": meta.get("part") or None,
                "topics": [t for t in meta.get("topics", "").split(",") if t],
                "has_solution": meta.get("has_solution") == "True",
                "has_diagram": meta.get("has_diagram") == "True",
                "pdf_path": meta.get("pdf_path", ""),
                "solution_pdf_path": meta.get("solution_pdf_path", ""),
                "page_number": int(meta.get("page_number", 0)),
                "solution_page_number": int(meta.get("solution_page_number", 0)),
            })
        output.sort(key=lambda x: x["problem_number"] or 0)
        return output
    except Exception as e:
        print(f"⚠ ChromaDB get error: {e}")
        return []


def list_available_contests() -> list[dict]:
    """Return summary of all indexed contests (name, year counts)."""
    col = _get_collection()
    if not col or col.count() == 0:
        return []
    try:
        all_meta = col.get(include=["metadatas"])["metadatas"]
        seen: dict[str, set] = {}
        for meta in all_meta:
            c = meta.get("contest", "Unknown")
            y = meta.get("year", "?")
            seen.setdefault(c, set()).add(y)
        return [
            {"contest": c, "years": sorted(ys, reverse=True), "count": len(ys)}
            for c, ys in sorted(seen.items())
        ]
    except Exception:
        return []