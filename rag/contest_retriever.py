"""
rag/contest_retriever.py

ChromaDB-backed semantic retriever for Waterloo contest problems.
Uses sentence-transformers for embeddings (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import os
from typing import Any

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "contest_chroma")
COLLECTION_NAME = "waterloo_contests"

_client = None
_collection = None
_embed_fn = None


def _is_missing_collection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "collection" in text and "does not exist" in text


def _reset_collection_cache() -> None:
    global _collection
    _collection = None


def _count_with_recovery(col) -> int:
    try:
        return col.count()
    except Exception as e:
        if _is_missing_collection_error(e):
            _reset_collection_cache()
            fresh = _get_collection()
            if not fresh:
                return 0
            try:
                return fresh.count()
            except Exception as e2:
                print(f"⚠ ChromaDB count error after recovery: {e2}")
                return 0
        print(f"⚠ ChromaDB count error: {e}")
        return 0


_TOPIC_SIGNALS: dict[str, list[str]] = {
    "algebra": ["equation", "polynomial", "quadratic", "factor", "expression", "solve"],
    "number_theory": ["integer", "prime", "divisible", "remainder", "mod", "gcd", "lcm"],
    "geometry": ["triangle", "circle", "angle", "polygon", "perimeter", "area", "chord", "radius"],
    "combinatorics": ["permutation", "combination", "probability", "arrangement", "ways"],
    "sequences": ["sequence", "recurrence", "fibonacci", "common ratio", "common difference", "nth term", "n-th term"],
    "inequalities": ["inequality", "maximum", "minimum", "bound", "absolute value"],
    "trigonometry": ["sine", "cosine", "tangent", "sin", "cos", "tan", "radian"],
    "logic": ["prove", "proof", "if and only if", "contradiction", "induction"],
}


def _contains_signal(text: str, signals: list[str]) -> bool:
    return any(s in text for s in signals)


def _normalize_topics(doc: str, raw_topics: list[str]) -> list[str]:
    text = (doc or "").lower()

    cleaned: list[str] = []
    for topic in raw_topics:
        if topic == "calculus":
            # Waterloo contests should not be labeled calculus; remap using text cues.
            continue
        signals = _TOPIC_SIGNALS.get(topic)
        if signals and not _contains_signal(text, signals):
            continue
        cleaned.append(topic)

    # If labels were empty/filtered, infer one or two likely topics from the problem text.
    if not cleaned:
        for topic, signals in _TOPIC_SIGNALS.items():
            if _contains_signal(text, signals):
                cleaned.append(topic)
            if len(cleaned) >= 2:
                break

    if not cleaned:
        cleaned = ["algebra"]

    return cleaned


def _get_embedding_function():
    global _embed_fn
    if _embed_fn is None:
        from rag.embeddings import get_embedding_function
        _embed_fn = get_embedding_function()
    return _embed_fn


def _get_collection():
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
    col = _get_collection()
    return _count_with_recovery(col) if col else 0


def add_chunks(chunks: list[dict]) -> None:
    col = _get_collection()
    if not col or not chunks:
        return

    seen: dict[str, dict] = {}
    for c in chunks:
        seen[c["chunk_id"]] = c
    unique_chunks = list(seen.values())

    dupes = len(chunks) - len(unique_chunks)
    if dupes:
        print(f"  ℹ Deduplicated {dupes} chunk(s) before writing to ChromaDB")

    batch_size = 100
    for i in range(0, len(unique_chunks), batch_size):
        batch = unique_chunks[i: i + batch_size]
        payload = {
            "ids": [c["chunk_id"] for c in batch],
            "documents": [c["document"] for c in batch],
            "metadatas": [c["metadata"] for c in batch],
        }
        try:
            col.upsert(**payload)
        except Exception as e:
            if _is_missing_collection_error(e):
                _reset_collection_cache()
                col = _get_collection()
                if not col:
                    print("⚠ ChromaDB unavailable while writing chunks")
                    return
                col.upsert(**payload)
            else:
                raise


def _meta_to_result(doc: str, meta: dict, dist: float | None = None) -> dict:
    """Convert a ChromaDB metadata dict into a clean result dict."""
    raw_topics = [t for t in meta.get("topics", "").split(",") if t]
    result = {
        "document": doc,
        "contest": meta.get("contest", ""),
        "year": int(meta.get("year", 0)),
        "problem_number": int(meta.get("problem_number", 0)) or None,
        "part": meta.get("part") or None,
        "topics": _normalize_topics(doc, raw_topics),
        "grades": [int(g) for g in meta.get("grades", "").split(",") if g],
        "has_solution": meta.get("has_solution") == "True",
        "has_diagram": meta.get("has_diagram") == "True",
        "source_file": meta.get("source_file", ""),
        "pdf_path": meta.get("pdf_path", ""),
        "solution_pdf_path": meta.get("solution_pdf_path", ""),
        "page_number": int(meta.get("page_number", 0)),
        "solution_page_number": int(meta.get("solution_page_number", 0)),
        # solution_text stored directly in metadata since v5
        "solution_text": meta.get("solution_text", ""),
    }
    if dist is not None:
        result["similarity"] = round(max(0.0, 1.0 - dist), 3)
    return result


def query(
    text: str,
    n_results: int = 5,
    contest: str | None = None,
    year: int | None = None,
    grade: int | None = None,
    topic: str | None = None,
) -> list[dict]:
    col = _get_collection()
    if not col:
        return []

    total = _count_with_recovery(col)
    if total == 0:
        return []

    col = _get_collection()
    if not col:
        return []

    where: dict = {}
    conditions = []
    if contest:
        conditions.append({"contest": {"$eq": contest}})
    if year:
        conditions.append({"year": {"$eq": str(year)}})
    if topic:
        conditions.append({"topics": {"$contains": topic}})
    if grade:
        conditions.append({"grades": {"$contains": str(grade)}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        kwargs: dict[str, Any] = {
            "query_texts": [text],
            "n_results": min(n_results, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
    except Exception as e:
        if _is_missing_collection_error(e):
            _reset_collection_cache()
            col = _get_collection()
            if not col:
                return []
            try:
                kwargs = {
                    "query_texts": [text],
                    "n_results": min(n_results, _count_with_recovery(col)),
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    kwargs["where"] = where
                results = col.query(**kwargs)
            except Exception as e2:
                print(f"⚠ ChromaDB query error after recovery: {e2}")
                return []
        else:
            print(f"⚠ ChromaDB query error: {e}")
            return []

    output = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        output.append(_meta_to_result(doc, meta, dist))

    return output


def get_by_contest_year(contest: str, year: int, n: int = 20) -> list[dict]:
    col = _get_collection()
    if not col or _count_with_recovery(col) == 0:
        return []
    try:
        results = col.get(
            where={"$and": [
                {"contest": {"$eq": contest}},
                {"year": {"$eq": str(year)}},
            ]},
            include=["documents", "metadatas"],
            limit=n,
        )
        output = [
            _meta_to_result(doc, meta)
            for doc, meta in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            )
        ]
        output.sort(key=lambda x: x["problem_number"] or 0)
        return output
    except Exception as e:
        if _is_missing_collection_error(e):
            _reset_collection_cache()
            col = _get_collection()
            if not col:
                return []
            try:
                results = col.get(
                    where={"$and": [
                        {"contest": {"$eq": contest}},
                        {"year": {"$eq": str(year)}},
                    ]},
                    include=["documents", "metadatas"],
                    limit=n,
                )
                output = [
                    _meta_to_result(doc, meta)
                    for doc, meta in zip(
                        results.get("documents", []),
                        results.get("metadatas", []),
                    )
                ]
                output.sort(key=lambda x: x["problem_number"] or 0)
                return output
            except Exception as e2:
                print(f"⚠ ChromaDB get error after recovery: {e2}")
                return []
        print(f"⚠ ChromaDB get error: {e}")
        return []


def list_available_contests() -> list[dict]:
    col = _get_collection()
    if not col or _count_with_recovery(col) == 0:
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
    except Exception as e:
        if _is_missing_collection_error(e):
            _reset_collection_cache()
            col = _get_collection()
            if not col:
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
            except Exception as e2:
                print(f"⚠ ChromaDB list error after recovery: {e2}")
                return []
        print(f"⚠ ChromaDB list error: {e}")
        return []