"""rag/retriever.py - MentorRetriever, backed by the real MongoDB mentors
collection (integration test; read-only, never mutates data).

Regression coverage for a real bug: MentorRetriever used to load mentors from
a local data/lumi.db SQLite snapshot that was never migrated to MongoDB,
falling back to a CSV that doesn't exist outside local dev. On Cloud Run that
crashed mentor matching outright (FileNotFoundError). Mentors now come
straight from the MongoDB `mentors` collection.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_mongodb


@pytest.fixture(scope="module")
def retriever():
    from rag.retriever import MentorRetriever

    return MentorRetriever()


@requires_mongodb
def test_loads_mentors_from_mongodb_not_local_files(retriever):
    # If this were still falling back to the (deleted) local SQLite/CSV
    # files, construction above would have raised or returned zero mentors.
    assert len(retriever.mentors) > 0
    for mentor in retriever.mentors:
        assert mentor["name"]
        assert "profile_text" in mentor


@requires_mongodb
def test_retrieve_returns_ranked_results_within_top_k(retriever):
    results = retriever.retrieve("I need help with math", mentee_grade=9, top_k=3)
    assert 0 < len(results) <= 3
    for mentor in results:
        assert "similarity_score" in mentor
    # Ranked descending by similarity.
    scores = [m["similarity_score"] for m in results]
    assert scores == sorted(scores, reverse=True)


@requires_mongodb
def test_retrieve_on_empty_query_still_returns_something(retriever):
    results = retriever.retrieve("help", top_k=1)
    assert len(results) <= 1
