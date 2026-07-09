"""
rag/retriever.py
RAG component — embeds mentor profiles using sentence-transformers
(a HuggingFace model, runs fully locally, no API needed)
and retrieves the most relevant ones for a given mentee query.

This uses YOUR PyTorch/transformers knowledge directly —
sentence-transformers is built on top of the same transformer
architecture from CS230.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from api.db import DATABASE_URL, get_db
from rag.embeddings import get_embedding_function
from rag.subject_utils import SUBJECT_ALIASES, expand_query_text, subject_key

# This model runs 100% locally — no API key needed
MODEL_NAME = "all-MiniLM-L6-v2"


class MentorRetriever:
    def __init__(self, csv_path: str = "data/pairings.csv"):
        print("Loading embedding model (cached locally after the first download - see rag/embeddings.py)...")
        self.model = get_embedding_function()
        # Try to load mentors from the SQLite DB (only available mentors)
        self.mentors = self._load_mentors_from_db() or self._load_mentors_from_csv(csv_path)
        self.index = self._build_index()
        print(f"RAG index built - {len(self.mentors)} mentor profiles indexed")

    def _connect_db(self):
        if not DATABASE_URL:
            return None
        try:
            return get_db()
        except Exception:
            return None

    def _alias_text(self, subject: str | None) -> str:
        key = subject_key(subject)
        if not key:
            return (subject or "").strip()
        aliases = ", ".join((key, *SUBJECT_ALIASES[key]))
        return f"{key}. Related topics: {aliases}."

    def _load_mentors_from_db(self) -> list[dict] | None:
        conn = self._connect_db()
        if not conn:
            return None
        cur = conn.execute("SELECT name, grade, qualifications, subject FROM mentors WHERE available = 1")
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return None
        mentors = []
        for r in rows:
            name = r['name']
            profile_text = (
                f"{name} is a grade {r['grade']} student who wants to teach {r['subject']}. "
                f"{self._alias_text(r['subject'])} "
                f"Qualifications: {r['qualifications']}."
            )
            mentors.append({
                'name': name,
                'grade': int(r['grade']) if r['grade'] is not None else 0,
                'subject': r['subject'],
                'qualifications': r['qualifications'],
                'profile_text': profile_text,
            })
        return mentors

    def _load_mentors_from_csv(self, csv_path: str) -> list[dict]:
        import pandas as pd
        df = pd.read_csv(csv_path)
        seen = set()
        mentors = []
        for _, row in df.iterrows():
            name = row['mentor_name']
            if name in seen:
                continue
            seen.add(name)
            profile_text = (
                f"{name} is a grade {row['mentor_grade']} student "
                f"who wants to teach {row['mentor_subject']}. "
                f"{self._alias_text(row['mentor_subject'])} "
                f"Qualifications: {row['mentor_qualifications']}."
            )
            mentors.append({
                'name': name,
                'grade': int(row['mentor_grade']),
                'subject': row['mentor_subject'],
                'qualifications': row['mentor_qualifications'],
                'profile_text': profile_text,
            })
        return mentors

    def _build_index(self) -> np.ndarray:
        if not self.mentors:
            return np.empty((0, 0))
        texts = [m['profile_text'] for m in self.mentors]
        return np.array(self.model(texts))


    def retrieve(self, query_text: str, mentee_grade: int | None = None, top_k: int = 3) -> list[dict]:
        if not self.mentors:
            return []

        query = expand_query_text(query_text)
        if mentee_grade is not None:
            query = f"{query}\nThe mentee is in grade {mentee_grade}."

        query_vec = np.array(self.model([query]))

        similarities = cosine_similarity(query_vec, self.index)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            mentor = self.mentors[i].copy()
            mentor['similarity_score'] = float(similarities[i])
            results.append(mentor)
        return results