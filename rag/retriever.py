"""
rag/retriever.py
RAG component — embeds mentor profiles using a local ONNX embedding model
(all-MiniLM-L6-v2, via rag/embeddings.py) and retrieves the most relevant
ones for a given mentee query. Runs fully locally, no API needed.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from api.db import get_db
from rag.embeddings import get_embedding_function

from rag.subject_utils import SUBJECT_ALIASES, expand_query_text, subject_key

logger = logging.getLogger(__name__)

# This model runs 100% locally — no API key needed
MODEL_NAME = "all-MiniLM-L6-v2"


class MentorRetriever:
    def __init__(self, csv_path: str = "data/pairings.csv"):
        print("Loading embedding model (first run downloads ~90MB)...")
        self.model = get_embedding_function()
        # Mentors live in MongoDB Atlas (see scripts/migrate_to_mongo.py); the
        # CSV is only a last-resort fallback for local/offline dev.
        self.mentors = self._load_mentors_from_mongo() or self._load_mentors_from_csv(csv_path)
        self.index = self._build_index()
        print(f"RAG index built - {len(self.mentors)} mentor profiles indexed")

    def _alias_text(self, subject: str | None) -> str:
        key = subject_key(subject)
        if not key:
            return (subject or "").strip()
        aliases = ", ".join((key, *SUBJECT_ALIASES[key]))
        return f"{key}. Related topics: {aliases}."

    def _load_mentors_from_mongo(self) -> list[dict] | None:
        try:
            rows = list(get_db()["mentors"].find({"available": {"$in": [1, True]}}))
        except Exception:
            logger.warning("Could not load mentors from MongoDB", exc_info=True)
            return None
        if not rows:
            return None
        mentors = []
        for r in rows:
            name = r.get("name")
            grade = r.get("grade")
            subject = r.get("subject")
            qualifications = r.get("qualifications")
            profile_text = (
                f"{name} is a grade {grade} student who wants to teach {subject}. "
                f"{self._alias_text(subject)} "
                f"Qualifications: {qualifications}."
            )
            mentors.append({
                'name': name,
                'grade': int(grade) if grade is not None else 0,
                'subject': subject,
                'qualifications': qualifications,
                'profile_text': profile_text,
            })
        return mentors

    def _load_mentors_from_csv(self, csv_path: str) -> list[dict]:
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
        except (FileNotFoundError, OSError):
            logger.warning("No mentor CSV fallback available at %s", csv_path)
            return []
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