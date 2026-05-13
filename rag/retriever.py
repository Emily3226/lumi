"""
rag/retriever.py
RAG component — embeds mentor profiles using sentence-transformers
(a HuggingFace model, runs fully locally, no API needed)
and retrieves the most relevant ones for a given mentee query.

This uses YOUR PyTorch/transformers knowledge directly —
sentence-transformers is built on top of the same transformer
architecture from CS230.
"""

import json
import numpy as np
import os
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# This model runs 100% locally — no API key needed
MODEL_NAME = "all-MiniLM-L6-v2"


class MentorRetriever:
    def __init__(self, csv_path: str = "data/pairings.csv"):
        print("Loading embedding model (first run downloads ~90MB)...")
        self.model = SentenceTransformer(MODEL_NAME)
        # Try to load mentors from the SQLite DB (only available mentors)
        self.mentors = self._load_mentors_from_db() or self._load_mentors_from_csv(csv_path)
        self.index = self._build_index()
        print(f"RAG index built — {len(self.mentors)} mentor profiles indexed")

    def _connect_db(self):
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lumi.db")
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
            profile_text = f"{name} is a grade {r['grade']} student who wants to teach {r['subject']}. Qualifications: {r['qualifications']}."
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
        texts = [m['profile_text'] for m in self.mentors]
        return self.model.encode(texts, show_progress_bar=False)

    def retrieve(self, mentee_subject: str, mentee_grade: int, top_k: int = 3) -> list[dict]:
        query = f"Looking for a mentor to teach {mentee_subject} for a grade {mentee_grade} student."
        query_vec = self.model.encode([query])
        similarities = cosine_similarity(query_vec, self.index)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            mentor = self.mentors[i].copy()
            mentor['similarity_score'] = float(similarities[i])
            results.append(mentor)
        return results
