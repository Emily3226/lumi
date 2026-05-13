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
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# This model runs 100% locally — no API key needed
# It's a small transformer (22M params) fine-tuned for semantic similarity
MODEL_NAME = "all-MiniLM-L6-v2"

class MentorRetriever:
    def __init__(self, csv_path: str = "data/pairings.csv"):
        print("Loading embedding model (first run downloads ~90MB)...")
        self.model   = SentenceTransformer(MODEL_NAME)
        self.df      = pd.read_csv(csv_path)
        self.mentors = self._build_mentor_profiles()
        self.index   = self._build_index()
        print(f"RAG index built — {len(self.mentors)} mentor profiles indexed")

    def _build_mentor_profiles(self) -> list[dict]:
        """
        Deduplicate mentors from pairings data and build text profiles.
        Each profile is a natural language description we embed.
        """
        seen   = set()
        mentors = []
        for _, row in self.df.iterrows():
            name = row["mentor_name"]
            if name in seen:
                continue
            seen.add(name)
            profile_text = (
                f"{name} is a grade {row['mentor_grade']} student "
                f"who wants to teach {row['mentor_subject']}. "
                f"Qualifications: {row['mentor_qualifications']}."
            )
            mentors.append({
                "name":           name,
                "grade":          int(row["mentor_grade"]),
                "subject":        row["mentor_subject"],
                "qualifications": row["mentor_qualifications"],
                "profile_text":   profile_text,
            })
        return mentors

    def _build_index(self) -> np.ndarray:
        """Embed all mentor profiles into a matrix. Shape: (n_mentors, embed_dim)"""
        texts = [m["profile_text"] for m in self.mentors]
        return self.model.encode(texts, show_progress_bar=False)

    def retrieve(self, mentee_subject: str, mentee_grade: int, top_k: int = 3) -> list[dict]:
        """
        Given a mentee's subject and grade, retrieve the top_k most
        semantically similar mentor profiles.
        """
        query = (
            f"Looking for a mentor to teach {mentee_subject} "
            f"for a grade {mentee_grade} student."
        )
        query_vec    = self.model.encode([query])
        similarities = cosine_similarity(query_vec, self.index)[0]

        top_indices = similarities.argsort()[::-1][:top_k]
        results     = []
        for i in top_indices:
            mentor = self.mentors[i].copy()
            mentor["similarity_score"] = float(similarities[i])
            results.append(mentor)

        return results
