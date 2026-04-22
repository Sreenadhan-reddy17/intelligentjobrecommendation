"""
Module 2: Feature Engineering
Converts text profiles and job descriptions into numeric feature vectors
using Sentence Transformers context embeddings and skill-overlap features.
"""

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import torch

# Strictly limit PyTorch threads to prevent memory spikes on free tier hosting
torch.set_num_threads(1)


class FeatureEngineer:
    """
    Uses Sentence Transformers to encode text into dense semantic vectors.
    """

    def __init__(self, model_dir: str = "models", model_name="all-MiniLM-L6-v2"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        # Load pre-trained SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.scaler = MinMaxScaler()
        self._fitted = True # Sentence Transformers are pre-trained

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, job_descriptions: list):
        """Fit is a no-op for pre-trained sentence transformer."""
        pass # Already pre-trained

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform_text(self, text: str) -> np.ndarray:
        """Return embedding vector for a single document."""
        return self.model.encode(text)

    def transform_batch(self, texts: list) -> np.ndarray:
        """Return embedding matrix for a list of documents."""
        # Using batch encoding for speed
        return self.model.encode(texts)

    def build_candidate_vector(self, profile: dict) -> np.ndarray:
        """
        Combine resume text + explicit skill list into one dense vector.
        """
        combined = profile.get("raw_text", "")
        skills = " ".join(profile.get("skills", []))
        boosted = f"Candidate Profile: {combined}. Key skills: {skills}"
        return self.transform_text(boosted)

    def build_job_vector(self, job_row: pd.Series) -> np.ndarray:
        """Combine job description + skills_required into one vector."""
        text = str(job_row.get("description", "")) + " " + str(job_row.get("skills_required", ""))
        return self.transform_text(f"Job Description: {text}")

    def build_all_job_vectors(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Return matrix of shape (n_jobs, n_features)."""
        texts = [
            f"Job Description: {row['description']} ({row.get('skills_required', '')})" 
            if pd.notna(row.get('description')) else ""
            for _, row in jobs_df.iterrows()
        ]
        return self.transform_batch(texts)

    def skill_overlap_score(self, candidate_skills: list, job_skills_str: str) -> float:
        """
        Simple Jaccard overlap between candidate skill set and job's required skills.
        Returns a float in [0, 1].
        """
        if pd.isna(job_skills_str) or not isinstance(job_skills_str, str):
            job_skills_str = ""
        job_skills = [s.strip().lower() for s in job_skills_str.split(",") if s.strip()]
        c_set = set(s.lower() for s in candidate_skills)
        j_set = set(job_skills)
        if not j_set:
            return 0.0
        return len(c_set & j_set) / len(j_set)
