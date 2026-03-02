"""
Module 2: Feature Engineering
Converts text profiles and job descriptions into numeric feature vectors
using TF-IDF vectorization and skill-overlap features.
"""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    """
    Fits a TF-IDF vectorizer on all job descriptions + resume text
    and provides helpers for building feature vectors.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )
        self.scaler = MinMaxScaler()
        self._fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, job_descriptions: list):
        """Fit the vectorizer on a corpus of job descriptions."""
        self.vectorizer.fit(job_descriptions)
        self._fitted = True
        self._save()

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform_text(self, text: str) -> np.ndarray:
        """Return TF-IDF vector for a single document."""
        self._ensure_fitted()
        return self.vectorizer.transform([text]).toarray()[0]

    def transform_batch(self, texts: list) -> np.ndarray:
        """Return TF-IDF matrix for a list of documents."""
        self._ensure_fitted()
        return self.vectorizer.transform(texts).toarray()

    def build_candidate_vector(self, profile: dict) -> np.ndarray:
        """
        Combine resume text + explicit skill list into one TF-IDF vector.
        Skill keywords are appended 3× for extra weight.
        """
        combined = profile.get("raw_text", "")
        skills = " ".join(profile.get("skills", []))
        boosted = combined + " " + (skills + " ") * 3
        return self.transform_text(boosted)

    def build_job_vector(self, job_row: pd.Series) -> np.ndarray:
        """Combine job description + skills_required into one TF-IDF vector."""
        text = str(job_row.get("description", "")) + " " + str(job_row.get("skills_required", ""))
        return self.transform_text(text)

    def build_all_job_vectors(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Return matrix of shape (n_jobs, n_features)."""
        texts = (
            jobs_df["description"].fillna("") + " " + jobs_df["skills_required"].fillna("")
        ).tolist()
        return self.transform_batch(texts)

    def skill_overlap_score(self, candidate_skills: list, job_skills_str: str) -> float:
        """
        Simple Jaccard overlap between candidate skill set and job's required skills.
        Returns a float in [0, 1].
        """
        job_skills = [s.strip().lower() for s in job_skills_str.split(",")]
        c_set = set(s.lower() for s in candidate_skills)
        j_set = set(job_skills)
        if not j_set:
            return 0.0
        return len(c_set & j_set) / len(j_set)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        path = os.path.join(self.model_dir, "vectorizer.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self._fitted = True

    def _ensure_fitted(self):
        if not self._fitted:
            self.load()
        if not self._fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit() first.")
