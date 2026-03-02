"""
Module 3: Content-Based Filtering
Computes cosine similarity between a candidate's feature vector
and every job's feature vector to produce a ranked recommendation list.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    """
    Given a candidate vector and pre-computed job vectors,
    returns cosine similarity scores for all jobs.
    """

    def recommend(
        self,
        candidate_vec: np.ndarray,
        job_vectors: np.ndarray,
        jobs_df: pd.DataFrame,
        skill_overlap_scores: list = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        candidate_vec      : 1-D numpy array (TF-IDF vector of candidate)
        job_vectors        : 2-D numpy array  shape (n_jobs, n_features)
        jobs_df            : DataFrame with job metadata
        skill_overlap_scores: list of pre-computed Jaccard scores per job
        top_n              : number of results to return

        Returns
        -------
        DataFrame of top-N jobs with cb_score column.
        """
        # Cosine similarity: shape (1, n_jobs) → flatten
        cos_scores = cosine_similarity(
            candidate_vec.reshape(1, -1), job_vectors
        ).flatten()

        # Blend with skill overlap (if provided)
        if skill_overlap_scores is not None:
            overlap = np.array(skill_overlap_scores)
            cos_scores = 0.7 * cos_scores + 0.3 * overlap

        result = jobs_df.copy()
        result["cb_score"] = cos_scores
        result = result.sort_values("cb_score", ascending=False).head(top_n)
        return result.reset_index(drop=True)

    def experience_filter(
        self, jobs_df: pd.DataFrame, candidate_exp: int, tolerance: int = 2
    ) -> pd.DataFrame:
        """
        Soft-filter: prefer jobs where candidate_exp is within [required - tolerance, required + tolerance].
        Penalises jobs that are significantly out of range instead of hard removal.
        """

        def _penalty(row):
            req = row.get("experience_years", 0)
            if req is None or str(req).strip() == "":
                return 1.0
            try:
                diff = abs(candidate_exp - int(req))
                return max(0.0, 1.0 - diff * 0.1)
            except Exception:
                return 1.0

        jobs_df = jobs_df.copy()
        jobs_df["exp_penalty"] = jobs_df.apply(_penalty, axis=1)
        if "cb_score" in jobs_df.columns:
            jobs_df["cb_score"] = jobs_df["cb_score"] * jobs_df["exp_penalty"]
        return jobs_df.drop(columns=["exp_penalty"])
