"""
Module 4: Collaborative Filtering
Uses a user-job interaction matrix and KNN / cosine similarity
to find jobs liked by similar users.
Handles the cold-start problem gracefully.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilter:
    """
    Memory-based collaborative filtering.
    Builds a user-item matrix from recorded interactions,
    then finds similar users to generate score boosts.
    """

    def __init__(self, interactions_df: pd.DataFrame = None):
        self.user_item_matrix = None
        self.user_ids = []
        self.job_ids = []
        if interactions_df is not None:
            self.fit(interactions_df)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, interactions_df: pd.DataFrame):
        """
        Build user-item matrix from interactions DataFrame with columns:
        [user_id, job_id, score]
        """
        pivot = interactions_df.pivot_table(
            index="user_id", columns="job_id", values="score", fill_value=0
        )
        self.user_item_matrix = pivot.values.astype(float)
        self.user_ids = list(pivot.index)
        self.job_ids = list(pivot.columns)

    # ── Recommend ─────────────────────────────────────────────────────────────

    def get_scores(
        self,
        current_user_id: str,
        candidate_skills: list,
        all_job_ids: list,
        top_k_users: int = 5,
    ) -> dict:
        """
        Return a dict {job_id: cf_score} for all jobs.
        If user is new (cold-start), return zero scores.
        """
        if self.user_item_matrix is None or current_user_id not in self.user_ids:
            # Cold start → return zeros
            return {jid: 0.0 for jid in all_job_ids}

        user_idx = self.user_ids.index(current_user_id)
        user_vec = self.user_item_matrix[user_idx].reshape(1, -1)

        # Similarity with all other users
        sims = cosine_similarity(user_vec, self.user_item_matrix).flatten()
        sims[user_idx] = -1  # exclude self

        # Top-K similar users
        top_k_idx = np.argsort(sims)[-top_k_users:][::-1]
        top_k_sims = sims[top_k_idx]

        # Weighted average of their scores
        weighted_scores = {}
        for jid in all_job_ids:
            if jid in self.job_ids:
                j_col = self.job_ids.index(jid)
                scores = self.user_item_matrix[top_k_idx, j_col]
                w_sum = np.dot(top_k_sims, scores)
                sim_sum = np.sum(np.abs(top_k_sims)) + 1e-9
                weighted_scores[jid] = float(w_sum / sim_sum)
            else:
                weighted_scores[jid] = 0.0

        # Normalise to [0, 1]
        max_score = max(weighted_scores.values()) if weighted_scores else 1.0
        if max_score > 0:
            weighted_scores = {k: v / max_score for k, v in weighted_scores.items()}

        return weighted_scores

    def record_interaction(
        self,
        interactions_df: pd.DataFrame,
        user_id: str,
        job_id: int,
        interaction_type: str,
    ) -> pd.DataFrame:
        """Append a new interaction and refit the model."""
        score_map = {"click": 1, "save": 2, "apply": 3}
        score = score_map.get(interaction_type, 1)
        new_row = pd.DataFrame(
            [{"user_id": user_id, "job_id": job_id, "interaction_type": interaction_type, "score": score}]
        )
        interactions_df = pd.concat([interactions_df, new_row], ignore_index=True)
        self.fit(interactions_df)
        return interactions_df
