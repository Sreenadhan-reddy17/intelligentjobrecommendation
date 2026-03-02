"""
Module 5: Hybrid Recommendation Model
Combines content-based (CB) and collaborative filtering (CF) scores
using configurable weighted blending.

Also supports future plug-in of Sentence-BERT embeddings.
"""

import numpy as np
import pandas as pd


class HybridRecommender:
    """
    Weighted linear combination:
        hybrid_score = α * cb_score + β * cf_score + γ * overlap_score
    Default weights: α=0.55, β=0.25, γ=0.20
    Weights are auto-adjusted when CF data is sparse (cold-start).
    """

    def __init__(self, alpha: float = 0.55, beta: float = 0.25, gamma: float = 0.20):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1."
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ── Main method ───────────────────────────────────────────────────────────

    def combine(
        self,
        cb_df: pd.DataFrame,         # output from ContentBasedFilter.recommend()
        cf_scores: dict,             # {job_id: cf_score} from CollaborativeFilter
        overlap_scores: dict = None, # {job_id: overlap_score}
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Merge CB and CF scores, return top-N ranked jobs.

        Parameters
        ----------
        cb_df          : DataFrame with 'job_id' and 'cb_score' columns
        cf_scores      : dict mapping job_id → cf collaborative score
        overlap_scores : dict mapping job_id → skill overlap score
        top_n          : number of results

        Returns
        -------
        DataFrame with 'hybrid_score' column, sorted descending.
        """
        result = cb_df.copy()

        # Attach CF score
        result["cf_score"] = result["job_id"].map(cf_scores).fillna(0.0)

        # Attach overlap score
        if overlap_scores:
            result["overlap_score"] = result["job_id"].map(overlap_scores).fillna(0.0)
        else:
            result["overlap_score"] = result.get("overlap_score", 0.0)

        # Detect cold-start: if all CF scores are zero, boost CB weight
        if result["cf_score"].sum() == 0:
            alpha, beta, gamma = 0.70, 0.00, 0.30
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma

        result["hybrid_score"] = (
            alpha * result["cb_score"].fillna(0)
            + beta * result["cf_score"].fillna(0)
            + gamma * result["overlap_score"].fillna(0)
        )

        result = result.sort_values("hybrid_score", ascending=False).head(top_n)
        return result.reset_index(drop=True)

    # ── Future: Sentence-BERT embeddings (enhancement) ────────────────────────

    def sbert_similarity(self, candidate_text: str, job_texts: list) -> np.ndarray:
        """
        FUTURE ENHANCEMENT: Use Sentence-BERT for semantic similarity.
        Install: pip install sentence-transformers
        Usage: Replaces or supplements TF-IDF cosine similarity.
        Returns ndarray of similarity scores.
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            c_emb = model.encode(candidate_text, convert_to_tensor=True)
            j_embs = model.encode(job_texts, convert_to_tensor=True)
            scores = util.cos_sim(c_emb, j_embs).squeeze().numpy()
            return scores
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
