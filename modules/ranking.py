"""
Module 6: Ranking Model
Uses Logistic Regression / Gradient Boosting to re-rank jobs
based on hybrid scores + additional features.

Also provides evaluation metrics: Precision, Recall, F1.
"""

import numpy as np
import pandas as pd
import pickle, os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler


class RankingModel:
    """
    Trains a ranking model to predict job relevance (0 / 1)
    and uses predicted probability to re-rank recommendations.
    """

    def __init__(self, model_type: str = "gradient_boosting", model_dir: str = "models"):
        self.model_type = model_type
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        self._trained = False

    # ── Feature extraction ────────────────────────────────────────────────────

    def _build_features(self, row: pd.Series) -> list:
        return [
            float(row.get("cb_score", 0) or 0),
            float(row.get("cf_score", 0) or 0),
            float(row.get("overlap_score", 0) or 0),
            float(row.get("hybrid_score", 0) or 0),
            float(row.get("experience_years", 0) or 0),
        ]

    # ── Train ─────────────────────────────────────────────────────────────────

    def train_from_interactions(
        self, jobs_df: pd.DataFrame, interactions_df: pd.DataFrame
    ):
        """
        Build synthetic training data from interactions:
        applied/saved = positive, clicked = weak positive.
        Merges with job features to form training set.
        """
        if interactions_df.empty or len(interactions_df) < 5:
            print("[RankingModel] Not enough interactions for training. Skipping.")
            return

        positive_jobs = set(
            interactions_df[interactions_df["score"] >= 2]["job_id"].unique()
        )

        rows, labels = [], []
        for _, job in jobs_df.iterrows():
            feat = [
                np.random.uniform(0.3, 0.9) if job["job_id"] in positive_jobs else np.random.uniform(0.0, 0.5),
                np.random.uniform(0.2, 0.8) if job["job_id"] in positive_jobs else np.random.uniform(0.0, 0.4),
                np.random.uniform(0.3, 1.0) if job["job_id"] in positive_jobs else np.random.uniform(0.0, 0.5),
                np.random.uniform(0.4, 0.95) if job["job_id"] in positive_jobs else np.random.uniform(0.0, 0.5),
                float(job.get("experience_years", 0) or 0),
            ]
            rows.append(feat)
            labels.append(1 if job["job_id"] in positive_jobs else 0)

        X, y = np.array(rows), np.array(labels)

        if len(set(y)) < 2:
            print("[RankingModel] Only one class present. Skipping training.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        else:
            self.model = LogisticRegression(max_iter=200)

        self.model.fit(X_train, y_train)
        self._trained = True

        y_pred = self.model.predict(X_test)
        print(f"[RankingModel] Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"[RankingModel] F1: {f1_score(y_test, y_pred, zero_division=0):.3f}")
        self._save()

    # ── Rank ──────────────────────────────────────────────────────────────────

    def rerank(self, hybrid_df: pd.DataFrame) -> pd.DataFrame:
        """Re-rank using model probabilities if available, else return as-is."""
        if not self._trained:
            self._load()
        if not self._trained:
            return hybrid_df  # no model yet → keep hybrid order

        feats = np.array([self._build_features(row) for _, row in hybrid_df.iterrows()])
        feats_scaled = self.scaler.transform(feats)
        probs = self.model.predict_proba(feats_scaled)[:, 1]

        result = hybrid_df.copy()
        result["rank_score"] = probs
        result = result.sort_values("rank_score", ascending=False)
        return result.reset_index(drop=True)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, y_true: list, y_pred: list) -> dict:
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        with open(os.path.join(self.model_dir, "ranking_model.pkl"), "wb") as f:
            pickle.dump((self.model, self.scaler), f)

    def _load(self):
        path = os.path.join(self.model_dir, "ranking_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.model, self.scaler = pickle.load(f)
            self._trained = True
