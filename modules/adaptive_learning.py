"""
Module 7: Adaptive Learning
Tracks user interactions (clicks, saves, applications, feedback ratings)
and continuously updates preference profiles to improve personalization.
"""

import json
import os
import time
from collections import defaultdict
from typing import Optional


class AdaptiveLearner:
    """
    Maintains a per-user preference profile that evolves with every interaction.
    Skills seen in applied / saved jobs receive higher preference weights.
    """

    def __init__(self, profile_dir: str = "data/user_profiles"):
        self.profile_dir = profile_dir
        os.makedirs(profile_dir, exist_ok=True)
        self._cache = {}  # in-memory cache

    # ── Profile management ────────────────────────────────────────────────────

    def load_profile(self, user_id: str) -> dict:
        """Load or initialise a user preference profile."""
        if user_id in self._cache:
            return self._cache[user_id]
        path = self._profile_path(user_id)
        if os.path.exists(path):
            with open(path) as f:
                profile = json.load(f)
        else:
            profile = {
                "user_id": user_id,
                "skill_weights": {},
                "category_weights": {},
                "interactions": [],
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        self._cache[user_id] = profile
        return profile

    def save_profile(self, user_id: str, profile: dict):
        """Persist profile to disk."""
        profile["updated_at"] = time.time()
        with open(self._profile_path(user_id), "w") as f:
            json.dump(profile, f, indent=2)
        self._cache[user_id] = profile

    # ── Record interaction ─────────────────────────────────────────────────────

    def record(
        self,
        user_id: str,
        job_id: int,
        interaction_type: str,       # 'click' | 'save' | 'apply' | 'feedback'
        job_metadata: dict,          # {'skills_required': '...', 'category': '...'}
        feedback_score: Optional[int] = None,  # 1–5 star rating
    ):
        """
        Update user preference profile based on new interaction.
        Weights: apply=3, save=2, feedback*0.6, click=0.5
        """
        weight_map = {"click": 0.5, "save": 2.0, "apply": 3.0, "feedback": 0.0}
        base_weight = weight_map.get(interaction_type, 0.5)
        if interaction_type == "feedback" and feedback_score is not None:
            base_weight = feedback_score * 0.6  # scale 1-5 → 0.6-3.0

        profile = self.load_profile(user_id)

        # Update skill weights
        skills = [s.strip().lower() for s in str(job_metadata.get("skills_required", "")).split(",")]
        for skill in skills:
            if skill:
                profile["skill_weights"][skill] = (
                    profile["skill_weights"].get(skill, 0) + base_weight
                )

        # Update category weights
        category = str(job_metadata.get("category", "")).strip()
        if category:
            profile["category_weights"][category] = (
                profile["category_weights"].get(category, 0) + base_weight
            )

        # Log interaction
        profile["interactions"].append({
            "job_id": job_id,
            "type": interaction_type,
            "weight": base_weight,
            "timestamp": time.time(),
        })

        self.save_profile(user_id, profile)

    # ── Preference-boosted scoring ─────────────────────────────────────────────

    def preference_boost(self, user_id: str, jobs_df, max_boost: float = 0.15) -> dict:
        """
        Return {job_id: boost_score} based on user's skill & category preferences.
        Boost is capped at max_boost to avoid over-personalisation.
        """
        profile = self.load_profile(user_id)
        skill_weights = profile.get("skill_weights", {})
        category_weights = profile.get("category_weights", {})

        boosts = {}
        for _, row in jobs_df.iterrows():
            score = 0.0
            job_skills = [s.strip().lower() for s in str(row.get("skills_required", "")).split(",")]
            for skill in job_skills:
                score += skill_weights.get(skill, 0)

            category = str(row.get("category", "")).strip()
            score += category_weights.get(category, 0) * 0.5

            # Normalize (cap at max_boost)
            boosts[row["job_id"]] = min(score / (sum(skill_weights.values()) + 1e-9), max_boost)

        return boosts

    def get_top_skills(self, user_id: str, top_n: int = 10) -> list:
        """Return top-N preferred skills for display."""
        profile = self.load_profile(user_id)
        sw = profile.get("skill_weights", {})
        return sorted(sw.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def get_interaction_count(self, user_id: str) -> int:
        profile = self.load_profile(user_id)
        return len(profile.get("interactions", []))

    def _profile_path(self, user_id: str) -> str:
        safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
        return os.path.join(self.profile_dir, f"{safe_id}.json")
