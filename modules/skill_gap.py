"""
Module 8 (Future Enhancement): Skill Gap Analysis
Identifies missing skills between a candidate's profile and target jobs,
and suggests learning resources to bridge those gaps.
"""

import pandas as pd


# Curated learning resource map (expandable)
LEARNING_RESOURCES = {
    "python": {"platform": "Coursera", "url": "https://www.coursera.org/learn/python", "duration": "1 month"},
    "machine learning": {"platform": "Andrew Ng - Coursera", "url": "https://www.coursera.org/learn/machine-learning", "duration": "3 months"},
    "deep learning": {"platform": "deeplearning.ai", "url": "https://www.deeplearning.ai/courses/", "duration": "2 months"},
    "nlp": {"platform": "Hugging Face", "url": "https://huggingface.co/learn/nlp-course/", "duration": "6 weeks"},
    "sql": {"platform": "Mode Analytics", "url": "https://mode.com/sql-tutorial/", "duration": "2 weeks"},
    "aws": {"platform": "AWS Training", "url": "https://aws.amazon.com/training/", "duration": "2 months"},
    "docker": {"platform": "Docker Docs", "url": "https://docs.docker.com/get-started/", "duration": "1 week"},
    "kubernetes": {"platform": "Linux Foundation", "url": "https://training.linuxfoundation.org/", "duration": "4 weeks"},
    "react": {"platform": "React Docs", "url": "https://react.dev/learn", "duration": "3 weeks"},
    "tensorflow": {"platform": "TensorFlow", "url": "https://www.tensorflow.org/learn", "duration": "4 weeks"},
    "pytorch": {"platform": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "duration": "4 weeks"},
    "bert": {"platform": "Hugging Face", "url": "https://huggingface.co/docs/transformers/", "duration": "3 weeks"},
    "java": {"platform": "Codecademy", "url": "https://www.codecademy.com/learn/learn-java", "duration": "5 weeks"},
    "spring boot": {"platform": "Spring.io", "url": "https://spring.io/guides", "duration": "4 weeks"},
    "git": {"platform": "GitHub Skills", "url": "https://skills.github.com/", "duration": "1 week"},
    "statistics": {"platform": "Khan Academy", "url": "https://www.khanacademy.org/math/statistics-probability", "duration": "6 weeks"},
    "data structures": {"platform": "GeeksforGeeks", "url": "https://www.geeksforgeeks.org/data-structures/", "duration": "4 weeks"},
    "algorithms": {"platform": "LeetCode", "url": "https://leetcode.com/", "duration": "ongoing"},
    "power bi": {"platform": "Microsoft Learn", "url": "https://learn.microsoft.com/en-us/power-bi/", "duration": "2 weeks"},
    "tableau": {"platform": "Tableau Public", "url": "https://public.tableau.com/en-us/s/resources", "duration": "2 weeks"},
}


class SkillGapAnalyzer:
    """
    Compares candidate skills against job requirements
    and returns structured gap analysis with learning suggestions.
    """

    def analyze(
        self,
        candidate_skills: list,
        jobs_df: pd.DataFrame,
        target_job_ids: list = None,
        top_n_jobs: int = 5,
    ) -> dict:
        """
        Parameters
        ----------
        candidate_skills : list of skills extracted from resume
        jobs_df          : full jobs DataFrame
        target_job_ids   : specific jobs to analyse (None = top N by index)
        top_n_jobs       : how many jobs to include in gap report

        Returns
        -------
        dict with per-job gap details and aggregated missing skills.
        """
        candidate_set = set(s.strip().lower() for s in candidate_skills)

        if target_job_ids:
            target_jobs = jobs_df[jobs_df["job_id"].isin(target_job_ids)].head(top_n_jobs)
        else:
            target_jobs = jobs_df.head(top_n_jobs)

        job_gaps = []
        all_missing = {}

        for _, job in target_jobs.iterrows():
            required = [s.strip().lower() for s in str(job.get("skills_required", "")).split(",")]
            required_set = set(s for s in required if s)

            matched = candidate_set & required_set
            missing = required_set - candidate_set

            match_pct = len(matched) / max(len(required_set), 1) * 100

            gap_detail = {
                "job_id": job["job_id"],
                "title": job["title"],
                "company": job["company"],
                "match_percent": round(match_pct, 1),
                "matched_skills": sorted(matched),
                "missing_skills": sorted(missing),
                "learning_resources": self._get_resources(missing),
            }
            job_gaps.append(gap_detail)

            for skill in missing:
                all_missing[skill] = all_missing.get(skill, 0) + 1

        # Priority missing skills (most frequently required across target jobs)
        priority_gaps = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)

        return {
            "candidate_skills": sorted(candidate_set),
            "job_gaps": job_gaps,
            "priority_missing_skills": priority_gaps,
            "top_learning_resources": self._get_resources([s for s, _ in priority_gaps[:8]]),
            "overall_readiness": self._readiness_score(job_gaps),
        }

    def _get_resources(self, skills: list) -> list:
        resources = []
        for skill in skills:
            if skill in LEARNING_RESOURCES:
                res = LEARNING_RESOURCES[skill].copy()
                res["skill"] = skill
                resources.append(res)
        return resources

    def _readiness_score(self, job_gaps: list) -> str:
        if not job_gaps:
            return "Unknown"
        avg = sum(j["match_percent"] for j in job_gaps) / len(job_gaps)
        if avg >= 80:
            return "High (80%+)"
        elif avg >= 55:
            return "Medium (55–79%)"
        else:
            return f"Low ({avg:.0f}%)"
