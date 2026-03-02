"""
app.py – Intelligent Job Recommendation System
Flask backend orchestrating all ML modules.
"""

import os
import uuid
import json
import time
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash,
)
from werkzeug.utils import secure_filename

# ── Modules ───────────────────────────────────────────────────────────────────
from modules.resume_parser import ResumeParser
from modules.feature_engineering import FeatureEngineer
from modules.content_based import ContentBasedFilter
from modules.collaborative import CollaborativeFilter
from modules.hybrid import HybridRecommender
from modules.ranking import RankingModel
from modules.adaptive_learning import AdaptiveLearner
from modules.skill_gap import SkillGapAnalyzer

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# ── Load data & initialise models ─────────────────────────────────────────────
jobs_df = pd.read_csv("data/jobs.csv")
interactions_df = pd.read_csv("data/interactions.csv")

resume_parser = ResumeParser()
feature_eng = FeatureEngineer(model_dir="models")
cb_filter = ContentBasedFilter()
cf_filter = CollaborativeFilter(interactions_df)
hybrid = HybridRecommender()
ranker = RankingModel(model_dir="models")
adaptive = AdaptiveLearner()
gap_analyzer = SkillGapAnalyzer()

# Fit TF-IDF on job corpus
job_corpus = (jobs_df["description"].fillna("") + " " + jobs_df["skills_required"].fillna("")).tolist()
feature_eng.fit(job_corpus)
job_vectors = feature_eng.build_all_job_vectors(jobs_df)

# Train ranking model
ranker.train_from_interactions(jobs_df, interactions_df)

# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_id():
    if "user_id" not in session:
        session["user_id"] = "user_" + str(uuid.uuid4())[:8]
    return session["user_id"]


def run_pipeline(profile: dict, user_id: str, top_n: int = 10):
    """Full recommendation pipeline for a given candidate profile."""

    # 1. Feature vector for candidate
    candidate_vec = feature_eng.build_candidate_vector(profile)

    # 2. Skill overlap per job
    overlap_scores_list = [
        feature_eng.skill_overlap_score(profile["skills"], str(row["skills_required"]))
        for _, row in jobs_df.iterrows()
    ]
    overlap_dict = {row["job_id"]: overlap_scores_list[i] for i, (_, row) in enumerate(jobs_df.iterrows())}

    # 3. Content-based recommendations
    cb_results = cb_filter.recommend(
        candidate_vec, job_vectors, jobs_df,
        skill_overlap_scores=overlap_scores_list, top_n=len(jobs_df)
    )
    cb_results = cb_filter.experience_filter(cb_results, profile.get("experience_years", 0))

    # 4. Collaborative filtering scores
    cf_scores = cf_filter.get_scores(
        user_id, profile["skills"], list(jobs_df["job_id"])
    )

    # 5. Adaptive learning boost
    pref_boost = adaptive.preference_boost(user_id, jobs_df)
    for jid in cf_scores:
        cf_scores[jid] = cf_scores[jid] + pref_boost.get(jid, 0)

    # 6. Hybrid combination
    hybrid_results = hybrid.combine(cb_results, cf_scores, overlap_dict, top_n=top_n * 2)

    # 7. Re-rank
    final_results = ranker.rerank(hybrid_results).head(top_n)

    # Prepare output
    output = []
    for _, row in final_results.iterrows():
        required_skills = [s.strip() for s in str(row.get("skills_required", "")).split(",")]
        matched = [s for s in required_skills if s.lower() in [c.lower() for c in profile["skills"]]]
        output.append({
            "job_id": int(row["job_id"]),
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "category": row["category"],
            "salary_range": row["salary_range"],
            "experience_years": row["experience_years"],
            "description": row["description"][:200] + "...",
            "skills_required": required_skills,
            "matched_skills": matched,
            "missing_skills": [s for s in required_skills if s not in matched],
            "match_percent": round(len(matched) / max(len(required_skills), 1) * 100),
            "hybrid_score": round(float(row.get("hybrid_score", row.get("cb_score", 0))), 3),
        })
    return output


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    user_id = get_user_id()
    top_skills = adaptive.get_top_skills(user_id, 6)
    interaction_count = adaptive.get_interaction_count(user_id)
    return render_template(
        "index.html",
        user_id=user_id,
        top_skills=top_skills,
        interaction_count=interaction_count,
        total_jobs=len(jobs_df),
    )


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    user_id = get_user_id()

    if request.method == "POST":
        profile = {}
        source = request.form.get("source", "text")

        if source == "file" and "resume" in request.files:
            f = request.files["resume"]
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                f.save(path)
                profile = resume_parser.parse_file(path)
            else:
                flash("Invalid file type. Please upload PDF, DOCX, or TXT.", "danger")
                return redirect(url_for("recommend"))

        elif source == "text":
            resume_text = request.form.get("resume_text", "")
            if not resume_text.strip():
                flash("Please paste your resume text or upload a file.", "warning")
                return redirect(url_for("recommend"))
            profile = resume_parser.parse_text(resume_text)

        elif source == "manual":
            skills_raw = request.form.get("skills", "")
            exp = int(request.form.get("experience", 0) or 0)
            profile = {
                "raw_text": skills_raw,
                "skills": [s.strip().lower() for s in skills_raw.split(",") if s.strip()],
                "experience_years": exp,
                "education": [],
                "name": request.form.get("name", "Candidate"),
                "summary": "",
                "email": "",
                "phone": "",
            }

        if not profile.get("skills"):
            flash("No skills detected. Please add skills to improve recommendations.", "info")

        session["profile"] = profile
        top_n = int(request.form.get("top_n", 10))
        recommendations = run_pipeline(profile, user_id, top_n=top_n)
        session["last_recommendations"] = [r["job_id"] for r in recommendations]

        return render_template(
            "results.html",
            profile=profile,
            recommendations=recommendations,
            user_id=user_id,
        )

    return render_template("recommend.html", user_id=user_id)


@app.route("/interact", methods=["POST"])
def interact():
    """Record a user interaction (click / save / apply)."""
    user_id = get_user_id()
    data = request.get_json()
    job_id = int(data.get("job_id"))
    itype = data.get("type", "click")

    job_row = jobs_df[jobs_df["job_id"] == job_id]
    if not job_row.empty:
        job_meta = job_row.iloc[0].to_dict()
        adaptive.record(user_id, job_id, itype, job_meta)

        global interactions_df
        interactions_df = cf_filter.record_interaction(
            interactions_df, user_id, job_id, itype
        )

    return jsonify({"status": "ok", "message": f"Interaction '{itype}' recorded."})


@app.route("/skill-gap")
def skill_gap():
    user_id = get_user_id()
    profile = session.get("profile", {})
    target_ids = session.get("last_recommendations", [])

    if not profile:
        flash("Please get recommendations first.", "info")
        return redirect(url_for("recommend"))

    analysis = gap_analyzer.analyze(
        profile.get("skills", []),
        jobs_df,
        target_job_ids=target_ids[:5],
    )
    return render_template("skill_gap.html", analysis=analysis, profile=profile)


@app.route("/api/jobs")
def api_jobs():
    """REST API: return all jobs as JSON."""
    return jsonify(jobs_df.to_dict(orient="records"))


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """REST API: accept JSON profile, return recommendations."""
    user_id = get_user_id()
    data = request.get_json()
    profile = resume_parser.parse_text(data.get("resume_text", ""))
    if data.get("skills"):
        profile["skills"] = data["skills"]
    top_n = data.get("top_n", 10)
    results = run_pipeline(profile, user_id, top_n=top_n)
    return jsonify({"recommendations": results})


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run()
