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
import database
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
load_dotenv()

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

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', 'placeholder_client_id_must_change'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', 'placeholder_client_secret_must_change'),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

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

# Encode job corpus with Semantic Embeddings
job_corpus = (jobs_df["description"].fillna("") + " " + jobs_df["skills_required"].fillna("")).tolist()
feature_eng.fit(job_corpus)
job_vectors = feature_eng.build_all_job_vectors(jobs_df)

# Train ranking model
ranker.train_from_interactions(jobs_df, interactions_df)

# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_resume(profile):
    """Strict heuristic to check if the parsed profile actually belongs to an ATS resume/CV."""
    text = profile.get("raw_text", "").lower()
    if not text or len(text.strip()) < 100:
        return False
    
    has_email = bool(profile.get("email"))
    has_phone = bool(profile.get("phone"))
    has_skills = bool(profile.get("skills"))
    
    ats_sections = ["experience", "work history", "employment", "education", "skills", "summary", "objective"]
    section_count = sum(1 for sec in ats_sections if sec in text)
    
    # Must have contact info and at least 3 standard ATS resume sections
    if (has_email or has_phone) and section_count >= 3:
        return True
        
    # Highly likely to be a resume if it has both email and phone + skills
    if has_email and has_phone and has_skills:
        return True
        
    return False


def get_user_id():
    if "user" in session and "id" in session["user"]:
        return session["user"]["id"]
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

# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.route("/login")
def login():
    session["next"] = request.referrer or url_for("index")
    redirect_uri = url_for("auth", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/auth")
def auth():
    token = google.authorize_access_token()
    user_info = google.parse_id_token(token, nonce=None)
    if not user_info:
        resp = google.get("userinfo")
        user_info = resp.json()
    
    user_id = str(user_info.get("sub") or user_info.get("id"))
    email = user_info.get("email")
    name = user_info.get("name", "User")
    
    database.upsert_user(user_id, name, email, dict(user_info))
    session["user"] = {"id": user_id, "name": name, "email": email}
    session["user_id"] = user_id
    
    return redirect(session.get("next") or url_for("index"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("user_id", None)
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    user_id = session["user"]["id"]
    db_saved = database.get_saved_jobs(user_id)
    db_applied = database.get_applied_jobs(user_id)
    
    saved_jobs = []
    for s in db_saved:
        r = jobs_df[jobs_df["job_id"] == s["job_id"]]
        if not r.empty:
            j = r.iloc[0].to_dict()
            j["saved_at"] = s["saved_at"]
            saved_jobs.append(j)
            
    applied_jobs = []
    for s in db_applied:
        r = jobs_df[jobs_df["job_id"] == s["job_id"]]
        if not r.empty:
            j = r.iloc[0].to_dict()
            j["applied_at"] = s["applied_at"]
            applied_jobs.append(j)

    return render_template("dashboard.html", saved_jobs=saved_jobs, applied_jobs=applied_jobs, user=session["user"])

# ── Main Routes ───────────────────────────────────────────────────────────────

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
                
                if not is_valid_resume(profile):
                    flash("provided file is not a resume, please upload a resume to get Recommendations", "danger")
                    return redirect(url_for("recommend"))
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

    if itype in ["save", "apply"] and "user" not in session:
        return jsonify({"status": "error", "message": "Please login to save or apply.", "redirect": url_for("login")}), 401

    if itype == "save":
        database.save_job(user_id, job_id)
    elif itype == "apply":
        database.apply_job(user_id, job_id)

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


@app.route("/recruiter", methods=["GET", "POST"])
def recruiter():
    user_id = get_user_id()
    if request.method == "POST":
        job_id = request.form.get("job_id")
        custom_job = request.form.get("custom_job", "")
        top_n = int(request.form.get("top_n", 5))

        # Get job requirement text and skills
        job_req_text = ""
        job_skills = ""
        if job_id and job_id != "custom":
            row = jobs_df[jobs_df["job_id"] == int(job_id)]
            if not row.empty:
                job_req_text = f"Job Description: {row.iloc[0]['description']} {row.iloc[0]['skills_required']}"
                job_skills = str(row.iloc[0]["skills_required"])
        else:
            job_req_text = f"Job Description: {custom_job}"
            job_skills = custom_job

        job_vec = feature_eng.transform_text(job_req_text)

        candidates = []
        files = request.files.getlist("resumes")
        for f in files:
            if f and allowed_file(f.filename):
                from werkzeug.utils import secure_filename
                filename = secure_filename(f.filename)
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                f.save(path)
                profile = resume_parser.parse_file(path)

                if not is_valid_resume(profile):
                    candidates.append({
                        "name": "Invalid File",
                        "filename": filename,
                        "is_valid": False,
                        "score": -1
                    })
                    continue

                cand_vec = feature_eng.build_candidate_vector(profile)
                
                from sklearn.metrics.pairwise import cosine_similarity
                cos_score = float(cosine_similarity(cand_vec.reshape(1, -1), job_vec.reshape(1, -1))[0][0])
                overlap = float(feature_eng.skill_overlap_score(profile.get("skills", []), job_skills))

                final_score = 0.7 * cos_score + 0.3 * overlap

                candidates.append({
                    "name": profile.get("name", "Candidate"),
                    "email": profile.get("email", "Not provided"),
                    "phone": profile.get("phone", "Not provided"),
                    "skills": profile.get("skills", []),
                    "experience": profile.get("experience_years", 0),
                    "score": round(final_score * 100, 1),
                    "filename": filename,
                    "is_valid": True
                })

        valid_cands = sorted([c for c in candidates if c.get("is_valid")], key=lambda x: x["score"], reverse=True)[:top_n]
        invalid_cands = [c for c in candidates if not c.get("is_valid")]
        candidates = valid_cands + invalid_cands
        return render_template("recruiter.html", candidates=candidates, jobs=jobs_df.to_dict('records'), user_id=user_id)

    return render_template("recruiter.html", jobs=jobs_df.to_dict('records'), user_id=user_id)


if __name__ == "__main__":
    app.run(debug=True)
