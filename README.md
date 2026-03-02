# SmartJob AI — Intelligent Job Recommendation System

> Full-stack Machine Learning project based on SP301 Project Report.  
> **7 active ML modules + Skill Gap Analysis future enhancement.**

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Local Setup (Step-by-Step)](#5-local-setup-step-by-step)
6. [Running the App](#6-running-the-app)
7. [Using the Application](#7-using-the-application)
8. [REST API Reference](#8-rest-api-reference)
9. [Docker Deployment](#9-docker-deployment)
10. [Cloud Deployment (Render / Heroku)](#10-cloud-deployment)
11. [Troubleshooting](#11-troubleshooting)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. Project Overview

SmartJob AI is a hybrid intelligent job recommendation system that:

| Module | Technique | Purpose |
|--------|-----------|---------|
| Resume Parser | spaCy / regex / pdfplumber | Extract skills, experience, education from PDF/DOCX |
| Feature Engineering | TF-IDF (500 features, bigrams) | Convert text to numeric vectors |
| Content-Based Filtering | Cosine Similarity + Jaccard | Match candidate vector to job vectors |
| Collaborative Filtering | KNN, User-Item Matrix | Leverage patterns from similar users |
| Hybrid Recommender | Weighted Blend (α·CB + β·CF + γ·Overlap) | Combine both approaches |
| Ranking Model | Gradient Boosting / Logistic Regression | Re-rank by predicted relevance |
| Adaptive Learning | Interaction tracking, profile updates | Improve recommendations over time |
| Skill Gap Analyzer | Gap analysis + Resource map | (Future Enhancement — already built) |

---

## 2. Architecture

```
User Input (Resume / Skills)
        │
        ▼
  NLP Resume Parser
        │
        ▼
  TF-IDF Feature Vectors
        │
   ┌────┴────┐
   ▼         ▼
Content   Collaborative
Based     Filtering
Filter    (KNN Matrix)
   │         │
   └────┬────┘
        ▼
  Hybrid Blend
  (α·CB + β·CF + γ·Overlap)
        │
        ▼
  Gradient Boosting Re-ranker
        │
        ▼
  Top-N Job Recommendations
        │
        ▼
  Feedback Loop → Adaptive Learning Profile
```

---

## 3. Project Structure

```
job_recommender/
├── app.py                    ← Flask application (main entry point)
├── requirements.txt          ← Python dependencies
├── Dockerfile                ← Docker containerization
├── Procfile                  ← Heroku/Render deployment
│
├── modules/                  ← All ML modules
│   ├── __init__.py
│   ├── resume_parser.py      ← Module 1: NLP resume parsing
│   ├── feature_engineering.py← Module 2: TF-IDF vectorization
│   ├── content_based.py      ← Module 3: Cosine similarity CB filter
│   ├── collaborative.py      ← Module 4: KNN collaborative filter
│   ├── hybrid.py             ← Module 5: Weighted hybrid blend
│   ├── ranking.py            ← Module 6: Gradient Boosting ranker
│   ├── adaptive_learning.py  ← Module 7: Interaction-based learning
│   └── skill_gap.py          ← Module 8: Skill gap + resources (Future Enh.)
│
├── data/
│   ├── jobs.csv              ← Job listings dataset
│   ├── interactions.csv      ← User interaction history
│   └── user_profiles/        ← Per-user adaptive profiles (auto-created)
│
├── models/                   ← Saved ML models (auto-created)
│   ├── vectorizer.pkl        ← Fitted TF-IDF vectorizer
│   └── ranking_model.pkl     ← Trained Gradient Boosting model
│
├── uploads/                  ← Uploaded resumes (auto-created)
│
├── templates/                ← Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── recommend.html
│   ├── results.html
│   ├── skill_gap.html
│   └── about.html
│
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## 4. Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.9 – 3.12 | `python --version` |
| pip | 23+ | `pip --version` |
| Git | Any | `git --version` |
| Docker (optional) | 24+ | `docker --version` |

---

## 5. Local Setup (Step-by-Step)

### Step 1 — Clone or Download

```bash
# If using git:
git clone https://github.com/your-username/job_recommender.git
cd job_recommender

# OR simply unzip the downloaded folder and cd into it
cd job_recommender
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

> ✅ You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: Flask, Pandas, NumPy, scikit-learn, pdfplumber, python-docx, gunicorn.

> **If pdfplumber fails on Windows**, run:
> ```bash
> pip install pdfplumber --no-binary pdfminer.six
> ```

### Step 4 — Verify Installation

```bash
python -c "import flask, pandas, sklearn, pdfplumber; print('All OK')"
```

You should see: `All OK`

### Step 5 — Create Required Folders

These are auto-created on first run, but you can pre-create them:

```bash
mkdir -p uploads models data/user_profiles
```

---

## 6. Running the App

### Development Mode

```bash
python app.py
```

Open your browser: **http://localhost:5000**

### Production Mode (Gunicorn)

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --timeout 60
```

### Set a Secure Secret Key (Important for production)

```bash
# Linux / macOS
export SECRET_KEY="your-random-secret-key-here"

# Windows PowerShell
$env:SECRET_KEY = "your-random-secret-key-here"
```

---

## 7. Using the Application

### Getting Recommendations

1. Open **http://localhost:5000**
2. Click **"Get Recommendations"**
3. Choose one of three input methods:
   - **Paste Resume**: Copy-paste your entire CV text
   - **Upload File**: Upload PDF, DOCX, or TXT (max 5MB)
   - **Enter Skills**: Type comma-separated skills manually

4. Select number of recommendations (5, 10, or 15)
5. Click **"Analyze & Recommend"**

### Interacting with Results

- **Save** — Saves job to your profile (score +2)
- **Apply** — Marks as applied (score +3)
- **Hover** — Auto-records click interaction (score +1)

These interactions train the adaptive learning system in real time.

### Skill Gap Analysis

After getting recommendations, click **"Skill Gap"** in the navbar or the yellow CTA button.  
This shows:
- Matched vs. missing skills per job
- Your overall readiness score
- Priority skills to learn
- Curated learning resources (Coursera, Hugging Face, etc.)

---

## 8. REST API Reference

### Get All Jobs

```http
GET /api/jobs
```

Response: JSON array of all job listings.

### Get Recommendations via API

```http
POST /api/recommend
Content-Type: application/json

{
  "resume_text": "Python developer with 3 years experience in ML, TensorFlow, SQL...",
  "top_n": 10
}
```

Or pass explicit skills:
```json
{
  "skills": ["python", "machine learning", "tensorflow"],
  "top_n": 5
}
```

### Record Interaction

```http
POST /interact
Content-Type: application/json

{
  "job_id": 4,
  "type": "apply"
}
```

`type` options: `"click"`, `"save"`, `"apply"`

---

## 9. Docker Deployment

### Build Image

```bash
docker build -t smartjob-ai .
```

### Run Container

```bash
docker run -d \
  -p 5000:5000 \
  -e SECRET_KEY="your-secret-key" \
  --name smartjob \
  smartjob-ai
```

Open: **http://localhost:5000**

### Stop Container

```bash
docker stop smartjob
docker rm smartjob
```

---

## 10. Cloud Deployment

### Option A — Render (Free, Recommended)

1. Push your code to GitHub
2. Go to [https://render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
   - **Environment Variable**: `SECRET_KEY = your-secret-key`
5. Click **Deploy** — live in ~3 minutes ✅

### Option B — Heroku

```bash
heroku login
heroku create your-smartjob-app
git push heroku main
heroku config:set SECRET_KEY="your-secret-key"
heroku open
```

### Option C — AWS EC2 / DigitalOcean

```bash
# On your server:
sudo apt update && sudo apt install python3 python3-pip -y
git clone <your-repo>
cd job_recommender
pip3 install -r requirements.txt

# Run with Gunicorn
gunicorn app:app --bind 0.0.0.0:80 --workers 2 --daemon
```

For HTTPS, use **Nginx** as a reverse proxy:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 11. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'flask'`
**Fix**: Activate your virtual environment first:
```bash
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### ❌ `pdfplumber` fails / poppler error on Windows
**Fix**: Install Poppler for Windows from https://github.com/oschwartz10612/poppler-windows/releases  
Add to PATH. Then:
```bash
pip install pdfplumber
```

### ❌ `No such file: data/jobs.csv`
**Fix**: Ensure you're running `python app.py` from inside the `job_recommender/` folder:
```bash
cd job_recommender
python app.py
```

### ❌ Port 5000 already in use
**Fix**: Use a different port:
```bash
python app.py  # Edit app.py last line: port=5001
# or
gunicorn app:app --bind 0.0.0.0:8000
```

### ❌ `OSError: [Errno 13] Permission denied: 'models/'`
**Fix**:
```bash
chmod -R 755 models/ data/ uploads/
```

### ❌ Recommendations are all the same score
This is normal on first run (no interactions yet). After saving / applying to jobs, the adaptive learning system will differentiate scores. Try:
1. Get recommendations with "Python Machine Learning SQL TensorFlow Docker"
2. Click Apply on 2–3 jobs
3. Get recommendations again — scores will be personalized.

---

## 12. Future Enhancements

All future enhancements described in the SP301 report have code stubs:

### Enable Sentence-BERT (Semantic Embeddings)

```bash
pip install sentence-transformers
```

Then in `hybrid.py`:
```python
sbert_scores = self.sbert_similarity(candidate_text, job_texts)
# Blend with TF-IDF scores: 0.5 * sbert + 0.5 * tfidf
```

### Enable Real-time Job APIs

```python
# In app.py, replace CSV loading with:
import requests
jobs = requests.get("https://api.adzuna.com/v1/api/jobs/in/search/1?...").json()
```

### Enable Market Demand Prediction

```bash
pip install prophet
```

```python
# Predict which skills will trend using historical job posting frequency
from prophet import Prophet
model = Prophet()
model.fit(skill_trend_df)  # df with 'ds' (date) and 'y' (frequency) columns
forecast = model.predict(future)
```

### Enable Recruiter Dashboard

Add `/recruiter` route in `app.py` with:
- Candidate match analytics
- Funnel visualization (applied → shortlisted → hired)
- Skill demand heatmaps
- Average days-to-hire metrics

---

## Evaluation Metrics

The system evaluates itself using:

| Metric | Formula | Target |
|--------|---------|--------|
| Precision@K | Relevant jobs in top-K / K | > 0.7 |
| Recall@K | Relevant jobs found / Total relevant | > 0.6 |
| F1-Score | 2 × P × R / (P + R) | > 0.65 |
| Accuracy | Correct predictions / Total | > 0.75 |

---

## License

MIT License — Free to use, modify, and distribute.

---

*Built based on SP301 Project Report: "An Intelligent Job Recommendation System using Machine Learning"*
