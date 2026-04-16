import urllib.request
import zlib
from base64 import urlsafe_b64encode
import fpdf
import os

mmd = '''flowchart TD
    JS[Job Seeker UI] -->|Uploads Resume / Text| app[app.py Routes & Session]
    RP[Recruiter UI] -->|Uploads Resumes + Job ID| app
    app --> uploads[(Uploads Folder)]
    app --> parser[Resume Parser PDF, DOCX, TXT]
    parser -->|Raw Text & Skills| feat_eng[Feature Engineering Sentence Transformers]
    csv[(CSV Databases jobs & interactions)] -.->|Job Descriptions| feat_eng
    models[(Pre-trained Models)] -.->|all-MiniLM-L6-v2| feat_eng
    feat_eng -->|Embeddings| cb[Content-Based Filter]
    cb --> hyb[Hybrid Recommender]
    csv -.->|Past Clicks/Applies| cf[Collaborative Filter]
    cf --> hyb
    app -->|Session Events| al[Adaptive Learner]
    al --> hyb
    hyb -->|Ensembled Scores| rank[Ranking Model]
    rank -->|Predict Probabilities| gap[Skill Gap Analyzer]
    gap -->|Missing Skills| app
    rank -->|Ordered Jobs| app
    app -->|JSON / Rendered HTML| JS
    app -->|Ranked Candidates| RP
'''

compressed = zlib.compress(mmd.encode('utf-8'))
encoded = urlsafe_b64encode(compressed).decode('ascii')
url = f'https://kroki.io/mermaid/png/{encoded}'

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        with open('arch_diag.png', 'wb') as f:
            f.write(response.read())
    print('Image saved!')
except Exception as e:
    print(f'Error fetching: {e}')

class PDF(fpdf.FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Intelligent Job Recommendation System Architecture', 0, 1, 'C')
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 8, 'This document provides a high-level overview of the system architecture for the Intelligent Job Recommendation System, highlighting industry-standard features like real-time personalization.')
pdf.ln(5)

if os.path.exists('arch_diag.png'):
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Architecture Diagram', 0, 1)
    pdf.image('arch_diag.png', x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
    pdf.ln(10)

pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Industry-Level Personalization Features', 0, 1)
pdf.ln(2)

personalization = [
    ('1. User Profile-Based Recommendations', 
     '- The system generates dynamic recommendations based on precise capability mapping using extracted attributes (skills, experience, and domain knowledge) from candidate resumes.'),
    ('2. Learning from User Behavior',
     '- Adaptive Learning tracks live interactions (clicks) across sessions, applying temporal ranking boosts to dynamically respond to shifting user interests, mimicking heavy-duty e-commerce engines.'),
    ('3. Saved Jobs & Implicit Preferences',
     '- The Collaborative Filter leverages past interaction records across similar user profiles to predict and highlight career paths that might conceptually appeal to the candidate.')
]

for title, desc in personalization:
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, title, 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, desc)
    pdf.ln(2)

pdf.ln(4)
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Component Breakdown', 0, 1)
pdf.ln(2)

components = [
    ('1. Frontend & Presentation Layer', 
     '- Flask Templates: UI built using Jinja2 templates.\n- User Roles: Job Seekers & Recruiters.'),
    ('2. Core Controller (app.py)',
     '- The Flask app manages requests, file uploads, ML init, and runs recommendation pipelines.'),
    ('3. Data Processing & Feature Engineering',
     '- Resume Parser: Reads files to extract candidate fields and skills.\n- Feature Engineer: Embeddings via Sentence Transformers (all-MiniLM), calculates Jaccard overlaps.'),
    ('4. Recommendation Pipeline',
     '- Stage 1: Content-Based Filtering (Cosine Similarity).\n- Stage 2: Collaborative Filtering (Historic behaviors).\n- Stage 3: Adaptive Learning (In-session behaviors).\n- Stage 4: Hybrid Recommender (Weighted scoring).'),
    ('5. Final Ranking & Insights',
     '- Re-Ranking Model: Gradient Boosting maps scores to probabilities.\n- Skill Gap Analyzer: Evaluates skill differentials.')
]

for title, desc in components:
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, title, 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, desc)
    pdf.ln(4)

pdf.output('architecture_overview.pdf')
print('PDF generated!')
