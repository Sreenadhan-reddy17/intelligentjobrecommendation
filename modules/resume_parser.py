"""
Module 1: Resume Parsing and NLP
Extracts skills, experience, qualifications from resumes (PDF/DOCX/Text)
"""

import re
import os

# ── Try importing optional heavy deps ──────────────────────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ── Skill dictionary (expandable) ─────────────────────────────────────────────
SKILL_KEYWORDS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
    "kotlin", "swift", "go", "rust", "php", "ruby",
    # Web
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring boot", "html", "css", "bootstrap",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
    "bert", "transformers", "spacy", "nltk",
    # Data
    "sql", "mysql", "postgresql", "mongodb", "cassandra", "redis",
    "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "jenkins", "ci/cd", "linux", "git", "github",
    # Other
    "rest api", "graphql", "microservices", "agile", "scrum",
    "data structures", "algorithms", "statistics", "mathematics",
    "excel", "etl", "data warehousing",
]

EDUCATION_KEYWORDS = [
    "b.tech", "b.e", "btech", "be", "bachelor", "b.sc", "bsc",
    "m.tech", "m.e", "mtech", "me", "master", "m.sc", "msc",
    "mba", "phd", "ph.d", "doctorate", "diploma",
    "computer science", "information technology", "electronics",
    "electrical", "mechanical", "data science", "ai", "statistics",
]

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
    r"experience\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)",
    r"(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)",
]


class ResumeParser:
    """Parses resumes and returns a structured profile dictionary."""

    def __init__(self):
        self.skill_keywords = SKILL_KEYWORDS
        self.education_keywords = EDUCATION_KEYWORDS

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_file(self, filepath: str) -> dict:
        """Main entry: accept a file path and return parsed profile."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            text = self._extract_pdf(filepath)
        elif ext in (".docx", ".doc"):
            text = self._extract_docx(filepath)
        else:
            with open(filepath, "r", errors="ignore") as f:
                text = f.read()
        return self.parse_text(text)

    def parse_text(self, text: str) -> dict:
        """Accept raw text and return a structured profile dict."""
        text_lower = text.lower()
        profile = {
            "raw_text": text,
            "skills": self._extract_skills(text_lower),
            "experience_years": self._extract_experience(text_lower),
            "education": self._extract_education(text_lower),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "name": self._extract_name(text),
            "summary": self._generate_summary(text),
        }
        return profile

    # ── Extraction helpers ─────────────────────────────────────────────────────

    def _extract_skills(self, text: str) -> list:
        found = []
        for skill in self.skill_keywords:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text):
                found.append(skill)
        return list(dict.fromkeys(found))  # preserve order, deduplicate

    def _extract_experience(self, text: str) -> int:
        for pattern in EXPERIENCE_PATTERNS:
            m = re.search(pattern, text)
            if m:
                return int(m.group(1))
        # Fallback: count occurrences of year-like phrases
        return 0

    def _extract_education(self, text: str) -> list:
        found = []
        for kw in self.education_keywords:
            if kw in text:
                found.append(kw)
        return list(dict.fromkeys(found))

    def _extract_email(self, text: str) -> str:
        m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        return m.group(0) if m else ""

    def _extract_phone(self, text: str) -> str:
        m = re.search(r"[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,5}[-\s\.]?[0-9]{3,5}", text)
        return m.group(0) if m else ""

    def _extract_name(self, text: str) -> str:
        """Heuristic: first non-empty line that looks like a name."""
        for line in text.splitlines():
            line = line.strip()
            if line and len(line.split()) in (2, 3) and line.replace(" ", "").isalpha():
                return line
        return "Candidate"

    def _generate_summary(self, text: str) -> str:
        """Return first 300 chars of meaningful text as a summary."""
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        return lines[0][:300] if lines else text[:300]

    # ── File readers ───────────────────────────────────────────────────────────

    def _extract_pdf(self, path: str) -> str:
        if not PDF_AVAILABLE:
            return ""
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text

    def _extract_docx(self, path: str) -> str:
        if not DOCX_AVAILABLE:
            return ""
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
