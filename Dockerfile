# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads models data/user_profiles

# Expose port
EXPOSE 5000

# ── Environment ───────────────────────────────────────────────────────────────
ENV FLASK_ENV=production
ENV SECRET_KEY=change-this-in-production
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# ── Run ───────────────────────────────────────────────────────────────────────
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120
