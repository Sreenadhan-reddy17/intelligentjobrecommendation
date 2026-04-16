import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "job_recommender.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        profile_data TEXT
    );
    CREATE TABLE IF NOT EXISTS saved_jobs (
        user_id TEXT,
        job_id INTEGER,
        saved_at TIMESTAMP,
        PRIMARY KEY (user_id, job_id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS applied_jobs (
        user_id TEXT,
        job_id INTEGER,
        applied_at TIMESTAMP,
        PRIMARY KEY (user_id, job_id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    ''')
    conn.commit()
    conn.close()

def get_user(user_id):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

def upsert_user(user_id, name, email, profile_data=None):
    conn = get_db()
    existing = get_user(user_id)
    if existing:
        conn.execute("UPDATE users SET name=?, email=? WHERE id=?", 
                     (name, email, user_id))
    else:
        conn.execute("INSERT INTO users (id, name, email, profile_data) VALUES (?, ?, ?, ?)",
                     (user_id, name, email, json.dumps(profile_data) if profile_data else "{}"))
    conn.commit()
    conn.close()

def update_user_profile(user_id, profile_dict):
    conn = get_db()
    conn.execute("UPDATE users SET profile_data=? WHERE id=?", (json.dumps(profile_dict), user_id))
    conn.commit()
    conn.close()

def save_job(user_id, job_id):
    conn = get_db()
    try:
        conn.execute("INSERT INTO saved_jobs (user_id, job_id, saved_at) VALUES (?, ?, ?)",
                     (user_id, job_id, datetime.now()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Already saved
    conn.close()

def remove_saved_job(user_id, job_id):
    conn = get_db()
    conn.execute("DELETE FROM saved_jobs WHERE user_id=? AND job_id=?", (user_id, job_id))
    conn.commit()
    conn.close()

def apply_job(user_id, job_id):
    conn = get_db()
    try:
        conn.execute("INSERT INTO applied_jobs (user_id, job_id, applied_at) VALUES (?, ?, ?)",
                     (user_id, job_id, datetime.now()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def get_saved_jobs(user_id):
    conn = get_db()
    res = conn.execute("SELECT job_id, saved_at FROM saved_jobs WHERE user_id=? ORDER BY saved_at DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in res]

def get_applied_jobs(user_id):
    conn = get_db()
    res = conn.execute("SELECT job_id, applied_at FROM applied_jobs WHERE user_id=? ORDER BY applied_at DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in res]

# Auto-initialize DB on import
init_db()
