"""Tiny SQLite store for the local demo login and assistant history."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("FINGUARD_DB_PATH", ROOT / "data" / "finguard_demo.sqlite3"))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt = salt or secrets.token_bytes(16)
    return salt, hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)


def init_db() -> None:
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt BLOB NOT NULL, password_hash BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assistant_history (id INTEGER PRIMARY KEY, username TEXT NOT NULL, client_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS history_user_client ON assistant_history(username, client_id, id);
        """)
        username = os.getenv("FINGUARD_DEMO_USER", "demo")
        password = os.getenv("FINGUARD_DEMO_PASSWORD", "FinGuardDemo!")
        if not db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            salt, digest = _hash_password(password)
            db.execute("INSERT INTO users(username,salt,password_hash) VALUES(?,?,?)", (username, salt, digest))


def verify_user(username: str, password: str) -> bool:
    init_db()
    with connect() as db:
        user = db.execute("SELECT salt,password_hash FROM users WHERE username=?", (username,)).fetchone()
    return bool(user and hmac.compare_digest(_hash_password(password, user["salt"])[1], user["password_hash"]))


def create_session(username: str) -> tuple[str, int]:
    init_db()
    token = secrets.token_urlsafe(32)
    expiry = datetime.now(timezone.utc) + timedelta(hours=12)
    with connect() as db:
        user = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        db.execute("INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), user["id"], expiry.isoformat()))
    return token, 12 * 60 * 60


def session_username(token: str | None) -> str | None:
    if not token:
        return None
    init_db()
    digest = hashlib.sha256(token.encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        row = db.execute("SELECT users.username FROM sessions JOIN users ON users.id=sessions.user_id WHERE token_hash=? AND expires_at>?", (digest, now)).fetchone()
        db.execute("DELETE FROM sessions WHERE expires_at<=?", (now,))
    return row["username"] if row else None


def delete_session(token: str | None) -> None:
    if token:
        with connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))


def save_chat(username: str, client_id: str, question: str, answer: str) -> None:
    with connect() as db:
        db.execute("INSERT INTO assistant_history(username,client_id,question,answer,created_at) VALUES(?,?,?,?,?)", (username, client_id, question, answer, datetime.now(timezone.utc).isoformat()))


def get_chat(username: str, client_id: str, limit: int = 30) -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT question,answer,created_at FROM assistant_history WHERE username=? AND client_id=? ORDER BY id DESC LIMIT ?", (username, client_id, limit)).fetchall()
    return [dict(row) for row in reversed(rows)]
