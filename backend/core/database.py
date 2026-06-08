"""SQLite database layer."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.core.config import get_settings


class Database:
    """Small SQLite repository with explicit schema initialization."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialized = False
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection configured for dict-like rows."""

        with self._lock:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if self._initialized:
                self._ensure_schema_exists(conn)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def initialize(self) -> None:
        """Create all required tables."""

        with self.connect() as conn:
            self._create_schema(conn)
            self._initialized = True

    def _ensure_schema_exists(self, conn: sqlite3.Connection) -> None:
        """Create tables if the active SQLite file is missing the schema."""

        row = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'users'").fetchone()
        if row is None:
            self._create_schema(conn)

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        """Create all required SQLite tables on the given connection."""

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                metadata TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )

    @staticmethod
    def now() -> str:
        """Return an ISO timestamp."""

        return datetime.now(timezone.utc).isoformat()

    def create_user(self, username: str, email: str, password_hash: str) -> sqlite3.Row:
        """Insert and return a user."""

        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, email, password_hash, self.now()),
            )
            return conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()

    def get_user_by_username(self, username: str) -> sqlite3.Row | None:
        """Return a user by username."""

        with self.connect() as conn:
            return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    def get_user_by_id(self, user_id: int) -> sqlite3.Row | None:
        """Return a user by id."""

        with self.connect() as conn:
            return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def add_document(self, user_id: int, filename: str) -> sqlite3.Row:
        """Insert document metadata."""

        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO documents (user_id, filename, upload_time) VALUES (?, ?, ?)",
                (user_id, filename, self.now()),
            )
            return conn.execute("SELECT * FROM documents WHERE id = ?", (cur.lastrowid,)).fetchone()

    def list_documents(self, user_id: int) -> list[sqlite3.Row]:
        """List documents for a user."""

        with self.connect() as conn:
            return list(conn.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY upload_time DESC", (user_id,)))

    def count_documents(self, user_id: int) -> int:
        """Count a user's PDFs."""

        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents WHERE user_id = ?", (user_id,)).fetchone()[0])

    def get_document(self, user_id: int, document_id: int) -> sqlite3.Row | None:
        """Return one document belonging to a user."""

        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id)
            ).fetchone()

    def delete_document(self, user_id: int, document_id: int) -> bool:
        """Delete a user document and cascaded chunks."""

        with self.connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))
            return cur.rowcount > 0

    def add_chunks(self, document_id: int, chunks: list[tuple[int, str, dict[str, Any]]]) -> None:
        """Persist document chunks."""

        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO chunks (document_id, page_number, chunk_text, metadata) VALUES (?, ?, ?, ?)",
                [(document_id, page, text, json.dumps(metadata)) for page, text, metadata in chunks],
            )

    def list_chunks(self, user_id: int) -> list[sqlite3.Row]:
        """List all chunks belonging to a user's documents."""

        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT chunks.*, documents.filename
                    FROM chunks
                    JOIN documents ON chunks.document_id = documents.id
                    WHERE documents.user_id = ?
                    ORDER BY chunks.id ASC
                    """,
                    (user_id,),
                )
            )

    def add_chat_message(self, user_id: int, role: str, message: str) -> sqlite3.Row:
        """Persist one chat message."""

        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO chat_history (user_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, role, message, self.now()),
            )
            return conn.execute("SELECT * FROM chat_history WHERE id = ?", (cur.lastrowid,)).fetchone()

    def list_chat_history(self, user_id: int, limit: int = 100) -> list[sqlite3.Row]:
        """Return recent chat history."""

        with self.connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM chat_history WHERE user_id = ? ORDER BY id ASC LIMIT ?",
                    (user_id, limit),
                )
            )

    def clear_user_data(self, user_id: int) -> None:
        """Delete a user's documents, chunks, and chat history while keeping the account."""

        with self.connect() as conn:
            conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))


db = Database()
