"""SQLite-backed session metadata store.

Persists session metadata (ID, status, model, project_path, timestamps)
to a SQLite database. Event buffers and graph checkpoints are stored
separately (event buffers are in-memory; checkpoints use SqliteSaver).
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".eunomia" / "eunomia.db"


class SQLiteSessionStore:
    """Persist session metadata to SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode for better concurrency."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self) -> None:
        """Create the sessions table if it doesn't exist."""
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'created',
                    model TEXT NOT NULL DEFAULT '',
                    project_path TEXT NOT NULL DEFAULT '',
                    prd_content TEXT DEFAULT '',
                    max_retries INTEGER DEFAULT 3,
                    hitl_level TEXT DEFAULT 'autonomous',
                    total_tasks INTEGER DEFAULT 0,
                    completed_tasks TEXT DEFAULT '[]',
                    failed_tasks TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_session(
        self,
        session_id: str,
        status: str = "created",
        model: str = "",
        project_path: str = "",
        prd_content: str = "",
        max_retries: int = 3,
        hitl_level: str = "autonomous",
        total_tasks: int = 0,
        completed_tasks: list[str] | None = None,
        failed_tasks: list[str] | None = None,
    ) -> None:
        """INSERT OR REPLACE session metadata."""
        now = _now_iso()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, status, model, project_path, prd_content,
                    max_retries, hitl_level, total_tasks,
                    completed_tasks, failed_tasks,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    status,
                    model,
                    project_path,
                    prd_content,
                    max_retries,
                    hitl_level,
                    total_tasks,
                    json.dumps(completed_tasks or []),
                    json.dumps(failed_tasks or []),
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """SELECT session by ID. Returns None if not found."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_dict(row)
        finally:
            conn.close()

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """SELECT recent sessions ordered by created_at DESC."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [_row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def update_status(self, session_id: str, status: str) -> bool:
        """UPDATE session status. Returns True if row was found."""
        conn = self._connect()
        try:
            result = conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (status, _now_iso(), session_id),
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def update_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        total_tasks: int | None = None,
        completed_tasks: list[str] | None = None,
        failed_tasks: list[str] | None = None,
    ) -> bool:
        """Update specific fields of a session. Returns True if row was found."""
        updates: list[str] = []
        params: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if total_tasks is not None:
            updates.append("total_tasks = ?")
            params.append(total_tasks)
        if completed_tasks is not None:
            updates.append("completed_tasks = ?")
            params.append(json.dumps(completed_tasks))
        if failed_tasks is not None:
            updates.append("failed_tasks = ?")
            params.append(json.dumps(failed_tasks))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(_now_iso())
        params.append(session_id)

        sql = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"
        conn = self._connect()
        try:
            result = conn.execute(sql, params)
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        """DELETE session by ID. Returns True if row was found."""
        conn = self._connect()
        try:
            result = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict with JSON-parsed fields."""
    d = dict(row)
    # Parse JSON array fields
    for field in ("completed_tasks", "failed_tasks"):
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except json.JSONDecodeError:
                d[field] = []
    return d
