import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

JSON_COLUMNS = ("settings", "analysts", "sources", "warnings", "review_notes", "usage", "error")
RUN_COLUMNS = (
    "topic", "language", "status", "settings", "analysts", "report", "sources",
    "warnings", "review_notes", "usage", "error",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    thread_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    status TEXT NOT NULL DEFAULT 'running',
    settings TEXT,
    analysts TEXT,
    report TEXT,
    sources TEXT,
    warnings TEXT,
    review_notes TEXT,
    usage TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        run = dict(row)
        for column in JSON_COLUMNS:
            raw = run.get(column)
            run[column] = json.loads(raw) if raw else None
        return run

    def create_run(self, thread_id: str, topic: str, language: str, settings: dict) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (thread_id, topic, language, status, settings, created_at, updated_at)"
                " VALUES (?, ?, ?, 'running', ?, ?, ?)",
                (thread_id, topic, language, json.dumps(settings, ensure_ascii=False), now, now),
            )
            self._conn.commit()

    def update_run(self, thread_id: str, **fields: Any) -> None:
        unknown = set(fields) - set(RUN_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown run columns: {sorted(unknown)}")
        if not fields:
            return
        values = [
            json.dumps(value, ensure_ascii=False) if name in JSON_COLUMNS and value is not None else value
            for name, value in fields.items()
        ]
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE runs SET {assignments}, updated_at = ? WHERE thread_id = ?",
                (*values, time.time(), thread_id),
            )
            self._conn.commit()

    def get_run(self, thread_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE thread_id = ?", (thread_id,)).fetchone()
        return self._decode(row)

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT thread_id, topic, language, status, created_at, updated_at,"
                " (report IS NOT NULL AND report != '') AS has_report"
                " FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) | {"has_report": bool(row["has_report"])} for row in rows]

    def delete_run(self, thread_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM runs WHERE thread_id = ?", (thread_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def mark_interrupted(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = 'interrupted', updated_at = ? WHERE status = 'running'",
                (time.time(),),
            )
            self._conn.commit()
            return cursor.rowcount
