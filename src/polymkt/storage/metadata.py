"""Metadata storage for run history and watermarks using SQLite."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from polymkt.models.schemas import RunRecord

logger = structlog.get_logger()


class MetadataStore:
    """SQLite-backed metadata storage for pipeline run history and watermarks."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    rows_read TEXT NOT NULL,
                    rows_written TEXT NOT NULL,
                    watermark_before TEXT,
                    watermark_after TEXT,
                    schema_version TEXT NOT NULL,
                    error_message TEXT,
                    duration_seconds REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watermarks (
                    entity TEXT PRIMARY KEY,
                    watermark_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def create_run(self, run: RunRecord) -> None:
        """Insert a new run record."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, run_type, start_time, end_time, status,
                    rows_read, rows_written, watermark_before, watermark_after,
                    schema_version, error_message, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.run_type,
                    run.start_time.isoformat(),
                    run.end_time.isoformat() if run.end_time else None,
                    run.status,
                    json.dumps(run.rows_read),
                    json.dumps(run.rows_written),
                    json.dumps(run.watermark_before) if run.watermark_before else None,
                    json.dumps(run.watermark_after) if run.watermark_after else None,
                    run.schema_version,
                    run.error_message,
                    run.duration_seconds,
                ),
            )
            conn.commit()
            logger.info("run_created", run_id=run.run_id, run_type=run.run_type)
        finally:
            conn.close()

    def update_run(self, run: RunRecord) -> None:
        """Update an existing run record."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE runs SET
                    end_time = ?,
                    status = ?,
                    rows_read = ?,
                    rows_written = ?,
                    watermark_before = ?,
                    watermark_after = ?,
                    error_message = ?,
                    duration_seconds = ?
                WHERE run_id = ?
                """,
                (
                    run.end_time.isoformat() if run.end_time else None,
                    run.status,
                    json.dumps(run.rows_read),
                    json.dumps(run.rows_written),
                    json.dumps(run.watermark_before) if run.watermark_before else None,
                    json.dumps(run.watermark_after) if run.watermark_after else None,
                    run.error_message,
                    run.duration_seconds,
                    run.run_id,
                ),
            )
            conn.commit()
            logger.info("run_updated", run_id=run.run_id, status=run.status)
        finally:
            conn.close()

    def get_run(self, run_id: str) -> RunRecord | None:
        """Get a run record by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_run(row)
        finally:
            conn.close()

    def list_runs(self, limit: int = 100) -> list[RunRecord]:
        """List recent runs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM runs ORDER BY start_time DESC LIMIT ?", (limit,)
            )
            return [self._row_to_run(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        """Convert a database row to a RunRecord."""
        return RunRecord(
            run_id=row["run_id"],
            run_type=row["run_type"],
            start_time=datetime.fromisoformat(row["start_time"]),
            end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
            status=row["status"],
            rows_read=json.loads(row["rows_read"]),
            rows_written=json.loads(row["rows_written"]),
            watermark_before=json.loads(row["watermark_before"]) if row["watermark_before"] else None,
            watermark_after=json.loads(row["watermark_after"]) if row["watermark_after"] else None,
            schema_version=row["schema_version"],
            error_message=row["error_message"],
            duration_seconds=row["duration_seconds"],
        )

    def get_watermark(self, entity: str) -> Any | None:
        """Get the current watermark for an entity."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT watermark_value FROM watermarks WHERE entity = ?", (entity,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def set_watermark(self, entity: str, value: Any) -> None:
        """Set the watermark for an entity."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO watermarks (entity, watermark_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(entity) DO UPDATE SET
                    watermark_value = excluded.watermark_value,
                    updated_at = excluded.updated_at
                """,
                (entity, json.dumps(value), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            logger.info("watermark_set", entity=entity)
        finally:
            conn.close()

    def get_all_watermarks(self) -> dict[str, Any]:
        """Get all current watermarks."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT entity, watermark_value FROM watermarks")
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        finally:
            conn.close()
