"""Dataset storage for persisting market sets with filters and market lists."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from polymkt.models.schemas import (
    DatasetCreateRequest,
    DatasetFilters,
    DatasetSchema,
    DatasetSummary,
    DatasetUpdateRequest,
)

logger = structlog.get_logger()


class DatasetNotFoundError(Exception):
    """Raised when a dataset is not found."""

    pass


class DatasetStore:
    """SQLite-backed storage for datasets (market sets with filters and market lists)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    filters TEXT,
                    market_ids TEXT NOT NULL,
                    excluded_market_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            # Create index for listing/sorting by updated_at
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_datasets_updated_at
                ON datasets(updated_at DESC)
            """)
            conn.commit()
        finally:
            conn.close()

    def create_dataset(self, request: DatasetCreateRequest) -> DatasetSchema:
        """Create a new dataset and return the created dataset."""
        now = datetime.now(timezone.utc)
        dataset_id = str(uuid.uuid4())

        dataset = DatasetSchema(
            id=dataset_id,
            name=request.name,
            description=request.description,
            filters=request.filters,
            market_ids=request.market_ids,
            excluded_market_ids=request.excluded_market_ids,
            created_at=now,
            updated_at=now,
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO datasets (
                    id, name, description, filters, market_ids,
                    excluded_market_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset.id,
                    dataset.name,
                    dataset.description,
                    json.dumps(dataset.filters.model_dump(mode="json")) if dataset.filters else None,
                    json.dumps(dataset.market_ids),
                    json.dumps(dataset.excluded_market_ids),
                    dataset.created_at.isoformat(),
                    dataset.updated_at.isoformat(),
                ),
            )
            conn.commit()
            logger.info("dataset_created", dataset_id=dataset.id, name=dataset.name)
        finally:
            conn.close()

        return dataset

    def get_dataset(self, dataset_id: str) -> DatasetSchema:
        """Get a dataset by ID. Raises DatasetNotFoundError if not found."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            if row is None:
                raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")
            return self._row_to_dataset(row)
        finally:
            conn.close()

    def update_dataset(
        self, dataset_id: str, request: DatasetUpdateRequest
    ) -> DatasetSchema:
        """Update a dataset. Returns the updated dataset. Raises DatasetNotFoundError if not found."""
        # First get the existing dataset
        existing = self.get_dataset(dataset_id)

        now = datetime.now(timezone.utc)

        # Apply updates only for provided fields
        updated = DatasetSchema(
            id=existing.id,
            name=request.name if request.name is not None else existing.name,
            description=request.description if request.description is not None else existing.description,
            filters=request.filters if request.filters is not None else existing.filters,
            market_ids=request.market_ids if request.market_ids is not None else existing.market_ids,
            excluded_market_ids=(
                request.excluded_market_ids
                if request.excluded_market_ids is not None
                else existing.excluded_market_ids
            ),
            created_at=existing.created_at,
            updated_at=now,
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE datasets SET
                    name = ?,
                    description = ?,
                    filters = ?,
                    market_ids = ?,
                    excluded_market_ids = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    updated.name,
                    updated.description,
                    json.dumps(updated.filters.model_dump(mode="json")) if updated.filters else None,
                    json.dumps(updated.market_ids),
                    json.dumps(updated.excluded_market_ids),
                    updated.updated_at.isoformat(),
                    dataset_id,
                ),
            )
            conn.commit()
            logger.info("dataset_updated", dataset_id=dataset_id)
        finally:
            conn.close()

        return updated

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset. Raises DatasetNotFoundError if not found."""
        # First check that the dataset exists
        self.get_dataset(dataset_id)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            conn.commit()
            logger.info("dataset_deleted", dataset_id=dataset_id)
        finally:
            conn.close()

    def list_datasets(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[DatasetSummary], int]:
        """List datasets with pagination. Returns (summaries, total_count)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Get total count
            cursor = conn.execute("SELECT COUNT(*) FROM datasets")
            total_count = cursor.fetchone()[0]

            # Get paginated results
            cursor = conn.execute(
                """
                SELECT id, name, description, market_ids, excluded_market_ids,
                       created_at, updated_at
                FROM datasets
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            summaries = [self._row_to_summary(row) for row in cursor.fetchall()]

            return summaries, total_count
        finally:
            conn.close()

    def _row_to_dataset(self, row: sqlite3.Row) -> DatasetSchema:
        """Convert a database row to a DatasetSchema."""
        filters_data = json.loads(row["filters"]) if row["filters"] else None
        return DatasetSchema(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            filters=DatasetFilters(**filters_data) if filters_data else None,
            market_ids=json.loads(row["market_ids"]),
            excluded_market_ids=json.loads(row["excluded_market_ids"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_summary(self, row: sqlite3.Row) -> DatasetSummary:
        """Convert a database row to a DatasetSummary."""
        market_ids = json.loads(row["market_ids"])
        excluded_market_ids = json.loads(row["excluded_market_ids"])
        return DatasetSummary(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            market_count=len(market_ids),
            excluded_count=len(excluded_market_ids),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
