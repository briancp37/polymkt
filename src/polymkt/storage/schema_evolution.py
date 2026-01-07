"""Schema evolution utilities for Parquet and DuckDB.

This module provides utilities for evolving schemas over time without
breaking reads or queries on older data. Key capabilities:

1. Safe column projection - reads data with old schema, fills missing columns
2. Schema versioning - tracks schema changes over time
3. Backward-compatible views - views that handle missing columns gracefully
4. Safe column access - utilities for accessing optional columns

PRD Reference:
- "Schema evolution: adding new columns later (e.g., tags or additional
   market metadata) should not break reads or backtests"
- "Verify older partitions remain readable"
- "Verify query layer selects required columns safely"
- "Verify backtest code handles missing optional columns gracefully"
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger()


@dataclass
class SchemaVersion:
    """Represents a schema version with its columns."""

    version: int
    name: str
    columns: list[str]
    added_columns: list[str]  # Columns added in this version
    created_at: datetime


class SchemaEvolutionManager:
    """Manages schema evolution for Parquet/DuckDB data.

    Provides methods for:
    - Tracking schema versions in metadata
    - Reading old Parquet files with new schema (null-filling missing columns)
    - Creating views that handle missing columns gracefully
    - Safe column access with default values
    """

    def __init__(self, metadata_db_path: Path) -> None:
        """Initialize the schema evolution manager.

        Args:
            metadata_db_path: Path to SQLite metadata database
        """
        self.metadata_db_path = metadata_db_path
        self._init_schema_table()

    def _init_schema_table(self) -> None:
        """Create schema_versions table if it doesn't exist."""
        import sqlite3

        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.metadata_db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    columns TEXT NOT NULL,
                    added_columns TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(entity, version)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def register_schema_version(
        self,
        entity: str,
        version: int,
        columns: list[str],
        added_columns: list[str] | None = None,
    ) -> SchemaVersion:
        """Register a new schema version for an entity.

        Args:
            entity: Entity name (e.g., "trades", "markets")
            version: Schema version number
            columns: All columns in this schema version
            added_columns: Columns added in this version (vs previous)

        Returns:
            SchemaVersion object
        """
        import json
        import sqlite3

        created_at = datetime.now(timezone.utc)
        added = added_columns or []

        conn = sqlite3.connect(str(self.metadata_db_path))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_versions
                (entity, version, columns, added_columns, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    entity,
                    version,
                    json.dumps(columns),
                    json.dumps(added),
                    created_at.isoformat(),
                ),
            )
            conn.commit()

            logger.info(
                "schema_version_registered",
                entity=entity,
                version=version,
                column_count=len(columns),
                added_columns=added,
            )

            return SchemaVersion(
                version=version,
                name=entity,
                columns=columns,
                added_columns=added,
                created_at=created_at,
            )
        finally:
            conn.close()

    def get_current_version(self, entity: str) -> SchemaVersion | None:
        """Get the current (latest) schema version for an entity.

        Args:
            entity: Entity name

        Returns:
            SchemaVersion or None if no versions registered
        """
        import json
        import sqlite3

        conn = sqlite3.connect(str(self.metadata_db_path))
        try:
            cursor = conn.execute(
                """
                SELECT version, columns, added_columns, created_at
                FROM schema_versions
                WHERE entity = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (entity,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return SchemaVersion(
                version=row[0],
                name=entity,
                columns=json.loads(row[1]),
                added_columns=json.loads(row[2]),
                created_at=datetime.fromisoformat(row[3]),
            )
        finally:
            conn.close()

    def get_all_versions(self, entity: str) -> list[SchemaVersion]:
        """Get all schema versions for an entity.

        Args:
            entity: Entity name

        Returns:
            List of SchemaVersion objects, newest first
        """
        import json
        import sqlite3

        conn = sqlite3.connect(str(self.metadata_db_path))
        try:
            cursor = conn.execute(
                """
                SELECT version, columns, added_columns, created_at
                FROM schema_versions
                WHERE entity = ?
                ORDER BY version DESC
                """,
                (entity,),
            )

            return [
                SchemaVersion(
                    version=row[0],
                    name=entity,
                    columns=json.loads(row[1]),
                    added_columns=json.loads(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()


def read_parquet_with_schema_evolution(
    parquet_path: Path,
    target_schema: pa.Schema,
) -> pa.Table:
    """Read a Parquet file and project to target schema.

    If the file has a different (older) schema, this function:
    - Keeps all columns that exist in both schemas
    - Fills missing columns with null values of the target type
    - Drops columns that are not in the target schema

    This enables reading old Parquet partitions with a newer schema.

    Args:
        parquet_path: Path to Parquet file or directory
        target_schema: Target PyArrow schema to project to

    Returns:
        PyArrow table with target schema
    """
    # Read the parquet file with its existing schema
    if parquet_path.is_dir():
        # Dataset read for partitioned data
        import pyarrow.dataset as ds

        dataset = ds.dataset(parquet_path, format="parquet")
        existing_table = dataset.to_table()
    else:
        existing_table = pq.read_table(parquet_path)

    return project_to_schema(existing_table, target_schema)


def project_to_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    """Project a table to a target schema.

    - Keeps columns that exist in both schemas (with type casting)
    - Adds missing columns as null arrays
    - Drops columns not in target schema

    Args:
        table: Source PyArrow table
        target_schema: Target schema to project to

    Returns:
        Table with target schema
    """
    existing_columns = set(table.column_names)
    arrays: list[pa.Array] = []

    for field in target_schema:
        if field.name in existing_columns:
            # Column exists - cast to target type if needed
            col = table.column(field.name)
            try:
                col = col.cast(field.type)
            except pa.ArrowInvalid:
                # Type cast failed - fill with nulls
                logger.warning(
                    "schema_evolution_type_cast_failed",
                    column=field.name,
                    source_type=str(col.type),
                    target_type=str(field.type),
                )
                col = pa.nulls(table.num_rows, type=field.type)
            arrays.append(col)
        else:
            # Column missing - fill with nulls
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            logger.debug(
                "schema_evolution_column_filled",
                column=field.name,
                type=str(field.type),
                rows=table.num_rows,
            )

    return pa.table(dict(zip([f.name for f in target_schema], arrays)))


def get_safe_column_sql(
    column_name: str,
    column_type: str = "VARCHAR",
    default_value: str | None = None,
    source_alias: str | None = None,
) -> str:
    """Generate SQL for safely accessing a column that may not exist.

    Uses TRY_CAST to handle type mismatches and COALESCE for default values.

    Args:
        column_name: Name of the column
        column_type: DuckDB type to cast to
        default_value: Default value if column is NULL (SQL expression)
        source_alias: Table alias prefix (e.g., "t" for "t.column_name")

    Returns:
        SQL expression for safe column access
    """
    if source_alias:
        full_name = f"{source_alias}.{column_name}"
    else:
        full_name = column_name

    if default_value is not None:
        return f"COALESCE(TRY_CAST({full_name} AS {column_type}), {default_value}) AS {column_name}"
    else:
        return f"TRY_CAST({full_name} AS {column_type}) AS {column_name}"


def create_evolution_safe_view(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    source: str,
    required_columns: list[tuple[str, str]],
    optional_columns: list[tuple[str, str, str]],
) -> None:
    """Create a DuckDB view that handles schema evolution gracefully.

    The view will:
    - Always include required columns (error if missing)
    - Include optional columns with defaults if missing

    Args:
        conn: DuckDB connection
        view_name: Name of the view to create
        source: SQL source (table or read_parquet expression)
        required_columns: List of (column_name, type) for required columns
        optional_columns: List of (column_name, type, default_sql) for optional columns
    """
    columns_sql: list[str] = []

    # Required columns - straight select
    for col_name, col_type in required_columns:
        columns_sql.append(f"TRY_CAST({col_name} AS {col_type}) AS {col_name}")

    # Optional columns - with defaults
    for col_name, col_type, default_sql in optional_columns:
        columns_sql.append(
            f"COALESCE(TRY_CAST({col_name} AS {col_type}), {default_sql}) AS {col_name}"
        )

    sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            {', '.join(columns_sql)}
        FROM {source}
    """

    conn.execute(sql)
    logger.info(
        "evolution_safe_view_created",
        view=view_name,
        required_count=len(required_columns),
        optional_count=len(optional_columns),
    )


def safe_get(
    record: dict[str, Any],
    column: str,
    default: Any = None,
    expected_type: type | None = None,
) -> Any:
    """Safely get a column value from a record dict.

    Handles missing columns and type mismatches gracefully.
    Use this in backtest and query code to handle schema evolution.

    Args:
        record: Dict-like record (row from query)
        column: Column name to access
        default: Default value if column is missing or None
        expected_type: Expected type - if value doesn't match, return default

    Returns:
        Column value or default

    Example:
        >>> trade = {"price": 0.75, "timestamp": ...}
        >>> safe_get(trade, "price", 0.0, float)
        0.75
        >>> safe_get(trade, "new_field", "unknown", str)
        "unknown"
    """
    value = record.get(column)

    if value is None:
        return default

    if expected_type is not None and not isinstance(value, expected_type):
        # Type mismatch - try to convert
        try:
            return expected_type(value)
        except (ValueError, TypeError):
            return default

    return value


def detect_schema_changes(
    old_schema: pa.Schema,
    new_schema: pa.Schema,
) -> tuple[list[str], list[str], list[str]]:
    """Detect changes between two schemas.

    Args:
        old_schema: Previous schema version
        new_schema: New schema version

    Returns:
        Tuple of (added_columns, removed_columns, type_changed_columns)
    """
    old_fields = {f.name: f for f in old_schema}
    new_fields = {f.name: f for f in new_schema}

    added = [name for name in new_fields if name not in old_fields]
    removed = [name for name in old_fields if name not in new_fields]

    type_changed = []
    for name in old_fields:
        if name in new_fields and old_fields[name].type != new_fields[name].type:
            type_changed.append(name)

    return added, removed, type_changed
