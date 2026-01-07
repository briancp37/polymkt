"""Bootstrap import pipeline for loading CSV data into Parquet."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.csv as csv
import structlog

from polymkt.config import settings
from polymkt.models.schemas import BootstrapSummary, RunRecord
from polymkt.pipeline.normalize import (
    ValidationResult,
    validate_and_normalize_events,
    validate_and_normalize_markets,
    validate_and_normalize_order_filled,
    validate_and_normalize_trades,
)
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.parquet import (
    EVENTS_SCHEMA,
    MARKETS_SCHEMA,
    ORDER_FILLED_SCHEMA,
    TRADES_SCHEMA,
    ParquetWriter,
)

logger = structlog.get_logger()

# Column mappings from CSV to our internal schema
EVENTS_COLUMN_MAPPING = {
    "eventId": "event_id",
    "createdAt": "created_at",
}

MARKETS_COLUMN_MAPPING = {
    "createdAt": "created_at",
    "closedTime": "closed_time",
    "eventId": "event_id",
}

TRADES_COLUMN_MAPPING = {
    "transactionHash": "transaction_hash",
}

ORDER_FILLED_COLUMN_MAPPING = {
    "makerAssetId": "maker_asset_id",
    "makerAmountFilled": "maker_amount_filled",
    "takerAssetId": "taker_asset_id",
    "takerAmountFilled": "taker_amount_filled",
    "transactionHash": "transaction_hash",
}


def _rename_columns(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    """Rename columns in a PyArrow table."""
    new_names = [mapping.get(name, name) for name in table.column_names]
    return table.rename_columns(new_names)


def _read_csv_with_schema(
    csv_path: Path,
    target_schema: pa.Schema,
    column_mapping: dict[str, str],
) -> pa.Table:
    """Read a CSV file and normalize to target schema."""
    logger.info("reading_csv", path=str(csv_path))

    # Read CSV with type inference
    read_options = csv.ReadOptions()
    parse_options = csv.ParseOptions()
    convert_options = csv.ConvertOptions(
        timestamp_parsers=[
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ],
    )

    table = csv.read_csv(
        csv_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    logger.info("csv_read", path=str(csv_path), rows=table.num_rows, columns=table.column_names)

    # Rename columns
    table = _rename_columns(table, column_mapping)

    # Build a new table with target schema columns
    columns: list[pa.Array] = []
    for field in target_schema:
        if field.name in table.column_names:
            col = table.column(field.name)
            # Cast to target type
            try:
                col = col.cast(field.type)
            except pa.ArrowInvalid:
                # For timestamp fields that might be strings
                if pa.types.is_timestamp(field.type):
                    col = pa.compute.strptime(col, format="%Y-%m-%d %H:%M:%S", unit="us")
                    col = col.cast(field.type)
                else:
                    raise
            columns.append(col)
        else:
            # Create null column for missing fields
            columns.append(pa.nulls(table.num_rows, type=field.type))

    return pa.table(dict(zip([f.name for f in target_schema], columns)), schema=target_schema)


def _read_events_csv(csv_path: Path) -> pa.Table:
    """
    Read events CSV with special handling for tags column (JSON list).

    Events CSV format expected:
    eventId,tags,title,description,createdAt
    evt_123,"[""tag1"",""tag2""]",My Event,Description,2025-01-01 00:00:00
    """
    import json

    logger.info("reading_events_csv", path=str(csv_path))

    # Read CSV as strings first
    table = csv.read_csv(csv_path)
    logger.info(
        "events_csv_read",
        path=str(csv_path),
        rows=table.num_rows,
        columns=table.column_names,
    )

    # Rename columns
    table = _rename_columns(table, EVENTS_COLUMN_MAPPING)

    rows = table.to_pylist()
    processed_rows: list[dict[str, Any]] = []

    for row in rows:
        processed_row = dict(row)

        # Parse tags from JSON string if present
        tags_value = row.get("tags")
        if tags_value is None or tags_value == "":
            processed_row["tags"] = []
        elif isinstance(tags_value, str):
            try:
                parsed_tags = json.loads(tags_value)
                if isinstance(parsed_tags, list):
                    processed_row["tags"] = [str(t) for t in parsed_tags]
                else:
                    processed_row["tags"] = [str(parsed_tags)]
            except json.JSONDecodeError:
                # If not valid JSON, treat as a single tag or comma-separated
                if "," in tags_value:
                    processed_row["tags"] = [t.strip() for t in tags_value.split(",")]
                else:
                    processed_row["tags"] = [tags_value]
        elif isinstance(tags_value, list):
            processed_row["tags"] = [str(t) for t in tags_value]
        else:
            processed_row["tags"] = []

        # Parse created_at timestamp
        from polymkt.pipeline.normalize import normalize_timestamp
        processed_row["created_at"] = normalize_timestamp(
            row.get("created_at"), "created_at"
        )

        processed_rows.append(processed_row)

    # Build PyArrow table with proper schema
    return pa.Table.from_pylist(processed_rows, schema=EVENTS_SCHEMA)


def _join_events_tags_to_markets(
    markets_table: pa.Table,
    events_table: pa.Table,
) -> pa.Table:
    """
    Join event tags to markets via event_id.

    This enriches markets with tags from their parent events.

    Args:
        markets_table: Markets table with event_id column
        events_table: Events table with event_id and tags columns

    Returns:
        Markets table with tags populated from events join
    """
    import duckdb

    logger.info(
        "joining_events_to_markets",
        markets_rows=markets_table.num_rows,
        events_rows=events_table.num_rows,
    )

    # Use DuckDB in-memory for efficient join
    conn = duckdb.connect(":memory:")

    # Register tables
    conn.register("markets", markets_table)
    conn.register("events", events_table)

    # Left join markets to events to get tags
    # Markets without an event_id or with no matching event get NULL tags
    result = conn.execute("""
        SELECT
            m.id,
            m.question,
            m.created_at,
            m.answer1,
            m.answer2,
            m.neg_risk,
            m.market_slug,
            m.token1,
            m.token2,
            m.condition_id,
            m.volume,
            m.ticker,
            m.closed_time,
            m.description,
            m.category,
            m.event_id,
            COALESCE(e.tags, []) AS tags
        FROM markets m
        LEFT JOIN events e ON m.event_id = e.event_id
    """).fetch_arrow_table()

    # Log unmapped markets (those with event_id but no matching event)
    unmapped_result = conn.execute("""
        SELECT COUNT(*) as cnt FROM markets m
        WHERE m.event_id IS NOT NULL
        AND m.event_id NOT IN (SELECT event_id FROM events)
    """).fetchone()
    unmapped_count: int = int(unmapped_result[0]) if unmapped_result else 0

    if unmapped_count > 0:
        logger.warning(
            "markets_with_unmapped_events",
            unmapped_count=unmapped_count,
            message=f"{unmapped_count} markets have event_id but no matching event in events table",
        )

    conn.close()

    logger.info(
        "events_join_complete",
        output_rows=result.num_rows,
        unmapped_markets=unmapped_count,
    )

    return result


class SchemaValidationError(Exception):
    """Raised when schema validation fails with actionable error information."""

    def __init__(self, message: str, missing_fields: list[str] | None = None,
                 invalid_fields: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.missing_fields = missing_fields or []
        self.invalid_fields = invalid_fields or {}


def validate_schema_requirements(
    markets_table: pa.Table | None = None,
    events_table: pa.Table | None = None,
    require_events_for_tags: bool = False,
) -> None:
    """
    Validate schema requirements and fail fast with actionable errors.

    Checks:
    - Required fields are present in markets (id, question)
    - Required fields are present in events if provided (event_id)
    - Join key (event_id) exists if events table is provided

    Args:
        markets_table: Markets table to validate
        events_table: Events table to validate (optional)
        require_events_for_tags: If True, require events table for tags derivation

    Raises:
        SchemaValidationError: If validation fails with details about what's wrong
    """
    errors: list[str] = []
    missing_fields: list[str] = []
    invalid_fields: dict[str, str] = {}

    if markets_table is not None:
        market_columns = set(markets_table.column_names)

        # Check required market fields
        required_market_fields = ["id", "question"]
        for field in required_market_fields:
            if field not in market_columns:
                missing_fields.append(f"markets.{field}")
                errors.append(f"Missing required field in markets: {field}")

        # Check that event_id exists if we want to join events
        if require_events_for_tags and "event_id" not in market_columns:
            missing_fields.append("markets.event_id")
            errors.append(
                "markets.event_id is required for joining event tags. "
                "Add event_id column to markets.csv or set require_events_for_tags=False"
            )

    if events_table is not None:
        event_columns = set(events_table.column_names)

        # Check required event fields
        if "event_id" not in event_columns:
            missing_fields.append("events.event_id")
            errors.append("Missing required field in events: event_id")

        if "tags" not in event_columns:
            missing_fields.append("events.tags")
            errors.append("Missing required field in events: tags")

    if require_events_for_tags and events_table is None:
        errors.append(
            "Events table is required for deriving market tags. "
            "Provide events.csv or set require_events_for_tags=False"
        )

    if errors:
        error_message = "Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(
            "schema_validation_failed",
            errors=errors,
            missing_fields=missing_fields,
            invalid_fields=invalid_fields,
        )
        raise SchemaValidationError(error_message, missing_fields, invalid_fields)

    logger.info("schema_validation_passed")


def run_bootstrap(
    markets_csv: Path | None = None,
    trades_csv: Path | None = None,
    order_filled_csv: Path | None = None,
    events_csv: Path | None = None,
    parquet_dir: Path | None = None,
    duckdb_path: Path | None = None,
    metadata_db_path: Path | None = None,
    normalize_addresses: bool = True,
    validate_data: bool = True,
    partitioning_enabled: bool | None = None,
    hash_bucket_count: int | None = None,
    require_events_for_tags: bool = False,
) -> BootstrapSummary:
    """
    Run the bootstrap import process.

    Reads existing CSV files and converts them to Parquet format,
    then creates DuckDB views over the Parquet files.

    Args:
        markets_csv: Path to markets CSV file
        trades_csv: Path to trades CSV file
        order_filled_csv: Path to orderFilled CSV file
        events_csv: Path to events CSV file (for deriving market tags)
        parquet_dir: Directory for Parquet output
        duckdb_path: Path to DuckDB database file
        metadata_db_path: Path to metadata SQLite database
        normalize_addresses: Whether to normalize Ethereum addresses (lowercase 0x)
        validate_data: Whether to validate and quarantine invalid rows
        partitioning_enabled: Whether to partition trades Parquet by year/month/day/hash_bucket
        hash_bucket_count: Number of hash buckets for market_id partitioning
        require_events_for_tags: If True, require events CSV for deriving market tags

    Returns:
        BootstrapSummary with run details including any validation issues
    """
    # Use settings defaults if not provided
    markets_csv = markets_csv or settings.markets_csv
    trades_csv = trades_csv or settings.trades_csv
    order_filled_csv = order_filled_csv or settings.order_filled_csv
    events_csv = events_csv or settings.events_csv
    parquet_dir = parquet_dir or settings.parquet_dir
    duckdb_path = duckdb_path or settings.duckdb_path
    metadata_db_path = metadata_db_path or settings.metadata_db_path
    partitioning_enabled = (
        partitioning_enabled
        if partitioning_enabled is not None
        else settings.parquet_partitioning_enabled
    )
    hash_bucket_count = hash_bucket_count or settings.parquet_hash_bucket_count

    # Initialize components
    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    timer_start = perf_counter()

    metadata_store = MetadataStore(metadata_db_path)
    parquet_writer = ParquetWriter(
        parquet_dir,
        partitioning_enabled=partitioning_enabled,
        hash_bucket_count=hash_bucket_count,
    )

    # Create initial run record
    run_record = RunRecord(
        run_id=run_id,
        run_type="bootstrap",
        start_time=start_time,
        status="running",
        schema_version="1.0.0",
    )
    metadata_store.create_run(run_record)

    parquet_files: list[str] = []
    rows_read: dict[str, int] = {}
    rows_written: dict[str, int] = {}
    rows_quarantined: dict[str, int] = {}
    validation_results: dict[str, ValidationResult] = {}

    try:
        # Process events first (needed for market tags join)
        events_table: pa.Table | None = None
        if events_csv.exists():
            logger.info("processing_events", path=str(events_csv))
            events_table = _read_events_csv(events_csv)
            rows_read["events"] = events_table.num_rows

            if validate_data:
                events_validation = validate_and_normalize_events(events_table)
                validation_results["events"] = events_validation
                events_table = events_validation.valid_table
                rows_quarantined["events"] = events_validation.rows_quarantined

            output_path = parquet_writer.write_events(events_table)
            rows_written["events"] = events_table.num_rows
            parquet_files.append(str(output_path))
        else:
            if require_events_for_tags:
                raise SchemaValidationError(
                    f"Events CSV not found at {events_csv}. "
                    "Events are required for deriving market tags. "
                    "Either provide events.csv or set require_events_for_tags=False"
                )
            logger.info("events_csv_not_found", path=str(events_csv))
            rows_read["events"] = 0
            rows_written["events"] = 0
            rows_quarantined["events"] = 0

        # Process markets
        if markets_csv.exists():
            logger.info("processing_markets", path=str(markets_csv))
            markets_table = _read_csv_with_schema(
                markets_csv, MARKETS_SCHEMA, MARKETS_COLUMN_MAPPING
            )
            rows_read["markets"] = markets_table.num_rows

            if validate_data:
                markets_validation = validate_and_normalize_markets(markets_table)
                validation_results["markets"] = markets_validation
                markets_table = markets_validation.valid_table
                rows_quarantined["markets"] = markets_validation.rows_quarantined

            # Validate schema requirements
            validate_schema_requirements(
                markets_table=markets_table,
                events_table=events_table,
                require_events_for_tags=require_events_for_tags,
            )

            # Join events tags to markets if events are available
            if events_table is not None and events_table.num_rows > 0:
                logger.info("joining_events_tags_to_markets")
                markets_table = _join_events_tags_to_markets(markets_table, events_table)

            output_path = parquet_writer.write_markets(markets_table)
            rows_written["markets"] = markets_table.num_rows
            parquet_files.append(str(output_path))
        else:
            logger.warning("csv_not_found", path=str(markets_csv))
            rows_read["markets"] = 0
            rows_written["markets"] = 0
            rows_quarantined["markets"] = 0

        # Process trades
        if trades_csv.exists():
            logger.info("processing_trades", path=str(trades_csv))
            trades_table = _read_csv_with_schema(
                trades_csv, TRADES_SCHEMA, TRADES_COLUMN_MAPPING
            )
            rows_read["trades"] = trades_table.num_rows

            if validate_data:
                trades_validation = validate_and_normalize_trades(
                    trades_table, normalize_addresses=normalize_addresses
                )
                validation_results["trades"] = trades_validation
                trades_table = trades_validation.valid_table
                rows_quarantined["trades"] = trades_validation.rows_quarantined

            output_path = parquet_writer.write_trades(trades_table)
            rows_written["trades"] = trades_table.num_rows
            parquet_files.append(str(output_path))
        else:
            logger.warning("csv_not_found", path=str(trades_csv))
            rows_read["trades"] = 0
            rows_written["trades"] = 0
            rows_quarantined["trades"] = 0

        # Process order filled
        if order_filled_csv.exists():
            logger.info("processing_order_filled", path=str(order_filled_csv))
            order_filled_table = _read_csv_with_schema(
                order_filled_csv, ORDER_FILLED_SCHEMA, ORDER_FILLED_COLUMN_MAPPING
            )
            rows_read["order_filled"] = order_filled_table.num_rows

            if validate_data:
                order_filled_validation = validate_and_normalize_order_filled(
                    order_filled_table, normalize_addresses=normalize_addresses
                )
                validation_results["order_filled"] = order_filled_validation
                order_filled_table = order_filled_validation.valid_table
                rows_quarantined["order_filled"] = order_filled_validation.rows_quarantined

            output_path = parquet_writer.write_order_filled(order_filled_table)
            rows_written["order_filled"] = order_filled_table.num_rows
            parquet_files.append(str(output_path))
        else:
            logger.warning("csv_not_found", path=str(order_filled_csv))
            rows_read["order_filled"] = 0
            rows_written["order_filled"] = 0
            rows_quarantined["order_filled"] = 0

        # Create DuckDB views
        logger.info("creating_duckdb_views", partitioned=partitioning_enabled)
        duckdb_layer = DuckDBLayer(
            duckdb_path, parquet_dir, partitioned=partitioning_enabled
        )
        try:
            created_views = duckdb_layer.create_views()
            view_counts = duckdb_layer.verify_views()
            logger.info("views_created", views=created_views, counts=view_counts)
        finally:
            duckdb_layer.close()

        # Calculate duration
        duration = perf_counter() - timer_start
        end_time = datetime.now(timezone.utc)

        # Update run record with success
        run_record.end_time = end_time
        run_record.status = "completed"
        run_record.rows_read = rows_read
        run_record.rows_written = rows_written
        run_record.duration_seconds = duration
        metadata_store.update_run(run_record)

        # Set watermarks based on max timestamps
        if rows_written.get("trades", 0) > 0:
            metadata_store.set_watermark("trades", {"bootstrap_completed": end_time.isoformat()})
        if rows_written.get("markets", 0) > 0:
            metadata_store.set_watermark("markets", {"bootstrap_completed": end_time.isoformat()})
        if rows_written.get("order_filled", 0) > 0:
            metadata_store.set_watermark(
                "order_filled", {"bootstrap_completed": end_time.isoformat()}
            )
        if rows_written.get("events", 0) > 0:
            metadata_store.set_watermark("events", {"bootstrap_completed": end_time.isoformat()})

        # Log quarantine summary if any rows were quarantined
        total_quarantined = sum(rows_quarantined.values())
        if total_quarantined > 0:
            logger.warning(
                "bootstrap_rows_quarantined",
                run_id=run_id,
                rows_quarantined=rows_quarantined,
                total_quarantined=total_quarantined,
            )

        logger.info(
            "bootstrap_completed",
            run_id=run_id,
            duration_seconds=duration,
            rows_read=rows_read,
            rows_written=rows_written,
            rows_quarantined=rows_quarantined,
        )

        return BootstrapSummary(
            run_id=run_id,
            status="completed",
            duration_seconds=duration,
            markets_rows=rows_written.get("markets", 0),
            trades_rows=rows_written.get("trades", 0),
            order_filled_rows=rows_written.get("order_filled", 0),
            events_rows=rows_written.get("events", 0),
            schema_version="1.0.0",
            parquet_files=parquet_files,
            rows_quarantined=rows_quarantined,
        )

    except Exception as e:
        # Update run record with failure
        duration = perf_counter() - timer_start
        run_record.end_time = datetime.now(timezone.utc)
        run_record.status = "failed"
        run_record.rows_read = rows_read
        run_record.rows_written = rows_written
        run_record.error_message = str(e)
        run_record.duration_seconds = duration
        metadata_store.update_run(run_record)

        logger.error("bootstrap_failed", run_id=run_id, error=str(e))
        raise
