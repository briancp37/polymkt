"""Incremental update pipeline for appending new data since the last watermark."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import duckdb
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import structlog

from polymkt.config import settings
from polymkt.models.schemas import RunRecord, UpdateSummary
from polymkt.pipeline.normalize import (
    validate_and_normalize_markets,
    validate_and_normalize_order_filled,
    validate_and_normalize_trades,
)
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.parquet import (
    MARKETS_SCHEMA,
    ORDER_FILLED_SCHEMA,
    TRADES_SCHEMA,
)

logger = structlog.get_logger()

# Column mappings from CSV to our internal schema (same as bootstrap)
MARKETS_COLUMN_MAPPING = {
    "createdAt": "created_at",
    "closedTime": "closed_time",
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


def _get_existing_transaction_hashes(parquet_path: Path) -> set[str]:
    """Get set of existing transaction hashes from a Parquet file."""
    if not parquet_path.exists():
        return set()

    table = pq.read_table(parquet_path, columns=["transaction_hash"])
    return set(table.column("transaction_hash").to_pylist())


def _get_existing_market_ids(parquet_path: Path) -> set[str]:
    """Get set of existing market IDs from a Parquet file."""
    if not parquet_path.exists():
        return set()

    table = pq.read_table(parquet_path, columns=["id"])
    return set(table.column("id").to_pylist())


def _deduplicate_trades(
    new_table: pa.Table,
    existing_hashes: set[str],
) -> tuple[pa.Table, int]:
    """
    Remove trades with transaction_hash that already exist.

    Returns:
        Tuple of (deduplicated table, number of rows skipped)
    """
    if new_table.num_rows == 0:
        return new_table, 0

    if not existing_hashes:
        return new_table, 0

    # Create mask for rows NOT in existing hashes
    hashes = new_table.column("transaction_hash").to_pylist()
    mask = [h not in existing_hashes for h in hashes]

    # Count skipped rows
    skipped = sum(1 for m in mask if not m)

    # Filter using mask (explicitly typed as boolean)
    mask_array = pa.array(mask, type=pa.bool_())
    deduplicated = new_table.filter(mask_array)

    if skipped > 0:
        logger.info("dedupe_skipped", entity="trades", skipped=skipped)

    return deduplicated, skipped


def _deduplicate_order_filled(
    new_table: pa.Table,
    existing_hashes: set[str],
) -> tuple[pa.Table, int]:
    """
    Remove order_filled events with transaction_hash that already exist.

    Returns:
        Tuple of (deduplicated table, number of rows skipped)
    """
    if new_table.num_rows == 0:
        return new_table, 0

    if not existing_hashes:
        return new_table, 0

    # Create mask for rows NOT in existing hashes
    hashes = new_table.column("transaction_hash").to_pylist()
    mask = [h not in existing_hashes for h in hashes]

    # Count skipped rows
    skipped = sum(1 for m in mask if not m)

    # Filter using mask (explicitly typed as boolean)
    mask_array = pa.array(mask, type=pa.bool_())
    deduplicated = new_table.filter(mask_array)

    if skipped > 0:
        logger.info("dedupe_skipped", entity="order_filled", skipped=skipped)

    return deduplicated, skipped


def _create_empty_table_like(table: pa.Table) -> pa.Table:
    """Create an empty table with the same schema as the input table."""
    return table.slice(0, 0)


def _deduplicate_markets(
    new_table: pa.Table,
    existing_ids: set[str],
) -> tuple[pa.Table, pa.Table, int]:
    """
    Separate markets into new and updates based on existing IDs.

    Returns:
        Tuple of (new markets table, updated markets table, num_updates)
    """
    if new_table.num_rows == 0:
        return new_table, _create_empty_table_like(new_table), 0

    if not existing_ids:
        return new_table, _create_empty_table_like(new_table), 0

    # Create masks
    ids = new_table.column("id").to_pylist()
    is_new = [id not in existing_ids for id in ids]
    is_update = [id in existing_ids for id in ids]

    # Count updates
    num_updates = sum(is_update)

    # Filter using masks (explicitly typed as boolean)
    new_markets = new_table.filter(pa.array(is_new, type=pa.bool_()))
    updated_markets = new_table.filter(pa.array(is_update, type=pa.bool_()))

    if num_updates > 0:
        logger.info("markets_updates_detected", count=num_updates)

    return new_markets, updated_markets, num_updates


def _append_to_parquet(
    existing_path: Path,
    new_table: pa.Table,
    schema: pa.Schema,
) -> int:
    """
    Append new rows to existing Parquet file.

    If the file doesn't exist, creates a new one.
    Returns number of rows written.
    """
    if new_table.num_rows == 0:
        return 0

    new_table = new_table.cast(schema)

    if existing_path.exists():
        # Read existing data
        existing_table = pq.read_table(existing_path)
        # Concatenate
        combined = pa.concat_tables([existing_table, new_table])
        # Overwrite with combined data
        pq.write_table(combined, existing_path, compression="zstd")
        logger.info(
            "parquet_appended",
            path=str(existing_path),
            existing_rows=existing_table.num_rows,
            new_rows=new_table.num_rows,
            total_rows=combined.num_rows,
        )
    else:
        # Create new file
        pq.write_table(new_table, existing_path, compression="zstd")
        logger.info(
            "parquet_created",
            path=str(existing_path),
            rows=new_table.num_rows,
        )

    return int(new_table.num_rows)


def _upsert_markets_parquet(
    existing_path: Path,
    new_markets: pa.Table,
    updated_markets: pa.Table,
) -> int:
    """
    Upsert markets: add new markets and update existing ones.

    Uses DuckDB to efficiently handle the upsert logic.
    Returns number of rows written (new + updated).
    """
    total_new = new_markets.num_rows if new_markets else 0
    total_updates = updated_markets.num_rows if updated_markets else 0

    if total_new == 0 and total_updates == 0:
        return 0

    if not existing_path.exists():
        # Just write new markets
        if total_new > 0:
            new_markets = new_markets.cast(MARKETS_SCHEMA)
            pq.write_table(new_markets, existing_path, compression="zstd")
        return total_new

    # Use DuckDB for efficient upsert
    conn = duckdb.connect(":memory:")

    try:
        # Load existing markets
        existing_table = pq.read_table(existing_path)
        conn.register("existing_markets", existing_table)

        if total_updates > 0:
            # Register updated markets
            conn.register("updated_markets", updated_markets)

            # Create merged table: updates replace existing by id
            conn.execute("""
                CREATE TABLE merged AS
                SELECT * FROM existing_markets e
                WHERE e.id NOT IN (SELECT id FROM updated_markets)
                UNION ALL
                SELECT * FROM updated_markets
            """)
        else:
            conn.execute("CREATE TABLE merged AS SELECT * FROM existing_markets")

        # Add new markets if any
        if total_new > 0:
            conn.register("new_markets", new_markets)
            conn.execute("INSERT INTO merged SELECT * FROM new_markets")

        # Export to Parquet
        result = conn.execute("SELECT * FROM merged").fetch_arrow_table()
        result = result.cast(MARKETS_SCHEMA)
        pq.write_table(result, existing_path, compression="zstd")

        logger.info(
            "markets_upserted",
            path=str(existing_path),
            existing_rows=existing_table.num_rows,
            new_rows=total_new,
            updated_rows=total_updates,
            final_rows=result.num_rows,
        )

    finally:
        conn.close()

    return total_new + total_updates


def _filter_new_data_by_watermark(
    table: pa.Table,
    watermark: dict[str, str] | None,
    timestamp_column: str = "timestamp",
) -> pa.Table:
    """
    Filter table to only include rows after the watermark timestamp.

    If watermark is None or doesn't have last_timestamp, returns all rows.
    """
    if watermark is None:
        return table

    last_timestamp_str = watermark.get("last_timestamp")
    if not last_timestamp_str:
        return table

    # Parse the watermark timestamp
    last_timestamp = datetime.fromisoformat(last_timestamp_str.replace("Z", "+00:00"))

    # Convert to PyArrow timestamp for comparison
    ts_scalar = pa.scalar(last_timestamp, type=pa.timestamp("us", tz="UTC"))

    # Filter rows where timestamp > watermark
    import pyarrow.compute as pc
    mask = pc.greater(table.column(timestamp_column), ts_scalar)
    filtered = table.filter(mask)

    logger.info(
        "watermark_filter_applied",
        column=timestamp_column,
        watermark=last_timestamp_str,
        input_rows=table.num_rows,
        output_rows=filtered.num_rows,
    )

    return filtered


def _get_max_timestamp(table: pa.Table, timestamp_column: str = "timestamp") -> str | None:
    """Get the maximum timestamp from a table as ISO string."""
    if table.num_rows == 0:
        return None

    import pyarrow.compute as pc
    max_ts = pc.max(table.column(timestamp_column)).as_py()
    if max_ts is None:
        return None
    return str(max_ts.isoformat())


def run_update(
    markets_csv: Path | None = None,
    trades_csv: Path | None = None,
    order_filled_csv: Path | None = None,
    parquet_dir: Path | None = None,
    duckdb_path: Path | None = None,
    metadata_db_path: Path | None = None,
    normalize_addresses: bool = True,
    validate_data: bool = True,
) -> "UpdateSummary":
    """
    Run an incremental update using the last watermark.

    This pipeline:
    1. Reads the current watermark for each entity
    2. Reads new data from CSVs (simulating an upstream fetch)
    3. Filters data to only rows after the watermark timestamp
    4. Deduplicates using transaction_hash (for trades/order_filled) or id (for markets)
    5. Appends new rows to Parquet files (or upserts for markets)
    6. Updates the watermark to the max timestamp of new data
    7. Updates DuckDB views if needed

    The runtime is proportional to new data, not total history.

    Args:
        markets_csv: Path to markets CSV file
        trades_csv: Path to trades CSV file
        order_filled_csv: Path to orderFilled CSV file
        parquet_dir: Directory for Parquet output
        duckdb_path: Path to DuckDB database file
        metadata_db_path: Path to metadata SQLite database
        normalize_addresses: Whether to normalize Ethereum addresses
        validate_data: Whether to validate and quarantine invalid rows

    Returns:
        UpdateSummary with run details
    """
    # Use settings defaults if not provided
    markets_csv = markets_csv or settings.markets_csv
    trades_csv = trades_csv or settings.trades_csv
    order_filled_csv = order_filled_csv or settings.order_filled_csv
    parquet_dir = parquet_dir or settings.parquet_dir
    duckdb_path = duckdb_path or settings.duckdb_path
    metadata_db_path = metadata_db_path or settings.metadata_db_path

    # Initialize components
    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    timer_start = perf_counter()

    metadata_store = MetadataStore(metadata_db_path)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Get current watermarks
    watermarks_before = metadata_store.get_all_watermarks()

    # Create initial run record
    run_record = RunRecord(
        run_id=run_id,
        run_type="update",
        start_time=start_time,
        status="running",
        schema_version="1.0.0",
        watermark_before=watermarks_before,
    )
    metadata_store.create_run(run_record)

    rows_read: dict[str, int] = {}
    rows_written: dict[str, int] = {}
    rows_skipped: dict[str, int] = {}
    rows_updated: dict[str, int] = {}
    rows_quarantined: dict[str, int] = {}
    watermarks_after: dict[str, dict[str, str]] = {}

    try:
        # Process trades
        trades_parquet = parquet_dir / "trades.parquet"
        if trades_csv.exists():
            logger.info("processing_trades_update", path=str(trades_csv))

            # Read new data from CSV
            trades_table = _read_csv_with_schema(
                trades_csv, TRADES_SCHEMA, TRADES_COLUMN_MAPPING
            )
            rows_read["trades"] = trades_table.num_rows

            # Filter by watermark if available
            trades_watermark = metadata_store.get_watermark("trades")
            trades_table = _filter_new_data_by_watermark(trades_table, trades_watermark)

            # Validate and normalize
            if validate_data:
                trades_validation = validate_and_normalize_trades(
                    trades_table, normalize_addresses=normalize_addresses
                )
                trades_table = trades_validation.valid_table
                rows_quarantined["trades"] = trades_validation.rows_quarantined

            # Deduplicate against existing data
            existing_hashes = _get_existing_transaction_hashes(trades_parquet)
            trades_table, skipped = _deduplicate_trades(trades_table, existing_hashes)
            rows_skipped["trades"] = skipped

            # Append to Parquet
            written = _append_to_parquet(trades_parquet, trades_table, TRADES_SCHEMA)
            rows_written["trades"] = written

            # Update watermark if we wrote data
            if written > 0:
                max_ts = _get_max_timestamp(trades_table)
                if max_ts:
                    new_watermark = {"last_timestamp": max_ts}
                    metadata_store.set_watermark("trades", new_watermark)
                    watermarks_after["trades"] = new_watermark
            elif trades_watermark:
                watermarks_after["trades"] = trades_watermark

        else:
            logger.warning("csv_not_found", path=str(trades_csv))
            rows_read["trades"] = 0
            rows_written["trades"] = 0
            rows_skipped["trades"] = 0
            rows_quarantined["trades"] = 0

        # Process markets
        markets_parquet = parquet_dir / "markets.parquet"
        if markets_csv.exists():
            logger.info("processing_markets_update", path=str(markets_csv))

            # Read all markets (markets are upserted, not just appended)
            markets_table = _read_csv_with_schema(
                markets_csv, MARKETS_SCHEMA, MARKETS_COLUMN_MAPPING
            )
            rows_read["markets"] = markets_table.num_rows

            # Filter by watermark for markets using created_at
            markets_watermark = metadata_store.get_watermark("markets")
            markets_table = _filter_new_data_by_watermark(
                markets_table, markets_watermark, timestamp_column="created_at"
            )

            # Validate
            if validate_data:
                markets_validation = validate_and_normalize_markets(markets_table)
                markets_table = markets_validation.valid_table
                rows_quarantined["markets"] = markets_validation.rows_quarantined

            # Separate into new and updates
            existing_ids = _get_existing_market_ids(markets_parquet)
            new_markets, updated_markets, num_updates = _deduplicate_markets(
                markets_table, existing_ids
            )
            rows_skipped["markets"] = 0  # Markets use upsert, not skip
            rows_updated["markets"] = num_updates

            # Upsert markets
            written = _upsert_markets_parquet(
                markets_parquet, new_markets, updated_markets
            )
            rows_written["markets"] = new_markets.num_rows  # Only count new rows as "written"

            # Update watermark
            if markets_table.num_rows > 0:
                max_ts = _get_max_timestamp(markets_table, "created_at")
                if max_ts:
                    new_watermark = {"last_timestamp": max_ts}
                    metadata_store.set_watermark("markets", new_watermark)
                    watermarks_after["markets"] = new_watermark
            elif markets_watermark:
                watermarks_after["markets"] = markets_watermark

        else:
            logger.warning("csv_not_found", path=str(markets_csv))
            rows_read["markets"] = 0
            rows_written["markets"] = 0
            rows_skipped["markets"] = 0
            rows_updated["markets"] = 0
            rows_quarantined["markets"] = 0

        # Process order filled
        order_filled_parquet = parquet_dir / "order_filled.parquet"
        if order_filled_csv.exists():
            logger.info("processing_order_filled_update", path=str(order_filled_csv))

            # Read new data from CSV
            order_filled_table = _read_csv_with_schema(
                order_filled_csv, ORDER_FILLED_SCHEMA, ORDER_FILLED_COLUMN_MAPPING
            )
            rows_read["order_filled"] = order_filled_table.num_rows

            # Filter by watermark
            order_filled_watermark = metadata_store.get_watermark("order_filled")
            order_filled_table = _filter_new_data_by_watermark(
                order_filled_table, order_filled_watermark
            )

            # Validate and normalize
            if validate_data:
                order_filled_validation = validate_and_normalize_order_filled(
                    order_filled_table, normalize_addresses=normalize_addresses
                )
                order_filled_table = order_filled_validation.valid_table
                rows_quarantined["order_filled"] = order_filled_validation.rows_quarantined

            # Deduplicate against existing data
            existing_hashes = _get_existing_transaction_hashes(order_filled_parquet)
            order_filled_table, skipped = _deduplicate_order_filled(
                order_filled_table, existing_hashes
            )
            rows_skipped["order_filled"] = skipped

            # Append to Parquet
            written = _append_to_parquet(
                order_filled_parquet, order_filled_table, ORDER_FILLED_SCHEMA
            )
            rows_written["order_filled"] = written

            # Update watermark
            if written > 0:
                max_ts = _get_max_timestamp(order_filled_table)
                if max_ts:
                    new_watermark = {"last_timestamp": max_ts}
                    metadata_store.set_watermark("order_filled", new_watermark)
                    watermarks_after["order_filled"] = new_watermark
            elif order_filled_watermark:
                watermarks_after["order_filled"] = order_filled_watermark

        else:
            logger.warning("csv_not_found", path=str(order_filled_csv))
            rows_read["order_filled"] = 0
            rows_written["order_filled"] = 0
            rows_skipped["order_filled"] = 0
            rows_quarantined["order_filled"] = 0

        # Refresh DuckDB views if any data was written
        total_written = sum(rows_written.values())
        if total_written > 0:
            logger.info("refreshing_duckdb_views")
            duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
            try:
                duckdb_layer.create_views()
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
        run_record.watermark_after = watermarks_after
        run_record.duration_seconds = duration
        metadata_store.update_run(run_record)

        logger.info(
            "update_completed",
            run_id=run_id,
            duration_seconds=duration,
            rows_read=rows_read,
            rows_written=rows_written,
            rows_skipped=rows_skipped,
            rows_updated=rows_updated,
        )

        return UpdateSummary(
            run_id=run_id,
            status="completed",
            duration_seconds=duration,
            rows_read=rows_read,
            rows_written=rows_written,
            rows_skipped=rows_skipped,
            rows_updated=rows_updated,
            rows_quarantined=rows_quarantined,
            watermark_before=watermarks_before,
            watermark_after=watermarks_after,
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

        logger.error("update_failed", run_id=run_id, error=str(e))
        raise
