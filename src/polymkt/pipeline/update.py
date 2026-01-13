"""Incremental update pipeline for appending new data since the last watermark.

This module implements poly_data-like incremental update logic:
- Watermark-based filtering to fetch only new data
- Deduplication using transaction_hash for trades/orders
- Upsert logic for markets with events tag refresh
- Events update module for refreshing event tags
- Resumable pipeline that can recover from interruption
- Structured logging with run_id for traceability
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import structlog

from polymkt.config import settings
from polymkt.logging import create_run_logger
from polymkt.models.schemas import RunRecord, UpdateSummary
from polymkt.pipeline.normalize import (
    normalize_timestamp,
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
)

# Module-level logger for functions outside of run_update
logger = structlog.get_logger()


class PipelineError(Exception):
    """Base class for pipeline errors with remediation guidance."""

    def __init__(
        self,
        message: str,
        remediation: str,
        entity: str | None = None,
        is_retryable: bool = False,
    ):
        """
        Initialize a pipeline error with remediation guidance.

        Args:
            message: Error description
            remediation: Actionable steps to fix the issue
            entity: Entity being processed (trades, markets, etc.)
            is_retryable: Whether the operation can be safely retried
        """
        super().__init__(message)
        self.message = message
        self.remediation = remediation
        self.entity = entity
        self.is_retryable = is_retryable

    def __str__(self) -> str:
        parts = [self.message]
        if self.entity:
            parts.append(f"Entity: {self.entity}")
        parts.append(f"Remediation: {self.remediation}")
        if self.is_retryable:
            parts.append("This operation can be safely retried.")
        return " | ".join(parts)


class DataSourceError(PipelineError):
    """Error when source data is unavailable or invalid."""

    def __init__(self, message: str, entity: str, path: str):
        super().__init__(
            message=message,
            remediation=f"Verify the source file exists at '{path}' and is readable. "
            "Check file permissions and ensure the file is not corrupted.",
            entity=entity,
            is_retryable=True,
        )
        self.path = path


class DataValidationError(PipelineError):
    """Error when data fails validation."""

    def __init__(self, message: str, entity: str, invalid_count: int):
        super().__init__(
            message=message,
            remediation=f"Review the source data for {entity}. {invalid_count} rows failed validation. "
            "Check for missing required fields, invalid timestamps, or out-of-range values. "
            "Invalid rows are quarantined and logged for review.",
            entity=entity,
            is_retryable=False,
        )
        self.invalid_count = invalid_count


class WatermarkError(PipelineError):
    """Error when watermark state is inconsistent."""

    def __init__(self, message: str, entity: str):
        super().__init__(
            message=message,
            remediation=f"The watermark for {entity} may be corrupted. "
            "Options: 1) Run a full bootstrap to reset state, or "
            "2) Manually clear the watermark via the metadata API.",
            entity=entity,
            is_retryable=False,
        )

# Column mappings from CSV to our internal schema (same as bootstrap)
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

EVENTS_COLUMN_MAPPING = {
    "eventId": "event_id",
    "createdAt": "created_at",
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


def _get_existing_event_ids(parquet_path: Path) -> set[str]:
    """Get set of existing event IDs from a Parquet file."""
    if not parquet_path.exists():
        return set()

    table = pq.read_table(parquet_path, columns=["event_id"])
    return set(table.column("event_id").to_pylist())


def _read_events_csv(csv_path: Path) -> pa.Table:
    """
    Read events CSV with special handling for tags column (JSON list).

    Events CSV format expected:
    eventId,tags,title,description,createdAt
    evt_123,"[""tag1"",""tag2""]",My Event,Description,2025-01-01 00:00:00

    This is modeled after poly_data/poly_utils/update_markets for event retrieval.
    """
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
        processed_row["created_at"] = normalize_timestamp(
            row.get("created_at"), "created_at"
        )

        processed_rows.append(processed_row)

    # Build PyArrow table with proper schema
    return pa.Table.from_pylist(processed_rows, schema=EVENTS_SCHEMA)


def _deduplicate_events(
    new_table: pa.Table,
    existing_ids: set[str],
) -> tuple[pa.Table, pa.Table, int]:
    """
    Separate events into new and updates based on existing IDs.

    Returns:
        Tuple of (new events table, updated events table, num_updates)
    """
    if new_table.num_rows == 0:
        return new_table, _create_empty_table_like(new_table), 0

    if not existing_ids:
        return new_table, _create_empty_table_like(new_table), 0

    # Create masks
    ids = new_table.column("event_id").to_pylist()
    is_new = [id not in existing_ids for id in ids]
    is_update = [id in existing_ids for id in ids]

    # Count updates
    num_updates = sum(is_update)

    # Filter using masks (explicitly typed as boolean)
    new_events = new_table.filter(pa.array(is_new, type=pa.bool_()))
    updated_events = new_table.filter(pa.array(is_update, type=pa.bool_()))

    if num_updates > 0:
        logger.info("events_updates_detected", count=num_updates)

    return new_events, updated_events, num_updates


def _upsert_events_parquet(
    existing_path: Path,
    new_events: pa.Table,
    updated_events: pa.Table,
) -> int:
    """
    Upsert events: add new events and update existing ones.

    Uses DuckDB to efficiently handle the upsert logic.
    Returns number of rows written (new + updated).
    """
    total_new = new_events.num_rows if new_events else 0
    total_updates = updated_events.num_rows if updated_events else 0

    if total_new == 0 and total_updates == 0:
        return 0

    if not existing_path.exists():
        # Just write new events
        if total_new > 0:
            new_events = new_events.cast(EVENTS_SCHEMA)
            pq.write_table(new_events, existing_path, compression="zstd")
        return total_new

    # Use DuckDB for efficient upsert
    conn = duckdb.connect(":memory:")

    try:
        # Load existing events
        existing_table = pq.read_table(existing_path)
        conn.register("existing_events", existing_table)

        if total_updates > 0:
            # Register updated events
            conn.register("updated_events", updated_events)

            # Create merged table: updates replace existing by event_id
            conn.execute("""
                CREATE TABLE merged AS
                SELECT * FROM existing_events e
                WHERE e.event_id NOT IN (SELECT event_id FROM updated_events)
                UNION ALL
                SELECT * FROM updated_events
            """)
        else:
            conn.execute("CREATE TABLE merged AS SELECT * FROM existing_events")

        # Add new events if any
        if total_new > 0:
            conn.register("new_events", new_events)
            conn.execute("INSERT INTO merged SELECT * FROM new_events")

        # Export to Parquet
        result = conn.execute("SELECT * FROM merged").fetch_arrow_table()
        result = result.cast(EVENTS_SCHEMA)
        pq.write_table(result, existing_path, compression="zstd")

        logger.info(
            "events_upserted",
            path=str(existing_path),
            existing_rows=existing_table.num_rows,
            new_rows=total_new,
            updated_rows=total_updates,
            final_rows=result.num_rows,
        )

    finally:
        conn.close()

    return total_new + total_updates


def _join_events_tags_to_markets(
    markets_parquet: Path,
    events_parquet: Path,
) -> int:
    """
    Re-join event tags to markets via event_id, refreshing market tags.

    This is called after events are updated to propagate tag changes to markets.
    Markets' category and closedTime are preserved while tags are refreshed.

    Returns number of markets updated.
    """
    if not markets_parquet.exists() or not events_parquet.exists():
        logger.info(
            "skip_events_join_missing_files",
            markets_exists=markets_parquet.exists(),
            events_exists=events_parquet.exists(),
        )
        return 0

    logger.info(
        "refreshing_market_tags_from_events",
        markets_path=str(markets_parquet),
        events_path=str(events_parquet),
    )

    # Use DuckDB in-memory for efficient join
    conn = duckdb.connect(":memory:")

    try:
        # Load tables
        markets_table = pq.read_table(markets_parquet)
        events_table = pq.read_table(events_parquet)

        conn.register("markets", markets_table)
        conn.register("events", events_table)

        # Left join markets to events to get refreshed tags
        # Preserves category, closed_time, and all other market fields
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

        # Log unmapped markets
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
            )

        # Write updated markets back to Parquet
        result = result.cast(MARKETS_SCHEMA)
        pq.write_table(result, markets_parquet, compression="zstd")

        logger.info(
            "events_tags_refresh_complete",
            markets_updated=result.num_rows,
            unmapped_markets=unmapped_count,
        )

        return int(result.num_rows)

    finally:
        conn.close()


def _generate_trade_alerts(
    trades_table: pa.Table,
    metadata_store: MetadataStore,
    run_logger: Any,
) -> int:
    """
    Generate alerts for trades from watchlisted wallets.

    Checks each trade's maker and taker addresses against active watchlist
    subscriptions. Creates alerts for matching trades with deduplication
    via transaction_hash.

    Args:
        trades_table: PyArrow table of newly written trades
        metadata_store: MetadataStore instance for watchlist/alert operations
        run_logger: Structured logger with run_id context

    Returns:
        Number of alerts generated
    """
    if trades_table.num_rows == 0:
        return 0

    # Get active subscriptions with watchlist type
    subscriptions = metadata_store.list_alert_subscriptions(
        rule_type="watchlist_trade", is_active=True
    )

    if not subscriptions:
        return 0

    alerts_created = 0

    # Build mapping of watchlist_id -> watched addresses
    watchlist_addresses: dict[str, set[str]] = {}
    for sub in subscriptions:
        if sub.watchlist_id and sub.watchlist_id not in watchlist_addresses:
            items = metadata_store.list_watchlist_items(sub.watchlist_id)
            watchlist_addresses[sub.watchlist_id] = {
                item.wallet_address.lower() for item in items
            }

    # Extract trade data for checking
    maker_col = trades_table.column("maker").to_pylist()
    taker_col = trades_table.column("taker").to_pylist()
    tx_hash_col = trades_table.column("transaction_hash").to_pylist()
    market_id_col = trades_table.column("market_id").to_pylist()
    timestamp_col = trades_table.column("timestamp").to_pylist()
    side_col = trades_table.column("side").to_pylist()
    size_col = trades_table.column("size").to_pylist()
    price_col = trades_table.column("price").to_pylist()

    for i in range(trades_table.num_rows):
        maker = maker_col[i].lower() if maker_col[i] else ""
        taker = taker_col[i].lower() if taker_col[i] else ""
        tx_hash = tx_hash_col[i]

        for sub in subscriptions:
            if not sub.watchlist_id:
                continue

            watched = watchlist_addresses.get(sub.watchlist_id, set())
            matched_address = None
            match_side = None

            if maker in watched:
                matched_address = maker
                match_side = "maker"
            elif taker in watched:
                matched_address = taker
                match_side = "taker"

            if matched_address:
                # Check for duplicate alert (same tx_hash and subscription)
                existing = metadata_store.list_alerts(
                    subscription_id=sub.id, limit=1000
                )
                is_duplicate = any(
                    a.trade_data and a.trade_data.get("transaction_hash") == tx_hash
                    for a in existing
                )

                if not is_duplicate:
                    trade_data = {
                        "transaction_hash": tx_hash,
                        "match_side": match_side,
                        "timestamp": str(timestamp_col[i]),
                        "side": side_col[i],
                        "size": str(size_col[i]),
                        "price": str(price_col[i]),
                    }
                    # Use tx_hash as event_id for deduplication
                    metadata_store.create_alert(
                        subscription_id=sub.id,
                        event_id=tx_hash,
                        market_id=market_id_col[i],
                        wallet_address=matched_address,
                        trade_data=trade_data,
                    )
                    alerts_created += 1

    if alerts_created > 0:
        run_logger.info("trade_alerts_generated", count=alerts_created)

    return alerts_created


def run_update(
    markets_csv: Path | None = None,
    trades_csv: Path | None = None,
    order_filled_csv: Path | None = None,
    events_csv: Path | None = None,
    parquet_dir: Path | None = None,
    duckdb_path: Path | None = None,
    metadata_db_path: Path | None = None,
    normalize_addresses: bool = True,
    validate_data: bool = True,
) -> "UpdateSummary":
    """
    Run an incremental update using the last watermark.

    This pipeline implements poly_data-like incremental update logic:
    1. Reads the current watermark for each entity
    2. Reads new data from CSVs (simulating an upstream fetch like poly_data)
    3. Filters data to only rows after the watermark timestamp
    4. Deduplicates using transaction_hash (for trades/order_filled) or id (for markets/events)
    5. Appends new rows to Parquet files (or upserts for markets/events)
    6. Refreshes market tags by re-joining with updated events
    7. Updates the watermark to the max timestamp of new data
    8. Updates DuckDB views if needed

    The runtime is proportional to new data, not total history.
    The pipeline is resumable - if interrupted, it can be re-run without duplicating data.

    Args:
        markets_csv: Path to markets CSV file
        trades_csv: Path to trades CSV file
        order_filled_csv: Path to orderFilled CSV file
        events_csv: Path to events CSV file (for updating event tags)
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
    events_csv = events_csv or settings.events_csv
    parquet_dir = parquet_dir or settings.parquet_dir
    duckdb_path = duckdb_path or settings.duckdb_path
    metadata_db_path = metadata_db_path or settings.metadata_db_path

    # Initialize components
    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    timer_start = perf_counter()

    # Create bound logger with run_id for all log entries
    run_logger = create_run_logger(run_id=run_id, operation="update")
    run_logger.info(
        "update_started",
        markets_csv=str(markets_csv),
        trades_csv=str(trades_csv),
        order_filled_csv=str(order_filled_csv),
        events_csv=str(events_csv),
        parquet_dir=str(parquet_dir),
    )

    metadata_store = MetadataStore(metadata_db_path)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Get current watermarks
    watermarks_before = metadata_store.get_all_watermarks()
    run_logger.info("watermarks_loaded", watermarks=watermarks_before)

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
            run_logger.info("processing_trades_update", path=str(trades_csv))

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

            # Generate alerts for watchlisted wallet trades
            _generate_trade_alerts(trades_table, metadata_store, run_logger)

        else:
            run_logger.warning("csv_not_found", entity="trades", path=str(trades_csv))
            rows_read["trades"] = 0
            rows_written["trades"] = 0
            rows_skipped["trades"] = 0
            rows_quarantined["trades"] = 0

        # Process markets
        markets_parquet = parquet_dir / "markets.parquet"
        if markets_csv.exists():
            run_logger.info("processing_markets_update", path=str(markets_csv))

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
            run_logger.warning("csv_not_found", entity="markets", path=str(markets_csv))
            rows_read["markets"] = 0
            rows_written["markets"] = 0
            rows_skipped["markets"] = 0
            rows_updated["markets"] = 0
            rows_quarantined["markets"] = 0

        # Process order filled
        order_filled_parquet = parquet_dir / "order_filled.parquet"
        if order_filled_csv.exists():
            run_logger.info("processing_order_filled_update", path=str(order_filled_csv))

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
            run_logger.warning("csv_not_found", entity="order_filled", path=str(order_filled_csv))
            rows_read["order_filled"] = 0
            rows_written["order_filled"] = 0
            rows_skipped["order_filled"] = 0
            rows_quarantined["order_filled"] = 0

        # Process events (poly_data-like event tags update)
        # This is modeled after poly_data/poly_utils/update_markets for event retrieval
        events_parquet = parquet_dir / "events.parquet"
        markets_parquet = parquet_dir / "markets.parquet"
        events_updated = False

        if events_csv.exists():
            run_logger.info("processing_events_update", path=str(events_csv))

            # Read events from CSV with special JSON tags handling
            events_table = _read_events_csv(events_csv)
            rows_read["events"] = events_table.num_rows

            # Filter by watermark for events using created_at
            events_watermark = metadata_store.get_watermark("events")
            events_table = _filter_new_data_by_watermark(
                events_table, events_watermark, timestamp_column="created_at"
            )

            # Validate events
            if validate_data:
                events_validation = validate_and_normalize_events(events_table)
                events_table = events_validation.valid_table
                rows_quarantined["events"] = events_validation.rows_quarantined

            # Separate into new and updates
            existing_ids = _get_existing_event_ids(events_parquet)
            new_events, updated_events, num_updates = _deduplicate_events(
                events_table, existing_ids
            )
            rows_skipped["events"] = 0  # Events use upsert, not skip
            rows_updated["events"] = num_updates

            # Upsert events
            written = _upsert_events_parquet(
                events_parquet, new_events, updated_events
            )
            rows_written["events"] = new_events.num_rows  # Only count new rows as "written"

            # Mark that events were updated (for market tags refresh)
            if written > 0 or num_updates > 0:
                events_updated = True

            # Update watermark
            if events_table.num_rows > 0:
                max_ts = _get_max_timestamp(events_table, "created_at")
                if max_ts:
                    new_watermark = {"last_timestamp": max_ts}
                    metadata_store.set_watermark("events", new_watermark)
                    watermarks_after["events"] = new_watermark
            elif events_watermark:
                watermarks_after["events"] = events_watermark

        else:
            run_logger.info("events_csv_not_found", path=str(events_csv))
            rows_read["events"] = 0
            rows_written["events"] = 0
            rows_skipped["events"] = 0
            rows_updated["events"] = 0
            rows_quarantined["events"] = 0

        # Re-join events tags to markets if events were updated
        # This ensures markets.tags is refreshed via the events join
        # Markets' category and closedTime are preserved
        if events_updated and events_parquet.exists() and markets_parquet.exists():
            run_logger.info("refreshing_market_tags_after_events_update")
            _join_events_tags_to_markets(markets_parquet, events_parquet)

        # Refresh DuckDB views if any data was written
        total_written = sum(rows_written.values())
        if total_written > 0:
            run_logger.info("refreshing_duckdb_views")
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

        run_logger.info(
            "update_completed",
            duration_seconds=duration,
            rows_read=rows_read,
            rows_written=rows_written,
            rows_skipped=rows_skipped,
            rows_updated=rows_updated,
            watermarks_after=watermarks_after,
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

    except PipelineError as e:
        # Handle pipeline-specific errors with actionable remediation
        duration = perf_counter() - timer_start
        run_record.end_time = datetime.now(timezone.utc)
        run_record.status = "failed"
        run_record.rows_read = rows_read
        run_record.rows_written = rows_written
        run_record.error_message = str(e)
        run_record.duration_seconds = duration
        metadata_store.update_run(run_record)

        run_logger.error(
            "update_failed",
            error_type=type(e).__name__,
            error_message=e.message,
            entity=e.entity,
            remediation=e.remediation,
            is_retryable=e.is_retryable,
            rows_written_before_failure=rows_written,
            watermarks_before=watermarks_before,
        )
        raise

    except Exception as e:
        # Handle unexpected errors with general remediation guidance
        duration = perf_counter() - timer_start
        run_record.end_time = datetime.now(timezone.utc)
        run_record.status = "failed"
        run_record.rows_read = rows_read
        run_record.rows_written = rows_written
        run_record.error_message = str(e)
        run_record.duration_seconds = duration
        metadata_store.update_run(run_record)

        # Provide actionable remediation for unexpected errors
        remediation = (
            "The pipeline can be safely re-run due to watermark-based deduplication. "
            "Partial data may have been written. Check the error message and stack trace for details. "
            "If the error persists, check: 1) Source CSV files exist and are readable, "
            "2) Parquet output directory is writable, 3) Sufficient disk space available."
        )

        run_logger.error(
            "update_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            remediation=remediation,
            is_retryable=True,
            rows_written_before_failure=rows_written,
            watermarks_before=watermarks_before,
        )
        raise
