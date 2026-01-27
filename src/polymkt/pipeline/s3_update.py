"""S3 partitioned update pipeline for incremental data updates.

This module implements partitioned S3 writes for scalable incremental updates:
- Trades and order_filled are partitioned by year/month/day
- Markets and events are small files that get rewritten
- Watermarks track the last processed timestamp
- New data is written as separate partition files (no rewriting large files)

S3 Structure:
    s3://bucket/prefix/
    ├── trades/year=YYYY/month=MM/day=DD/trades_YYYYMMDD_HHMMSS.parquet
    ├── order_filled/year=YYYY/month=MM/day=DD/order_filled_YYYYMMDD_HHMMSS.parquet
    ├── markets.parquet
    └── events.parquet
"""

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import structlog

from polymkt.config import settings
from polymkt.models.schemas import RunRecord
from polymkt.pipeline.normalize import (
    normalize_timestamp,
    validate_and_normalize_events,
    validate_and_normalize_markets,
    validate_and_normalize_order_filled,
    validate_and_normalize_trades,
)
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.parquet import (
    EVENTS_SCHEMA,
    MARKETS_SCHEMA,
    ORDER_FILLED_SCHEMA,
    TRADES_SCHEMA,
)

logger = structlog.get_logger()


# Column mappings from CSV to internal schema
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

MARKETS_COLUMN_MAPPING = {
    "createdAt": "created_at",
    "closedTime": "closed_time",
    "eventId": "event_id",
}

EVENTS_COLUMN_MAPPING = {
    "eventId": "event_id",
    "createdAt": "created_at",
}


def _get_s3_filesystem(region: str = "us-east-1") -> pafs.S3FileSystem:
    """Create a PyArrow S3FileSystem."""
    return pafs.S3FileSystem(region=region)


def _rename_columns(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    """Rename columns in a PyArrow table."""
    new_names = [mapping.get(name, name) for name in table.column_names]
    return table.rename_columns(new_names)


def _read_csv_with_mapping(csv_path: Path, column_mapping: dict[str, str]) -> pa.Table:
    """Read CSV and apply column renaming."""
    parse_options = csv.ParseOptions(newlines_in_values=True)
    convert_options = csv.ConvertOptions(timestamp_parsers=[])

    table = csv.read_csv(
        csv_path,
        parse_options=parse_options,
        convert_options=convert_options,
    )
    return _rename_columns(table, column_mapping)


def _filter_by_watermark(
    table: pa.Table,
    watermark: dict[str, str] | None,
    timestamp_column: str = "timestamp",
) -> pa.Table:
    """Filter table to rows after the watermark timestamp."""
    if watermark is None:
        return table

    last_ts_str = watermark.get("last_timestamp")
    if not last_ts_str:
        return table

    # Parse watermark timestamp
    last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))

    # Get timestamp column and filter
    if timestamp_column not in table.column_names:
        logger.warning("timestamp_column_missing", column=timestamp_column)
        return table

    ts_col = table.column(timestamp_column)

    # Convert to comparable format
    mask = []
    for i in range(ts_col.length()):
        val = ts_col[i].as_py()
        if val is None:
            mask.append(False)
        elif isinstance(val, datetime):
            mask.append(val > last_ts)
        else:
            # Try to parse string timestamp
            parsed = normalize_timestamp(val, timestamp_column)
            mask.append(parsed is not None and parsed > last_ts)

    mask_array = pa.array(mask)
    filtered = table.filter(mask_array)

    logger.info(
        "watermark_filter_applied",
        original_rows=table.num_rows,
        filtered_rows=filtered.num_rows,
        watermark=last_ts_str,
    )

    return filtered


def _partition_by_date(
    table: pa.Table,
    timestamp_column: str = "timestamp",
) -> dict[tuple[int, int, int], pa.Table]:
    """
    Partition a table by year/month/day based on timestamp column.

    Returns:
        Dict mapping (year, month, day) to table subset
    """
    if table.num_rows == 0:
        return {}

    ts_col = table.column(timestamp_column)

    # Group rows by date
    date_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for i in range(ts_col.length()):
        val = ts_col[i].as_py()
        if val is None:
            # Use epoch for null timestamps
            date_key = (1970, 1, 1)
        elif isinstance(val, datetime):
            date_key = (val.year, val.month, val.day)
        else:
            # Try to parse
            parsed = normalize_timestamp(val, timestamp_column)
            if parsed:
                date_key = (parsed.year, parsed.month, parsed.day)
            else:
                date_key = (1970, 1, 1)

        date_indices[date_key].append(i)

    # Create subtables for each date
    partitions = {}
    for date_key, indices in date_indices.items():
        indices_array = pa.array(indices)
        partitions[date_key] = table.take(indices_array)

    logger.info(
        "table_partitioned",
        total_rows=table.num_rows,
        num_partitions=len(partitions),
        partitions=[(k, v.num_rows) for k, v in partitions.items()],
    )

    return partitions


def _write_partitioned_to_s3(
    partitions: dict[tuple[int, int, int], pa.Table],
    s3_bucket: str,
    s3_prefix: str,
    entity_name: str,
    schema: pa.Schema,
    s3_fs: pafs.S3FileSystem,
    run_id: str,
) -> list[str]:
    """
    Write partitioned tables to S3 with Hive-style partitioning.

    Path format: s3://bucket/prefix/entity/year=YYYY/month=MM/day=DD/entity_YYYYMMDD_runid.parquet
    """
    written_files = []

    for (year, month, day), table in partitions.items():
        if table.num_rows == 0:
            continue

        # Cast to target schema
        table = _cast_to_schema(table, schema)

        # Build S3 path with Hive-style partitioning
        partition_path = f"year={year}/month={month:02d}/day={day:02d}"
        filename = f"{entity_name}_{year}{month:02d}{day:02d}_{run_id[:8]}.parquet"
        s3_key = f"{s3_bucket}/{s3_prefix}/{entity_name}/{partition_path}/{filename}"

        # Write to S3
        pq.write_table(
            table,
            s3_key,
            filesystem=s3_fs,
            compression="zstd",
        )

        full_path = f"s3://{s3_key}"
        written_files.append(full_path)

        logger.info(
            "partition_written",
            entity=entity_name,
            partition=f"{year}-{month:02d}-{day:02d}",
            rows=table.num_rows,
            path=full_path,
        )

    return written_files


def _cast_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast table columns to match target schema."""
    columns = {}

    for field in schema:
        if field.name in table.column_names:
            col = table.column(field.name)
            try:
                # Handle timestamp columns specially
                if pa.types.is_timestamp(field.type):
                    values = []
                    for i in range(col.length()):
                        val = col[i].as_py()
                        values.append(normalize_timestamp(val, field.name))
                    col = pa.array(values, type=field.type)
                else:
                    col = col.cast(field.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                # Keep original if cast fails
                pass
            columns[field.name] = col
        else:
            # Create null column for missing fields
            columns[field.name] = pa.nulls(table.num_rows, type=field.type)

    return pa.table(columns, schema=schema)


def _write_full_to_s3(
    table: pa.Table,
    s3_bucket: str,
    s3_prefix: str,
    filename: str,
    schema: pa.Schema,
    s3_fs: pafs.S3FileSystem,
) -> str:
    """Write a full table to S3 (for small files like markets/events)."""
    table = _cast_to_schema(table, schema)

    s3_key = f"{s3_bucket}/{s3_prefix}/{filename}"

    pq.write_table(
        table,
        s3_key,
        filesystem=s3_fs,
        compression="zstd",
    )

    full_path = f"s3://{s3_key}"
    logger.info(
        "file_written",
        path=full_path,
        rows=table.num_rows,
    )

    return full_path


def _get_max_timestamp(table: pa.Table, timestamp_column: str = "timestamp") -> datetime | None:
    """Get the maximum timestamp from a table."""
    if table.num_rows == 0:
        return None

    if timestamp_column not in table.column_names:
        return None

    ts_col = table.column(timestamp_column)
    max_ts = None

    for i in range(ts_col.length()):
        val = ts_col[i].as_py()
        if val is None:
            continue

        if isinstance(val, datetime):
            ts = val
        else:
            ts = normalize_timestamp(val, timestamp_column)

        if ts and (max_ts is None or ts > max_ts):
            max_ts = ts

    return max_ts


def _process_events_tags(table: pa.Table) -> pa.Table:
    """Process tags column from JSON strings to list[str]."""
    import json

    if "tags" not in table.column_names:
        return table

    tags_col = table.column("tags")
    processed_tags = []

    for i in range(tags_col.length()):
        value = tags_col[i].as_py()
        if value is None or value == "":
            processed_tags.append([])
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    # Extract labels if tags are objects
                    labels = []
                    for tag in parsed:
                        if isinstance(tag, dict):
                            label = tag.get("label") or tag.get("slug") or tag.get("id", "")
                            if label:
                                labels.append(str(label))
                        elif isinstance(tag, str):
                            labels.append(tag)
                    processed_tags.append(labels)
                else:
                    processed_tags.append([str(parsed)])
            except json.JSONDecodeError:
                if "," in value:
                    processed_tags.append([t.strip() for t in value.split(",")])
                else:
                    processed_tags.append([value])
        elif isinstance(value, list):
            processed_tags.append([str(t) for t in value])
        else:
            processed_tags.append([])

    # Replace tags column
    new_tags_array = pa.array(processed_tags, type=pa.list_(pa.string()))

    # Build new table
    columns = {}
    for field_name in table.column_names:
        if field_name == "tags":
            columns[field_name] = new_tags_array
        else:
            columns[field_name] = table.column(field_name)

    return pa.table(columns)


def run_s3_update(
    trades_csv: Path | None = None,
    order_filled_csv: Path | None = None,
    markets_csv: Path | None = None,
    events_csv: Path | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    s3_region: str | None = None,
    metadata_db_path: Path | None = None,
    normalize_addresses: bool = True,
    validate_data: bool = True,
) -> dict[str, Any]:
    """
    Run incremental S3 update with partitioned writes.

    - Trades and order_filled: Partitioned by year/month/day
    - Markets and events: Full rewrite (small files)

    Args:
        trades_csv: Path to trades CSV
        order_filled_csv: Path to order_filled CSV
        markets_csv: Path to markets CSV
        events_csv: Path to events CSV
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix (e.g., 'raw/polymarket')
        s3_region: AWS region
        metadata_db_path: Path to metadata SQLite database
        normalize_addresses: Whether to normalize Ethereum addresses
        validate_data: Whether to validate data

    Returns:
        Summary dict with rows written per entity and files created
    """
    # Use settings defaults
    trades_csv = trades_csv or settings.trades_csv
    order_filled_csv = order_filled_csv or settings.order_filled_csv
    markets_csv = markets_csv or settings.markets_csv
    events_csv = events_csv or settings.events_csv
    s3_bucket = s3_bucket or settings.s3_bucket
    s3_prefix = s3_prefix or settings.s3_raw_prefix
    s3_region = s3_region or settings.s3_region
    metadata_db_path = metadata_db_path or settings.metadata_db_path

    if not s3_bucket:
        raise ValueError("s3_bucket is required. Set POLYMKT_S3_BUCKET environment variable.")

    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    timer_start = perf_counter()

    logger.info(
        "s3_update_starting",
        run_id=run_id,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
    )

    # Initialize
    metadata_store = MetadataStore(metadata_db_path)
    s3_fs = _get_s3_filesystem(s3_region)

    # Create run record
    run_record = RunRecord(
        run_id=run_id,
        run_type="s3_update",
        start_time=start_time,
        status="running",
        schema_version="1.0.0",
    )
    metadata_store.create_run(run_record)

    results = {
        "run_id": run_id,
        "rows_written": {},
        "rows_quarantined": {},
        "files_created": [],
        "watermarks_updated": {},
    }

    try:
        # =====================================================================
        # Process Trades (Partitioned)
        # =====================================================================
        if trades_csv and trades_csv.exists():
            logger.info("processing_trades", path=str(trades_csv))

            # Read and filter by watermark
            trades_table = _read_csv_with_mapping(trades_csv, TRADES_COLUMN_MAPPING)
            trades_watermark = metadata_store.get_watermark("trades")
            trades_table = _filter_by_watermark(trades_table, trades_watermark, "timestamp")

            if trades_table.num_rows > 0:
                # Validate
                if validate_data:
                    validation = validate_and_normalize_trades(
                        trades_table, normalize_addresses=normalize_addresses
                    )
                    trades_table = validation.valid_table
                    results["rows_quarantined"]["trades"] = validation.rows_quarantined

                # Partition by date
                partitions = _partition_by_date(trades_table, "timestamp")

                # Write partitions to S3
                files = _write_partitioned_to_s3(
                    partitions,
                    s3_bucket,
                    s3_prefix,
                    "trades",
                    TRADES_SCHEMA,
                    s3_fs,
                    run_id,
                )

                results["files_created"].extend(files)
                results["rows_written"]["trades"] = trades_table.num_rows

                # Update watermark
                max_ts = _get_max_timestamp(trades_table, "timestamp")
                if max_ts:
                    new_watermark = {"last_timestamp": max_ts.isoformat()}
                    metadata_store.set_watermark("trades", new_watermark)
                    results["watermarks_updated"]["trades"] = new_watermark
            else:
                results["rows_written"]["trades"] = 0
                logger.info("no_new_trades", message="No trades after watermark filter")

        # =====================================================================
        # Process Order Filled (Partitioned)
        # =====================================================================
        if order_filled_csv and order_filled_csv.exists():
            logger.info("processing_order_filled", path=str(order_filled_csv))

            # Read and filter by watermark
            of_table = _read_csv_with_mapping(order_filled_csv, ORDER_FILLED_COLUMN_MAPPING)
            of_watermark = metadata_store.get_watermark("order_filled")
            of_table = _filter_by_watermark(of_table, of_watermark, "timestamp")

            if of_table.num_rows > 0:
                # Validate
                if validate_data:
                    validation = validate_and_normalize_order_filled(
                        of_table, normalize_addresses=normalize_addresses
                    )
                    of_table = validation.valid_table
                    results["rows_quarantined"]["order_filled"] = validation.rows_quarantined

                # Partition by date
                partitions = _partition_by_date(of_table, "timestamp")

                # Write partitions to S3
                files = _write_partitioned_to_s3(
                    partitions,
                    s3_bucket,
                    s3_prefix,
                    "order_filled",
                    ORDER_FILLED_SCHEMA,
                    s3_fs,
                    run_id,
                )

                results["files_created"].extend(files)
                results["rows_written"]["order_filled"] = of_table.num_rows

                # Update watermark
                max_ts = _get_max_timestamp(of_table, "timestamp")
                if max_ts:
                    new_watermark = {"last_timestamp": max_ts.isoformat()}
                    metadata_store.set_watermark("order_filled", new_watermark)
                    results["watermarks_updated"]["order_filled"] = new_watermark
            else:
                results["rows_written"]["order_filled"] = 0
                logger.info("no_new_order_filled", message="No order_filled after watermark filter")

        # =====================================================================
        # Process Markets (Full Rewrite - Small File)
        # =====================================================================
        if markets_csv and markets_csv.exists():
            logger.info("processing_markets", path=str(markets_csv))

            markets_table = _read_csv_with_mapping(markets_csv, MARKETS_COLUMN_MAPPING)

            if validate_data:
                validation = validate_and_normalize_markets(markets_table)
                markets_table = validation.valid_table
                results["rows_quarantined"]["markets"] = validation.rows_quarantined

            # Write full file to S3
            file_path = _write_full_to_s3(
                markets_table,
                s3_bucket,
                s3_prefix,
                "markets.parquet",
                MARKETS_SCHEMA,
                s3_fs,
            )

            results["files_created"].append(file_path)
            results["rows_written"]["markets"] = markets_table.num_rows

            # Update watermark
            max_ts = _get_max_timestamp(markets_table, "created_at")
            if max_ts:
                new_watermark = {"last_timestamp": max_ts.isoformat()}
                metadata_store.set_watermark("markets", new_watermark)
                results["watermarks_updated"]["markets"] = new_watermark

        # =====================================================================
        # Process Events (Full Rewrite - Small File)
        # =====================================================================
        if events_csv and events_csv.exists():
            logger.info("processing_events", path=str(events_csv))

            events_table = _read_csv_with_mapping(events_csv, EVENTS_COLUMN_MAPPING)
            events_table = _process_events_tags(events_table)

            if validate_data:
                validation = validate_and_normalize_events(events_table)
                events_table = validation.valid_table
                results["rows_quarantined"]["events"] = validation.rows_quarantined

            # Write full file to S3
            file_path = _write_full_to_s3(
                events_table,
                s3_bucket,
                s3_prefix,
                "events.parquet",
                EVENTS_SCHEMA,
                s3_fs,
            )

            results["files_created"].append(file_path)
            results["rows_written"]["events"] = events_table.num_rows

            # Update watermark
            max_ts = _get_max_timestamp(events_table, "created_at")
            if max_ts:
                new_watermark = {"last_timestamp": max_ts.isoformat()}
                metadata_store.set_watermark("events", new_watermark)
                results["watermarks_updated"]["events"] = new_watermark

        # Complete
        duration = perf_counter() - timer_start
        results["duration_seconds"] = duration
        results["status"] = "completed"

        # Update run record
        run_record.end_time = datetime.now(timezone.utc)
        run_record.status = "completed"
        run_record.rows_written = results["rows_written"]
        run_record.duration_seconds = duration
        metadata_store.update_run(run_record)

        logger.info(
            "s3_update_completed",
            run_id=run_id,
            duration_seconds=duration,
            rows_written=results["rows_written"],
            files_created=len(results["files_created"]),
        )

        return results

    except Exception as e:
        # Update run record with failure
        run_record.end_time = datetime.now(timezone.utc)
        run_record.status = "failed"
        run_record.error_message = str(e)
        metadata_store.update_run(run_record)

        logger.error("s3_update_failed", run_id=run_id, error=str(e))
        raise


def run_s3_update_incremental(
    entity: str,
    csv_path: Path | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    s3_region: str | None = None,
    metadata_db_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run incremental S3 update for a single entity.

    Useful for updating just trades or just events without touching other entities.

    Args:
        entity: Entity to update ('trades', 'order_filled', 'markets', 'events')
        csv_path: Path to CSV (uses default if not provided)
        s3_bucket: S3 bucket
        s3_prefix: S3 prefix
        s3_region: AWS region
        metadata_db_path: Metadata DB path

    Returns:
        Summary dict
    """
    kwargs = {
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "s3_region": s3_region,
        "metadata_db_path": metadata_db_path,
    }

    # Set only the specified entity's CSV
    if entity == "trades":
        kwargs["trades_csv"] = csv_path or settings.trades_csv
        kwargs["order_filled_csv"] = None
        kwargs["markets_csv"] = None
        kwargs["events_csv"] = None
    elif entity == "order_filled":
        kwargs["trades_csv"] = None
        kwargs["order_filled_csv"] = csv_path or settings.order_filled_csv
        kwargs["markets_csv"] = None
        kwargs["events_csv"] = None
    elif entity == "markets":
        kwargs["trades_csv"] = None
        kwargs["order_filled_csv"] = None
        kwargs["markets_csv"] = csv_path or settings.markets_csv
        kwargs["events_csv"] = None
    elif entity == "events":
        kwargs["trades_csv"] = None
        kwargs["order_filled_csv"] = None
        kwargs["markets_csv"] = None
        kwargs["events_csv"] = csv_path or settings.events_csv
    else:
        raise ValueError(f"Unknown entity: {entity}. Must be one of: trades, order_filled, markets, events")

    return run_s3_update(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run S3 partitioned update")
    parser.add_argument(
        "--entity", "-e",
        choices=["all", "trades", "order_filled", "markets", "events"],
        default="all",
        help="Entity to update (default: all)"
    )
    parser.add_argument(
        "--bucket", "-b",
        help="S3 bucket (uses POLYMKT_S3_BUCKET if not provided)"
    )
    parser.add_argument(
        "--prefix", "-p",
        default="raw/polymarket",
        help="S3 prefix (default: raw/polymarket)"
    )
    args = parser.parse_args()

    if args.entity == "all":
        result = run_s3_update(
            s3_bucket=args.bucket,
            s3_prefix=args.prefix,
        )
    else:
        result = run_s3_update_incremental(
            entity=args.entity,
            s3_bucket=args.bucket,
            s3_prefix=args.prefix,
        )

    print(f"Update completed: {result}")
