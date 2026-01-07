"""Curate pipeline for building analytics layer from raw layer."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from polymkt.config import settings
from polymkt.models.schemas import CurateSummary
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.parquet import (
    TRADES_ANALYTICS_SCHEMA,
    ParquetWriter,
)

logger = structlog.get_logger()


def _read_raw_trades(raw_dir: Path, partitioned: bool) -> pa.Table:
    """Read trades from raw layer."""
    if partitioned:
        trades_path = raw_dir / "trades"
        if not trades_path.exists():
            raise FileNotFoundError(f"Raw trades directory not found: {trades_path}")
        table = pq.read_table(str(trades_path) + "/**/*.parquet")
    else:
        trades_path = raw_dir / "trades.parquet"
        if not trades_path.exists():
            raise FileNotFoundError(f"Raw trades file not found: {trades_path}")
        table = pq.read_table(trades_path)

    logger.info("raw_trades_read", path=str(trades_path), rows=table.num_rows)
    return table


def _read_raw_markets(raw_dir: Path) -> pa.Table:
    """Read markets from raw layer."""
    markets_path = raw_dir / "markets.parquet"
    if not markets_path.exists():
        raise FileNotFoundError(f"Raw markets file not found: {markets_path}")
    table = pq.read_table(markets_path)
    logger.info("raw_markets_read", path=str(markets_path), rows=table.num_rows)
    return table


def _compute_days_to_exp(
    trades: pa.Table,
    markets: pa.Table,
) -> pa.Table:
    """
    Compute days_to_exp by joining trades to markets and calculating the derived field.

    days_to_exp = (market.closed_time - trade.timestamp) in days
    Returns NULL if market has no closed_time.
    """
    # Use DuckDB for efficient join and calculation
    conn = duckdb.connect(":memory:")

    # Register tables
    conn.register("trades", trades)
    conn.register("markets", markets)

    # Join and compute days_to_exp
    result = conn.execute("""
        SELECT
            t.timestamp,
            t.market_id,
            t.maker,
            t.taker,
            t.nonusdc_side,
            t.maker_direction,
            t.taker_direction,
            t.price,
            t.usd_amount,
            t.token_amount,
            t.transaction_hash,
            CASE
                WHEN m.closed_time IS NOT NULL
                THEN EXTRACT(EPOCH FROM (m.closed_time - t.timestamp)) / 86400.0
                ELSE NULL
            END AS days_to_exp
        FROM trades t
        LEFT JOIN markets m ON t.market_id = m.id
    """).fetch_arrow_table()

    conn.close()

    logger.info(
        "days_to_exp_computed",
        trades_count=trades.num_rows,
        result_count=result.num_rows,
    )

    return result


def run_curate(
    raw_dir: Path | None = None,
    analytics_dir: Path | None = None,
    metadata_db_path: Path | None = None,
    partitioning_enabled: bool | None = None,
    hash_bucket_count: int | None = None,
) -> CurateSummary:
    """
    Run the curate step to build analytics layer from raw layer.

    Reads raw Parquet files and builds analytics Parquet files with
    derived fields (e.g., days_to_exp).

    The raw layer is NOT modified by this operation.

    Args:
        raw_dir: Directory containing raw Parquet files
        analytics_dir: Directory for analytics Parquet output
        metadata_db_path: Path to metadata SQLite database
        partitioning_enabled: Whether to partition analytics Parquet
        hash_bucket_count: Number of hash buckets for market_id partitioning

    Returns:
        CurateSummary with run details
    """
    # Use settings defaults if not provided
    raw_dir = raw_dir or settings.parquet_raw_dir
    analytics_dir = analytics_dir or settings.parquet_analytics_dir
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

    # Create analytics writer
    analytics_writer = ParquetWriter(
        analytics_dir,
        partitioning_enabled=partitioning_enabled,
        hash_bucket_count=hash_bucket_count,
    )

    analytics_files: list[str] = []
    trades_rows_read = 0
    trades_rows_written = 0

    try:
        # Read raw data
        logger.info("curate_starting", raw_dir=str(raw_dir), analytics_dir=str(analytics_dir))

        raw_trades = _read_raw_trades(raw_dir, partitioned=partitioning_enabled)
        raw_markets = _read_raw_markets(raw_dir)
        trades_rows_read = raw_trades.num_rows

        # Compute derived fields
        trades_analytics = _compute_days_to_exp(raw_trades, raw_markets)

        # Write to analytics layer
        output_path = analytics_writer.write_trades_analytics(trades_analytics)
        trades_rows_written = trades_analytics.num_rows
        analytics_files.append(str(output_path))

        # Calculate duration
        duration = perf_counter() - timer_start

        # Set watermark for curate step
        metadata_store.set_watermark(
            "curate",
            {
                "last_run": datetime.now(timezone.utc).isoformat(),
                "trades_rows": trades_rows_written,
            },
        )

        logger.info(
            "curate_completed",
            run_id=run_id,
            duration_seconds=duration,
            trades_rows_read=trades_rows_read,
            trades_rows_written=trades_rows_written,
        )

        return CurateSummary(
            run_id=run_id,
            status="completed",
            duration_seconds=duration,
            trades_rows_read=trades_rows_read,
            trades_rows_written=trades_rows_written,
            analytics_files=analytics_files,
        )

    except Exception as e:
        duration = perf_counter() - timer_start
        logger.error("curate_failed", run_id=run_id, error=str(e))
        raise
