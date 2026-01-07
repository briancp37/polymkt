"""Parquet storage layer for analytics data."""

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger()

# Schema definitions for each entity
MARKETS_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("question", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")),
    ("answer1", pa.string()),
    ("answer2", pa.string()),
    ("neg_risk", pa.bool_()),
    ("market_slug", pa.string()),
    ("token1", pa.string()),
    ("token2", pa.string()),
    ("condition_id", pa.string()),
    ("volume", pa.float64()),
    ("ticker", pa.string()),
    ("closed_time", pa.timestamp("us", tz="UTC")),
    ("description", pa.string()),
    ("category", pa.string()),
])

TRADES_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("market_id", pa.string()),
    ("maker", pa.string()),
    ("taker", pa.string()),
    ("nonusdc_side", pa.string()),
    ("maker_direction", pa.string()),
    ("taker_direction", pa.string()),
    ("price", pa.float64()),
    ("usd_amount", pa.float64()),
    ("token_amount", pa.float64()),
    ("transaction_hash", pa.string()),
])

ORDER_FILLED_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("maker", pa.string()),
    ("maker_asset_id", pa.string()),
    ("maker_amount_filled", pa.float64()),
    ("taker", pa.string()),
    ("taker_asset_id", pa.string()),
    ("taker_amount_filled", pa.float64()),
    ("transaction_hash", pa.string()),
])


def compute_hash_bucket(market_id: pa.Array, bucket_count: int) -> pa.Array:
    """Compute hash bucket for market_id using consistent hashing."""
    import hashlib

    # Compute hash for each market_id using Python's hashlib
    def hash_to_bucket(value: str | None) -> int:
        if value is None:
            return 0
        # Use MD5 for fast, deterministic hashing
        hash_bytes = hashlib.md5(value.encode("utf-8")).digest()
        # Use first 8 bytes as integer
        hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
        return hash_int % bucket_count

    buckets = [hash_to_bucket(v) for v in market_id.to_pylist()]
    return pa.array(buckets, type=pa.int32())


def add_partition_columns(table: pa.Table, bucket_count: int = 8) -> pa.Table:
    """Add partition columns (year, month, day, hash_bucket) to a table with timestamp and market_id."""
    timestamp_col = table.column("timestamp")

    # Extract year, month, day from timestamp
    year = pc.year(timestamp_col)
    month = pc.month(timestamp_col)
    day = pc.day(timestamp_col)

    # Compute hash bucket from market_id
    market_id_col = table.column("market_id")
    hash_bucket = compute_hash_bucket(market_id_col, bucket_count)

    # Append partition columns to table
    table = table.append_column("year", year)
    table = table.append_column("month", month)
    table = table.append_column("day", day)
    table = table.append_column("hash_bucket", hash_bucket)

    return table


class ParquetWriter:
    """Writes DataFrames to Parquet with proper schema and compression."""

    def __init__(
        self,
        output_dir: Path,
        compression: str = "zstd",
        row_group_size: int = 100_000,
        partitioning_enabled: bool = False,
        hash_bucket_count: int = 8,
    ) -> None:
        self.output_dir = output_dir
        self.compression = compression
        self.row_group_size = row_group_size
        self.partitioning_enabled = partitioning_enabled
        self.hash_bucket_count = hash_bucket_count
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_markets(self, table: pa.Table) -> Path:
        """Write markets data to Parquet."""
        output_path = self.output_dir / "markets.parquet"
        table = table.cast(MARKETS_SCHEMA)
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )
        logger.info(
            "parquet_written",
            entity="markets",
            path=str(output_path),
            rows=table.num_rows,
        )
        return output_path

    def write_trades(self, table: pa.Table) -> Path:
        """Write trades data to Parquet, optionally partitioned."""
        table = table.cast(TRADES_SCHEMA)

        if self.partitioning_enabled:
            return self._write_trades_partitioned(table)
        else:
            return self._write_trades_monolithic(table)

    def _write_trades_monolithic(self, table: pa.Table) -> Path:
        """Write trades as a single Parquet file."""
        output_path = self.output_dir / "trades.parquet"
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )
        logger.info(
            "parquet_written",
            entity="trades",
            path=str(output_path),
            rows=table.num_rows,
            partitioned=False,
        )
        return output_path

    def _write_trades_partitioned(self, table: pa.Table) -> Path:
        """Write trades as partitioned Parquet dataset (year/month/day/hash_bucket)."""
        output_path = self.output_dir / "trades"

        # Add partition columns
        table_with_partitions = add_partition_columns(table, self.hash_bucket_count)

        # Define partitioning scheme
        partitioning = ds.partitioning(
            pa.schema([
                ("year", pa.int32()),
                ("month", pa.int32()),
                ("day", pa.int32()),
                ("hash_bucket", pa.int32()),
            ]),
            flavor="hive",
        )

        # Write using PyArrow dataset API
        ds.write_dataset(
            table_with_partitions,
            output_path,
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="delete_matching",
            file_options=ds.ParquetFileFormat().make_write_options(
                compression=self.compression,
            ),
        )

        # Count unique partitions for logging
        unique_dates = table_with_partitions.group_by(["year", "month", "day"]).aggregate([]).num_rows

        logger.info(
            "parquet_written",
            entity="trades",
            path=str(output_path),
            rows=table.num_rows,
            partitioned=True,
            partition_count=unique_dates,
            hash_buckets=self.hash_bucket_count,
        )
        return output_path

    def write_order_filled(self, table: pa.Table) -> Path:
        """Write order filled data to Parquet."""
        # Order filled doesn't have market_id, so no partitioning (just time-based optional)
        output_path = self.output_dir / "order_filled.parquet"
        table = table.cast(ORDER_FILLED_SCHEMA)
        pq.write_table(
            table,
            output_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )
        logger.info(
            "parquet_written",
            entity="order_filled",
            path=str(output_path),
            rows=table.num_rows,
        )
        return output_path

    def is_partitioned(self) -> bool:
        """Check if partitioning is enabled."""
        return self.partitioning_enabled


def iter_csv_batches(
    csv_path: Path,
    batch_size: int = 100_000,
) -> Iterator[pa.RecordBatch]:
    """Iterate over CSV file in batches using PyArrow streaming."""
    read_options = pa.csv.ReadOptions(block_size=batch_size * 1024)
    convert_options = pa.csv.ConvertOptions(
        timestamp_parsers=["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"],
    )

    reader = pa.csv.open_csv(
        csv_path,
        read_options=read_options,
        convert_options=convert_options,
    )

    for batch in reader:
        yield batch
