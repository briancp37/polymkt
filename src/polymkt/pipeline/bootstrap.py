"""Bootstrap import pipeline for loading CSV data into Parquet."""

import gc
import json
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import structlog

from polymkt.config import BootstrapConfig, settings
from polymkt.models.schemas import BootstrapSummary, RunRecord

# Default batch size - reduced from 100K to 10K for safer memory usage
DEFAULT_BATCH_SIZE = 10_000
from polymkt.pipeline.normalize import (
    BatchValidationResult,
    ValidationResult,
    validate_and_normalize_events,
    validate_and_normalize_markets,
    validate_and_normalize_order_filled,
    validate_and_normalize_trades,
    validate_events_batch,
    validate_markets_batch,
    validate_order_filled_batch,
    validate_trades_batch,
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


# =============================================================================
# S3 Streaming Support
# =============================================================================


def _get_s3_filesystem(region: str = "us-east-1") -> "pa.fs.S3FileSystem":
    """
    Create a PyArrow S3FileSystem for direct S3 writes.

    Uses AWS credentials from environment variables, AWS CLI profile,
    or IAM role (when running on EC2/ECS/Lambda).

    Args:
        region: AWS region for S3 bucket

    Returns:
        PyArrow S3FileSystem configured for the region
    """
    import pyarrow.fs as pafs

    return pafs.S3FileSystem(region=region)


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse an S3 URI into bucket and key.

    Args:
        s3_path: S3 URI like 's3://bucket/path/to/file.parquet'

    Returns:
        Tuple of (bucket, key)
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'")

    path = s3_path[5:]  # Remove 's3://'
    parts = path.split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid S3 path: {s3_path}. Must include bucket and key")

    return parts[0], parts[1]


# =============================================================================
# Streaming Partitioned S3 Writer
# =============================================================================


class StreamingPartitionedS3Writer:
    """Memory-efficient writer that streams CSV batches to partitioned S3 parquet.

    This class solves the memory problem by:
    1. Buffering rows by partition (year/month/day) up to a threshold
    2. Flushing partitions to S3 as soon as they reach the buffer limit
    3. Never holding the entire dataset in memory

    Each partition flush uploads directly to S3 without local temp files.
    """

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,  # e.g., "raw/polymarket"
        entity: str,  # e.g., "trades", "order_filled"
        schema: pa.Schema,
        s3_region: str = "us-east-1",
        buffer_size: int = 50_000,  # Rows per partition before flush
        timestamp_column: str = "timestamp",
    ) -> None:
        import pyarrow.fs as pafs

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.entity = entity
        self.schema = schema
        self.buffer_size = buffer_size
        self.timestamp_column = timestamp_column

        self.s3fs = pafs.S3FileSystem(region=s3_region)

        # Buffers: partition_key -> list of RecordBatches
        from collections import defaultdict
        self.buffers: dict[str, list[pa.RecordBatch]] = defaultdict(list)
        self.buffer_rows: dict[str, int] = defaultdict(int)

        # Track written files and rows
        self.files_written: dict[str, list[str]] = defaultdict(list)
        self.rows_written: dict[str, int] = defaultdict(int)
        self.total_rows_written = 0
        self.total_flushes = 0

    def add_batch(self, batch: pa.RecordBatch) -> int:
        """Add a validated batch to be partitioned and written.

        The batch is partitioned by timestamp into year/month/day buckets.
        Partitions that exceed buffer_size are immediately flushed to S3.

        Args:
            batch: RecordBatch with timestamp column for partitioning

        Returns:
            Number of rows flushed to S3 during this call
        """
        import pyarrow.compute as pc
        from collections import defaultdict

        if batch.num_rows == 0:
            return 0

        rows_flushed = 0
        table = pa.Table.from_batches([batch])

        # Extract partition keys from timestamp
        if self.timestamp_column not in table.column_names:
            # No timestamp column - use default partition
            partition_key = "year=1970/month=01/day=01"
            self.buffers[partition_key].append(batch)
            self.buffer_rows[partition_key] += batch.num_rows
            if self.buffer_rows[partition_key] >= self.buffer_size:
                rows_flushed += self._flush_partition(partition_key)
            return rows_flushed

        timestamps = table.column(self.timestamp_column)

        # Compute year, month, day arrays
        years = pc.year(timestamps)
        months = pc.month(timestamps)
        days = pc.day(timestamps)

        # Build partition keys for each row and group
        partition_indices: dict[str, list[int]] = defaultdict(list)
        for i in range(table.num_rows):
            y = years[i].as_py()
            m = months[i].as_py()
            d = days[i].as_py()
            if y is None or m is None or d is None:
                partition_key = "year=1970/month=01/day=01"
            else:
                partition_key = f"year={y}/month={m:02d}/day={d:02d}"
            partition_indices[partition_key].append(i)

        # Add rows to appropriate buffers
        for partition_key, indices in partition_indices.items():
            partition_table = table.take(indices)
            partition_batch = partition_table.to_batches()[0]

            self.buffers[partition_key].append(partition_batch)
            self.buffer_rows[partition_key] += partition_batch.num_rows

            # Flush if buffer is large enough
            if self.buffer_rows[partition_key] >= self.buffer_size:
                rows_flushed += self._flush_partition(partition_key)

        return rows_flushed

    def _flush_partition(self, partition_key: str) -> int:
        """Flush buffered data for a partition to S3.

        Returns:
            Number of rows written
        """
        if not self.buffers[partition_key]:
            return 0

        # Combine all batches
        table = pa.Table.from_batches(self.buffers[partition_key], schema=self.schema)
        row_count = table.num_rows

        # Generate filename with timestamp for uniqueness
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.entity}_{ts}_{row_count}.parquet"
        output_path = f"{self.s3_bucket}/{self.s3_prefix}/{self.entity}/{partition_key}/{filename}"

        # Ensure directory exists and write
        dir_path = f"{self.s3_bucket}/{self.s3_prefix}/{self.entity}/{partition_key}"
        self.s3fs.create_dir(dir_path)

        pq.write_table(
            table,
            output_path,
            filesystem=self.s3fs,
            compression="zstd",
        )

        # Track
        self.files_written[partition_key].append(f"s3://{output_path}")
        self.rows_written[partition_key] += row_count
        self.total_rows_written += row_count
        self.total_flushes += 1

        logger.info(
            "partition_flushed_to_s3",
            entity=self.entity,
            partition=partition_key,
            rows=row_count,
            file=filename,
            total_flushes=self.total_flushes,
            total_rows=self.total_rows_written,
        )

        # Clear buffer
        self.buffers[partition_key] = []
        self.buffer_rows[partition_key] = 0

        # Help GC
        del table
        gc.collect()

        return row_count

    def flush_all(self) -> int:
        """Flush all remaining buffered data.

        Returns:
            Total number of rows flushed
        """
        rows_flushed = 0
        for partition_key in list(self.buffers.keys()):
            if self.buffers[partition_key]:
                rows_flushed += self._flush_partition(partition_key)
        return rows_flushed

    def get_stats(self) -> dict:
        """Get statistics about written data."""
        return {
            "partitions": len(self.rows_written),
            "files": sum(len(files) for files in self.files_written.values()),
            "rows": self.total_rows_written,
            "flushes": self.total_flushes,
            "partition_details": dict(self.rows_written),
        }


def _stream_csv_to_partitioned_s3(
    csv_path: Path,
    s3_bucket: str,
    s3_prefix: str,
    entity: str,
    target_schema: pa.Schema,
    column_mapping: dict[str, str],
    validate_batch_fn,
    normalize_addresses: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    buffer_size: int = 50_000,
    s3_region: str = "us-east-1",
    timestamp_column: str = "timestamp",
    config: BootstrapConfig | None = None,
) -> dict[str, Any]:
    """
    Stream CSV file to partitioned S3 parquet with bounded memory.

    This is the key function for memory-efficient partitioned S3 writes:
    1. Reads CSV in batches (not all at once)
    2. Validates each batch
    3. Partitions by timestamp (year/month/day)
    4. Flushes partitions to S3 when buffer threshold is reached
    5. Never holds more than buffer_size rows per partition in memory

    Args:
        csv_path: Path to input CSV
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix (e.g., 'raw/polymarket')
        entity: Entity name (trades, order_filled)
        target_schema: Target Parquet schema
        column_mapping: CSV column name mappings
        validate_batch_fn: Batch validation function
        normalize_addresses: Whether to normalize address fields
        batch_size: Rows per CSV batch read
        buffer_size: Rows per partition before S3 flush
        s3_region: AWS region
        timestamp_column: Column to use for partitioning
        config: Bootstrap configuration for memory limits

    Returns:
        Dict with rows_read, rows_written, rows_quarantined, stats
    """
    from polymkt.pipeline.normalize import normalize_timestamp

    # Use config values if provided
    if config is not None:
        batch_size = config.batch_size

    stats = {
        "rows_read": 0,
        "rows_written": 0,
        "rows_quarantined": 0,
        "partitions": 0,
        "files": 0,
    }

    logger.info(
        "streaming_csv_to_partitioned_s3",
        input=str(csv_path),
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        entity=entity,
        batch_size=batch_size,
        buffer_size=buffer_size,
    )

    # Initialize partitioned S3 writer
    writer = StreamingPartitionedS3Writer(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        entity=entity,
        schema=target_schema,
        s3_region=s3_region,
        buffer_size=buffer_size,
        timestamp_column=timestamp_column,
    )

    # Initialize memory monitor if config provided
    memory_monitor: MemoryMonitor | None = None
    if config is not None:
        memory_monitor = MemoryMonitor(
            config.max_memory_mb,
            config.memory_check_interval,
        )

    batch_index = 0
    for batch in _iter_csv_batches(csv_path, column_mapping, batch_size):
        batch_index += 1
        stats["rows_read"] += batch.num_rows

        # Memory check - flush all partitions if under pressure
        if memory_monitor and memory_monitor.check_and_throttle():
            logger.warning("memory_pressure_flushing_all_partitions")
            writer.flush_all()
            gc.collect()

        # Normalize timestamp column
        table = pa.Table.from_batches([batch])
        if timestamp_column in table.column_names:
            ts_col = table.column(timestamp_column)
            normalized_timestamps = []
            for i in range(ts_col.length()):
                val = ts_col[i].as_py()
                normalized_timestamps.append(normalize_timestamp(val, timestamp_column))
            table = table.set_column(
                table.column_names.index(timestamp_column),
                timestamp_column,
                pa.array(normalized_timestamps, type=pa.timestamp("us", tz="UTC")),
            )
            batch = table.to_batches()[0]

        # Validate batch
        if validate_batch_fn:
            import inspect
            sig = inspect.signature(validate_batch_fn)
            if "normalize_addresses" in sig.parameters:
                validation_result = validate_batch_fn(
                    batch, target_schema, normalize_addresses=normalize_addresses
                )
            else:
                validation_result = validate_batch_fn(batch, target_schema)

            valid_batch = validation_result.valid_batch
            stats["rows_quarantined"] += validation_result.rows_quarantined
        else:
            # No validation, but still need to cast to target schema
            table = pa.Table.from_batches([batch])
            columns = {}
            for field in target_schema:
                if field.name in table.column_names:
                    col = table.column(field.name)
                    try:
                        col = col.cast(field.type)
                    except pa.ArrowInvalid:
                        pass
                    columns[field.name] = col
                else:
                    columns[field.name] = pa.nulls(table.num_rows, type=field.type)
            casted_table = pa.table(columns, schema=target_schema)
            valid_batch = casted_table.to_batches()[0] if casted_table.num_rows > 0 else batch.slice(0, 0)

        # Add valid rows to partitioned writer (flushes automatically when buffer full)
        if valid_batch.num_rows > 0:
            writer.add_batch(valid_batch)

        # Progress logging
        if batch_index % 100 == 0:
            writer_stats = writer.get_stats()
            mem_mb = memory_monitor.get_current_usage_mb() if memory_monitor else 0
            logger.info(
                "partition_progress",
                batches=batch_index,
                rows_read=stats["rows_read"],
                rows_to_s3=writer_stats["rows"],
                partitions=writer_stats["partitions"],
                flushes=writer_stats["flushes"],
                memory_mb=round(mem_mb, 1),
            )

        # GC after each batch
        gc.collect()

    # Flush remaining data
    writer.flush_all()

    # Get final stats
    writer_stats = writer.get_stats()
    stats["rows_written"] = writer_stats["rows"]
    stats["partitions"] = writer_stats["partitions"]
    stats["files"] = writer_stats["files"]
    stats["partition_details"] = writer_stats["partition_details"]

    logger.info(
        "partitioned_s3_streaming_complete",
        entity=entity,
        rows_read=stats["rows_read"],
        rows_written=stats["rows_written"],
        rows_quarantined=stats["rows_quarantined"],
        partitions=stats["partitions"],
        files=stats["files"],
    )

    return stats


def _open_parquet_writer(
    output_path: str | Path,
    schema: pa.Schema,
    s3_filesystem: "pa.fs.S3FileSystem | None" = None,
) -> pq.ParquetWriter:
    """
    Open a ParquetWriter for local or S3 output.

    Args:
        output_path: Local path or S3 URI (s3://bucket/key)
        schema: PyArrow schema for the Parquet file
        s3_filesystem: S3FileSystem for S3 writes (required if output_path is S3 URI)

    Returns:
        ParquetWriter that writes to local disk or S3
    """
    path_str = str(output_path)

    if path_str.startswith("s3://"):
        if s3_filesystem is None:
            raise ValueError("s3_filesystem required for S3 output paths")

        bucket, key = _parse_s3_path(path_str)
        # PyArrow expects bucket/key format without s3:// prefix for S3FileSystem
        s3_full_path = f"{bucket}/{key}"

        logger.info("opening_s3_parquet_writer", bucket=bucket, key=key)
        return pq.ParquetWriter(s3_full_path, schema, filesystem=s3_filesystem)
    else:
        # Local path - ensure directory exists
        local_path = Path(path_str)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return pq.ParquetWriter(local_path, schema)


# =============================================================================
# Memory-Safe Bootstrap Classes
# =============================================================================


class MemoryMonitor:
    """Monitor system memory and save checkpoint before crashing.

    This class helps prevent OOM crashes by monitoring memory usage
    during batch processing and triggering garbage collection when
    memory pressure is high.
    """

    def __init__(self, max_memory_mb: int, check_interval: int = 10) -> None:
        """Initialize memory monitor.

        Args:
            max_memory_mb: Maximum memory usage in MB before raising error
            check_interval: Check memory every N batches
        """
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.batch_count = 0

    def check_and_throttle(self) -> bool:
        """Check memory usage and trigger GC if needed.

        Returns:
            True if memory limit is critically exceeded (should save and stop)
        """
        self.batch_count += 1
        if self.batch_count % self.check_interval != 0:
            return False

        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)

        if mem_mb > self.max_memory_mb * 0.8:
            # Memory pressure - trigger garbage collection
            gc.collect()
            mem_after_gc = process.memory_info().rss / (1024 * 1024)
            logger.warning(
                "memory_pressure_detected",
                before_gc_mb=round(mem_mb, 1),
                after_gc_mb=round(mem_after_gc, 1),
                limit_mb=self.max_memory_mb,
            )
            # If still over 95% after GC, we should stop
            if mem_after_gc > self.max_memory_mb * 0.95:
                return True
        return False

    def get_current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / (1024 * 1024)


@dataclass
class BootstrapCheckpoint:
    """State saved for resumable bootstrap processing."""

    run_id: str
    entity: str  # 'events', 'markets', 'trades', 'order_filled'
    rows_processed: int
    last_batch_index: int
    timestamp: str  # ISO format string


class CheckpointManager:
    """Manage bootstrap checkpoints for resume capability.

    Saves processing state to disk so bootstrap can resume from
    where it left off after a crash or manual stop.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint: BootstrapCheckpoint) -> None:
        """Save checkpoint to disk."""
        path = self.checkpoint_dir / f"{checkpoint.run_id}_{checkpoint.entity}.json"
        path.write_text(json.dumps(asdict(checkpoint)))
        logger.debug(
            "checkpoint_saved",
            entity=checkpoint.entity,
            batch=checkpoint.last_batch_index,
            rows=checkpoint.rows_processed,
        )

    def load_checkpoint(self, run_id: str, entity: str) -> BootstrapCheckpoint | None:
        """Load checkpoint if exists.

        Args:
            run_id: The run ID to resume
            entity: The entity type (events, markets, trades, order_filled)

        Returns:
            BootstrapCheckpoint if found, None otherwise
        """
        path = self.checkpoint_dir / f"{run_id}_{entity}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return BootstrapCheckpoint(**data)
        return None

    def clear_checkpoints(self, run_id: str) -> None:
        """Clear all checkpoints for a run after successful completion."""
        for f in self.checkpoint_dir.glob(f"{run_id}_*.json"):
            f.unlink()
            logger.debug("checkpoint_cleared", path=str(f))

    def get_latest_run_id(self, entity: str) -> str | None:
        """Find the most recent run ID with a checkpoint for the given entity.

        Args:
            entity: The entity type to look for

        Returns:
            The run_id if found, None otherwise
        """
        checkpoints = list(self.checkpoint_dir.glob(f"*_{entity}.json"))
        if not checkpoints:
            return None
        # Sort by modification time and get the most recent
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        # Extract run_id from filename (format: {run_id}_{entity}.json)
        return latest.stem.rsplit("_", 1)[0]


class EventsLookup:
    """Disk-spillable events lookup for markets join.

    For small datasets, keeps events in memory for fast lookup.
    For large datasets (>max_in_memory), spills to SQLite on disk
    to prevent memory exhaustion.
    """

    def __init__(self, max_in_memory: int = 100_000) -> None:
        """Initialize events lookup.

        Args:
            max_in_memory: Maximum events to keep in memory before spilling to disk
        """
        self.max_in_memory = max_in_memory
        self.in_memory_lookup: dict[str, list[str]] = {}
        self.disk_db_path: Path | None = None
        self.use_disk = False
        self._conn: sqlite3.Connection | None = None

    def add_event(self, event_id: str, tags: list[str]) -> None:
        """Add event to lookup, spilling to disk if needed."""
        if self.use_disk:
            self._add_to_disk(event_id, tags)
        elif len(self.in_memory_lookup) >= self.max_in_memory:
            self._spill_to_disk()
            self._add_to_disk(event_id, tags)
        else:
            self.in_memory_lookup[event_id] = tags

    def get_tags(self, event_id: str | None) -> list[str]:
        """Get tags for event_id."""
        if event_id is None:
            return []
        if self.use_disk:
            return self._get_from_disk(event_id)
        return self.in_memory_lookup.get(event_id, [])

    def _spill_to_disk(self) -> None:
        """Move in-memory data to SQLite."""
        temp_dir = tempfile.mkdtemp(prefix="polymkt_events_")
        self.disk_db_path = Path(temp_dir) / "events_lookup.db"
        self._conn = sqlite3.connect(str(self.disk_db_path))
        self._conn.execute("""
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                tags TEXT
            )
        """)
        # Bulk insert existing in-memory data
        self._conn.executemany(
            "INSERT INTO events VALUES (?, ?)",
            [(eid, json.dumps(tags)) for eid, tags in self.in_memory_lookup.items()],
        )
        self._conn.commit()
        self.in_memory_lookup.clear()
        self.use_disk = True
        logger.info(
            "events_spilled_to_disk",
            path=str(self.disk_db_path),
            count=len(self.in_memory_lookup),
        )

    def _add_to_disk(self, event_id: str, tags: list[str]) -> None:
        """Add event to disk-based lookup."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO events VALUES (?, ?)",
            (event_id, json.dumps(tags)),
        )
        # Commit periodically for performance
        if self._conn.total_changes % 1000 == 0:
            self._conn.commit()

    def _get_from_disk(self, event_id: str) -> list[str]:
        """Get tags from disk-based lookup."""
        if self._conn is None:
            return []
        cursor = self._conn.execute(
            "SELECT tags FROM events WHERE event_id = ?", (event_id,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return []

    def finalize(self) -> None:
        """Finalize the lookup (commit any pending writes)."""
        if self._conn is not None:
            self._conn.commit()

    def close(self) -> None:
        """Close and cleanup resources."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self.disk_db_path is not None and self.disk_db_path.parent.exists():
            import shutil
            shutil.rmtree(self.disk_db_path.parent, ignore_errors=True)
            self.disk_db_path = None

    def __len__(self) -> int:
        """Get total number of events."""
        if self.use_disk and self._conn is not None:
            cursor = self._conn.execute("SELECT COUNT(*) FROM events")
            result = cursor.fetchone()
            return result[0] if result else 0
        return len(self.in_memory_lookup)


# =============================================================================
# CSV Processing Functions
# =============================================================================


def _rename_columns(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    """Rename columns in a PyArrow table."""
    new_names = [mapping.get(name, name) for name in table.column_names]
    return table.rename_columns(new_names)


def _iter_csv_batches(
    csv_path: Path,
    column_mapping: dict[str, str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """
    Stream CSV file in batches for memory-efficient processing.

    Args:
        csv_path: Path to CSV file
        column_mapping: Column name mappings
        batch_size: Number of rows per batch

    Yields:
        PyArrow RecordBatches with renamed columns
    """
    logger.info("streaming_csv", path=str(csv_path), batch_size=batch_size)

    read_options = csv.ReadOptions(block_size=batch_size * 1024)  # Approximate block size
    parse_options = csv.ParseOptions(newlines_in_values=True)
    convert_options = csv.ConvertOptions(
        timestamp_parsers=[],  # Disable auto-parsing, read as strings
    )

    # Open streaming reader
    reader = csv.open_csv(
        csv_path,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )

    batch_count = 0
    for batch in reader:
        # Rename columns in the batch
        table = pa.Table.from_batches([batch])
        table = _rename_columns(table, column_mapping)
        batch_count += 1

        if batch_count % 10 == 0:
            logger.info("csv_batch_read", batch=batch_count, rows=batch.num_rows)

        yield table.to_batches()[0]

    logger.info("csv_streaming_complete", path=str(csv_path), batches=batch_count)


def _stream_csv_to_parquet(
    csv_path: Path,
    output_path: Path | str,
    target_schema: pa.Schema,
    column_mapping: dict[str, str],
    validate_batch_fn,
    normalize_addresses: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    config: BootstrapConfig | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    run_id: str | None = None,
    entity: str | None = None,
    s3_filesystem: "pa.fs.S3FileSystem | None" = None,
) -> dict[str, int]:
    """
    Stream CSV file through validation to Parquet with bounded memory.

    This is the key memory-efficient processing function that:
    1. Reads CSV in batches (not all at once)
    2. Validates each batch using vectorized operations
    3. Writes validated batches directly to Parquet (local or S3)
    4. Never holds more than one batch in memory
    5. Monitors memory usage and saves checkpoints
    6. Can resume from checkpoints after a crash

    Args:
        csv_path: Path to input CSV
        output_path: Path to output Parquet file (local Path or S3 URI like s3://bucket/key)
        target_schema: Target Parquet schema
        column_mapping: CSV column name mappings
        validate_batch_fn: Batch validation function to use
        normalize_addresses: Whether to normalize address fields
        batch_size: Rows per batch
        config: Bootstrap configuration for memory limits
        checkpoint_manager: Manager for saving/loading checkpoints
        run_id: Run ID for checkpoint tracking
        entity: Entity name (events, markets, trades, order_filled)
        s3_filesystem: S3FileSystem for direct S3 writes (required if output_path is S3 URI)

    Returns:
        Dict with rows_read, rows_written, rows_quarantined
    """
    # Use config values if provided
    if config is not None:
        batch_size = config.batch_size

    stats = {
        "rows_read": 0,
        "rows_written": 0,
        "rows_quarantined": 0,
    }

    # Initialize memory monitor if config provided
    memory_monitor: MemoryMonitor | None = None
    if config is not None:
        memory_monitor = MemoryMonitor(
            config.max_memory_mb,
            config.memory_check_interval,
        )

    # Check for existing checkpoint to resume from
    start_batch = 0
    if checkpoint_manager and run_id and entity:
        checkpoint = checkpoint_manager.load_checkpoint(run_id, entity)
        if checkpoint:
            start_batch = checkpoint.last_batch_index
            stats["rows_read"] = checkpoint.rows_processed
            logger.info(
                "resuming_from_checkpoint",
                entity=entity,
                batch=start_batch,
                rows=checkpoint.rows_processed,
            )

    # Determine if writing to S3 or local
    output_str = str(output_path)
    is_s3_output = output_str.startswith("s3://")

    logger.info(
        "streaming_csv_to_parquet",
        input=str(csv_path),
        output=output_str,
        batch_size=batch_size,
        resume_from_batch=start_batch if start_batch > 0 else None,
        s3_streaming=is_s3_output,
    )

    # Open Parquet writer for incremental writes (local or S3)
    batch_index = 0
    with _open_parquet_writer(output_path, target_schema, s3_filesystem) as writer:
        for batch in _iter_csv_batches(csv_path, column_mapping, batch_size):
            batch_index += 1

            # Skip already-processed batches when resuming
            if batch_index <= start_batch:
                continue

            stats["rows_read"] += batch.num_rows

            # Check memory and save checkpoint if approaching limit
            if memory_monitor and memory_monitor.check_and_throttle():
                # Memory limit exceeded - save checkpoint and raise
                if checkpoint_manager and run_id and entity:
                    checkpoint_manager.save_checkpoint(BootstrapCheckpoint(
                        run_id=run_id,
                        entity=entity,
                        rows_processed=stats["rows_read"],
                        last_batch_index=batch_index,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                raise MemoryError(
                    f"Memory limit exceeded at batch {batch_index}. "
                    f"Current usage: {memory_monitor.get_current_usage_mb():.1f}MB, "
                    f"limit: {memory_monitor.max_memory_mb}MB. "
                    f"Checkpoint saved - you can resume with the same run_id."
                )

            # Validate batch
            if validate_batch_fn:
                # Check if function takes normalize_addresses parameter
                import inspect
                sig = inspect.signature(validate_batch_fn)
                if "normalize_addresses" in sig.parameters:
                    validation_result = validate_batch_fn(
                        batch, target_schema, normalize_addresses=normalize_addresses
                    )
                else:
                    validation_result = validate_batch_fn(batch, target_schema)

                valid_batch = validation_result.valid_batch
                stats["rows_quarantined"] += validation_result.rows_quarantined
            else:
                # No validation, but still need to cast to target schema
                table = pa.Table.from_batches([batch])
                # Cast each column to target schema
                columns = {}
                for field in target_schema:
                    if field.name in table.column_names:
                        col = table.column(field.name)
                        try:
                            col = col.cast(field.type)
                        except pa.ArrowInvalid:
                            pass  # Keep original if cast fails
                        columns[field.name] = col
                    else:
                        columns[field.name] = pa.nulls(table.num_rows, type=field.type)
                casted_table = pa.table(columns, schema=target_schema)
                valid_batch = casted_table.to_batches()[0] if casted_table.num_rows > 0 else batch.slice(0, 0)

            # Write valid rows to Parquet
            if valid_batch.num_rows > 0:
                writer.write_batch(valid_batch)
                stats["rows_written"] += valid_batch.num_rows

            # Periodic checkpoint save
            if (
                config
                and config.checkpoint_enabled
                and checkpoint_manager
                and run_id
                and entity
                and batch_index % config.checkpoint_interval == 0
            ):
                checkpoint_manager.save_checkpoint(BootstrapCheckpoint(
                    run_id=run_id,
                    entity=entity,
                    rows_processed=stats["rows_read"],
                    last_batch_index=batch_index,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

            # Explicit garbage collection after each batch
            gc.collect()

    logger.info(
        "streaming_complete",
        path=str(csv_path),
        rows_read=stats["rows_read"],
        rows_written=stats["rows_written"],
        rows_quarantined=stats["rows_quarantined"],
    )

    return stats


def _process_events_batch_tags(batch: pa.RecordBatch) -> pa.RecordBatch:
    """
    Process tags column in a batch for JSON parsing.

    This handles the JSON tags column in events CSV without loading the entire
    table into memory. Iterates through the batch but doesn't convert the whole
    table to Python.

    Args:
        batch: RecordBatch with potential "tags" column

    Returns:
        RecordBatch with tags parsed to list[str]
    """
    table = pa.Table.from_batches([batch])

    if "tags" not in table.column_names:
        return batch

    tags_col = table.column("tags")

    # Process tags - must iterate but only for this batch, not entire file
    processed_tags = []
    for i in range(tags_col.length()):
        value = tags_col[i].as_py()
        if value is None or value == "":
            processed_tags.append([])
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    processed_tags.append([str(t) for t in parsed])
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

    # Build new table with replaced column
    columns = {}
    for field_name in table.column_names:
        if field_name == "tags":
            columns[field_name] = new_tags_array
        else:
            columns[field_name] = table.column(field_name)

    new_table = pa.table(columns)
    return new_table.to_batches()[0]


def _stream_events_csv_to_parquet(
    csv_path: Path,
    output_path: Path | str,
    target_schema: pa.Schema,
    validate_batch_fn,
    batch_size: int = DEFAULT_BATCH_SIZE,
    config: BootstrapConfig | None = None,
    s3_filesystem: "pa.fs.S3FileSystem | None" = None,
) -> tuple[EventsLookup, dict[str, int]]:
    """
    Stream events CSV to Parquet with memory-efficient batch processing.

    Special handling for JSON tags column. Returns an EventsLookup for the
    markets join that can spill to disk if events are large.

    Args:
        csv_path: Path to events CSV
        output_path: Path for output Parquet file (local Path or S3 URI)
        target_schema: Target schema for events
        validate_batch_fn: Batch validation function
        batch_size: Rows per batch
        config: Bootstrap configuration for memory limits
        s3_filesystem: S3FileSystem for direct S3 writes

    Returns:
        Tuple of (EventsLookup for join, stats dict)
    """
    from polymkt.pipeline.normalize import normalize_timestamp

    # Use config values if provided
    if config is not None:
        batch_size = config.batch_size
        events_max_in_memory = config.events_max_in_memory
    else:
        events_max_in_memory = 100_000

    stats = {"rows_read": 0, "rows_written": 0, "rows_quarantined": 0}

    # Determine if writing to S3 or local
    output_str = str(output_path)
    is_s3_output = output_str.startswith("s3://")

    logger.info(
        "streaming_events_csv_to_parquet",
        input=str(csv_path),
        output=output_str,
        batch_size=batch_size,
        events_max_in_memory=events_max_in_memory,
        s3_streaming=is_s3_output,
    )

    # Use EventsLookup that can spill to disk for large datasets
    events_lookup = EventsLookup(max_in_memory=events_max_in_memory)

    with _open_parquet_writer(output_path, target_schema, s3_filesystem) as writer:
        for batch in _iter_csv_batches(csv_path, EVENTS_COLUMN_MAPPING, batch_size):
            stats["rows_read"] += batch.num_rows

            # Process tags JSON column
            batch_with_tags = _process_events_batch_tags(batch)

            # Normalize created_at timestamps in the batch
            table = pa.Table.from_batches([batch_with_tags])
            if "created_at" in table.column_names:
                created_at_col = table.column("created_at")
                normalized_timestamps = []
                for i in range(created_at_col.length()):
                    val = created_at_col[i].as_py()
                    normalized_timestamps.append(normalize_timestamp(val, "created_at"))
                table = table.set_column(
                    table.column_names.index("created_at"),
                    "created_at",
                    pa.array(normalized_timestamps, type=pa.timestamp("us", tz="UTC")),
                )
                batch_with_tags = table.to_batches()[0]

            # Validate batch
            if validate_batch_fn:
                validation_result = validate_batch_fn(batch_with_tags, target_schema)
                valid_batch = validation_result.valid_batch
                stats["rows_quarantined"] += validation_result.rows_quarantined
            else:
                valid_batch = batch_with_tags

            # Write valid rows to Parquet and add to lookup
            if valid_batch.num_rows > 0:
                writer.write_batch(valid_batch)
                stats["rows_written"] += valid_batch.num_rows

                # Add events to lookup for markets join
                valid_table = pa.Table.from_batches([valid_batch])
                if "event_id" in valid_table.column_names and "tags" in valid_table.column_names:
                    event_id_col = valid_table.column("event_id")
                    tags_col = valid_table.column("tags")
                    for i in range(valid_table.num_rows):
                        event_id = event_id_col[i].as_py()
                        tags = tags_col[i].as_py() or []
                        if event_id:
                            events_lookup.add_event(event_id, tags)

            # Explicit garbage collection after each batch
            gc.collect()

    # Finalize the lookup (commit any pending writes if disk-based)
    events_lookup.finalize()

    logger.info(
        "events_streaming_complete",
        path=str(csv_path),
        rows_read=stats["rows_read"],
        rows_written=stats["rows_written"],
        rows_quarantined=stats["rows_quarantined"],
        events_in_lookup=len(events_lookup),
        events_on_disk=events_lookup.use_disk,
    )

    # Garbage collection after processing
    gc.collect()

    return events_lookup, stats


def _add_tags_to_markets_batch(
    batch: pa.RecordBatch,
    events_lookup: EventsLookup | dict[str, list[str]],
    target_schema: pa.Schema,
) -> pa.RecordBatch:
    """
    Add tags to a markets batch by looking up event_id.

    Args:
        batch: Markets batch
        events_lookup: EventsLookup or dict mapping event_id to tags list
        target_schema: Target schema for output

    Returns:
        Batch with tags column added
    """
    table = pa.Table.from_batches([batch])

    if "event_id" not in table.column_names:
        # No event_id, add empty tags
        tags_array = pa.array([[]] * table.num_rows, type=pa.list_(pa.string()))
    else:
        event_id_col = table.column("event_id")
        tags_list = []
        for i in range(event_id_col.length()):
            event_id = event_id_col[i].as_py()
            # Support both EventsLookup and dict
            if isinstance(events_lookup, EventsLookup):
                tags = events_lookup.get_tags(event_id)
            else:
                tags = events_lookup.get(event_id, []) if event_id else []
            tags_list.append(tags)
        tags_array = pa.array(tags_list, type=pa.list_(pa.string()))

    # Add or replace tags column
    if "tags" in table.column_names:
        table = table.drop(["tags"])
    table = table.append_column("tags", tags_array)

    # Reorder columns to match target schema and cast types
    columns = {}
    for field in target_schema:
        if field.name in table.column_names:
            col = table.column(field.name)
            try:
                col = col.cast(field.type)
            except pa.ArrowInvalid:
                pass
            columns[field.name] = col
        else:
            columns[field.name] = pa.nulls(table.num_rows, type=field.type)

    final_table = pa.table(columns, schema=target_schema)
    return final_table.to_batches()[0] if final_table.num_rows > 0 else batch.slice(0, 0)


def _stream_markets_csv_to_parquet(
    csv_path: Path,
    output_path: Path | str,
    target_schema: pa.Schema,
    events_lookup: EventsLookup | None,
    validate_batch_fn,
    batch_size: int = DEFAULT_BATCH_SIZE,
    config: BootstrapConfig | None = None,
    s3_filesystem: "pa.fs.S3FileSystem | None" = None,
) -> dict[str, int]:
    """
    Stream markets CSV to Parquet with memory-efficient batch processing.

    Handles the events join in a memory-efficient way by:
    1. Using EventsLookup for tag lookup (supports disk spillover)
    2. Processing markets in batches
    3. Adding tags via lookup per batch

    Args:
        csv_path: Path to markets CSV
        output_path: Path for output Parquet file (local Path or S3 URI)
        target_schema: Target schema for markets
        events_lookup: EventsLookup for tag lookup (or None)
        validate_batch_fn: Batch validation function
        batch_size: Rows per batch
        config: Bootstrap configuration
        s3_filesystem: S3FileSystem for direct S3 writes

    Returns:
        Stats dict with rows_read, rows_written, rows_quarantined
    """
    from polymkt.pipeline.normalize import normalize_timestamp

    # Use config values if provided
    if config is not None:
        batch_size = config.batch_size

    stats = {"rows_read": 0, "rows_written": 0, "rows_quarantined": 0}

    # Determine if writing to S3 or local
    output_str = str(output_path)
    is_s3_output = output_str.startswith("s3://")

    logger.info(
        "streaming_markets_csv_to_parquet",
        input=str(csv_path),
        output=output_str,
        batch_size=batch_size,
        events_lookup_size=len(events_lookup) if events_lookup else 0,
        s3_streaming=is_s3_output,
    )

    # Create empty lookup if none provided
    if events_lookup is None:
        events_lookup = EventsLookup()

    with _open_parquet_writer(output_path, target_schema, s3_filesystem) as writer:
        for batch in _iter_csv_batches(csv_path, MARKETS_COLUMN_MAPPING, batch_size):
            stats["rows_read"] += batch.num_rows

            # Normalize timestamp columns in the batch
            table = pa.Table.from_batches([batch])
            for ts_col_name in ["created_at", "closed_time"]:
                if ts_col_name in table.column_names:
                    ts_col = table.column(ts_col_name)
                    normalized_timestamps = []
                    for i in range(ts_col.length()):
                        val = ts_col[i].as_py()
                        normalized_timestamps.append(normalize_timestamp(val, ts_col_name))
                    table = table.set_column(
                        table.column_names.index(ts_col_name),
                        ts_col_name,
                        pa.array(normalized_timestamps, type=pa.timestamp("us", tz="UTC")),
                    )
            batch = table.to_batches()[0]

            # Validate batch
            if validate_batch_fn:
                validation_result = validate_batch_fn(batch, target_schema)
                valid_batch = validation_result.valid_batch
                stats["rows_quarantined"] += validation_result.rows_quarantined
            else:
                valid_batch = batch

            if valid_batch.num_rows > 0:
                # Add tags from events lookup
                enriched_batch = _add_tags_to_markets_batch(
                    valid_batch, events_lookup, target_schema
                )
                writer.write_batch(enriched_batch)
                stats["rows_written"] += enriched_batch.num_rows

            # Explicit garbage collection after each batch
            gc.collect()

    logger.info(
        "markets_streaming_complete",
        path=str(csv_path),
        rows_read=stats["rows_read"],
        rows_written=stats["rows_written"],
        rows_quarantined=stats["rows_quarantined"],
    )

    # Garbage collection after processing
    gc.collect()

    return stats


def _read_csv_with_schema(
    csv_path: Path,
    target_schema: pa.Schema,
    column_mapping: dict[str, str],
) -> pa.Table:
    """Read a CSV file and normalize to target schema."""
    logger.info("reading_csv", path=str(csv_path))

    # Read CSV with type inference
    read_options = csv.ReadOptions()
    parse_options = csv.ParseOptions(newlines_in_values=True)
    # Don't auto-parse timestamps - we'll handle them manually to support +00 format
    convert_options = csv.ConvertOptions(
        timestamp_parsers=[],  # Disable auto-parsing, read as strings
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
                    # Use normalize_timestamp to handle various formats including +00
                    from polymkt.pipeline.normalize import normalize_timestamp
                    values = col.to_pylist()
                    normalized = [normalize_timestamp(v, field.name) for v in values]
                    col = pa.array(normalized, type=field.type)
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
    Uses a file-based DuckDB database to allow disk spilling for large datasets.

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

    # Use file-based DuckDB to allow disk spilling for large joins
    # Create a temp directory and a fresh db file path (not pre-existing)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db_path = Path(tmp_dir) / "join_temp.duckdb"
        conn = duckdb.connect(str(tmp_db_path))

        # Configure DuckDB for memory-efficient operation
        conn.execute("SET memory_limit='2GB'")
        conn.execute("SET temp_directory='/tmp'")

        # Register tables (DuckDB will reference Arrow data directly when possible)
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
    config: BootstrapConfig | None = None,
    resume_run_id: str | None = None,
    s3_streaming: bool | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    s3_region: str | None = None,
    s3_partitioned: bool = True,
    s3_partition_buffer_size: int = 50_000,
) -> BootstrapSummary:
    """
    Run the bootstrap import process.

    Reads existing CSV files and converts them to Parquet format,
    then creates DuckDB views over the Parquet files.

    Supports three output modes:
    1. Local: Write Parquet to local disk (default)
    2. S3 Streaming Monolithic: Write single Parquet files directly to S3
    3. S3 Streaming Partitioned: Write partitioned Parquet (year/month/day) directly to S3
       - This mode is memory-efficient and uploads partitions continuously
       - Use this for large datasets (100GB+) to avoid filling up local disk

    Args:
        markets_csv: Path to markets CSV file
        trades_csv: Path to trades CSV file
        order_filled_csv: Path to orderFilled CSV file
        events_csv: Path to events CSV file (for deriving market tags)
        parquet_dir: Directory for Parquet output (used for local mode)
        duckdb_path: Path to DuckDB database file
        metadata_db_path: Path to metadata SQLite database
        normalize_addresses: Whether to normalize Ethereum addresses (lowercase 0x)
        validate_data: Whether to validate and quarantine invalid rows
        partitioning_enabled: Whether to partition trades Parquet by year/month/day/hash_bucket
        hash_bucket_count: Number of hash buckets for market_id partitioning
        require_events_for_tags: If True, require events CSV for deriving market tags
        config: Bootstrap configuration for memory limits and checkpointing
        resume_run_id: Optional run ID to resume from checkpoint
        s3_streaming: If True, write directly to S3 (bypasses local disk)
        s3_bucket: S3 bucket for streaming output
        s3_prefix: S3 key prefix (e.g., 'raw/polymarket')
        s3_region: AWS region for S3
        s3_partitioned: If True with s3_streaming, write partitioned parquet to S3
            continuously (uploads as buffers fill, never stores all locally).
            Only applies to trades and order_filled. Default True.
        s3_partition_buffer_size: Rows per partition to buffer before uploading to S3.
            Lower = more frequent uploads, less memory. Default 50,000.

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

    # S3 streaming settings
    s3_streaming = s3_streaming if s3_streaming is not None else settings.s3_streaming
    s3_bucket = s3_bucket or settings.s3_bucket
    s3_prefix = s3_prefix or settings.s3_raw_prefix
    s3_region = s3_region or settings.s3_region

    # Get bootstrap config from settings if not provided
    if config is None:
        config = settings.get_bootstrap_config()

    # Initialize components
    run_id = resume_run_id or str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    timer_start = perf_counter()

    # Initialize checkpoint manager
    checkpoint_manager: CheckpointManager | None = None
    if config.checkpoint_enabled:
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    # Initialize S3 filesystem if streaming to S3
    s3_filesystem: pa.fs.S3FileSystem | None = None
    if s3_streaming:
        if not s3_bucket:
            raise ValueError(
                "s3_bucket is required when s3_streaming=True. "
                "Set POLYMKT_S3_BUCKET environment variable or pass s3_bucket parameter."
            )
        s3_filesystem = _get_s3_filesystem(s3_region)
        logger.info(
            "s3_streaming_enabled",
            bucket=s3_bucket,
            prefix=s3_prefix,
            region=s3_region,
        )

    logger.info(
        "bootstrap_starting",
        run_id=run_id,
        s3_streaming=s3_streaming,
        s3_partitioned=s3_partitioned if s3_streaming else False,
        s3_partition_buffer_size=s3_partition_buffer_size if s3_streaming and s3_partitioned else None,
        batch_size=config.batch_size,
        max_memory_mb=config.max_memory_mb,
        checkpoint_enabled=config.checkpoint_enabled,
        resume=resume_run_id is not None,
    )

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

    # Track events lookup for cleanup
    events_lookup: EventsLookup | None = None

    try:
        # Process events first (needed for market tags join)
        # Use streaming to avoid memory issues with large files
        if events_csv.exists():
            logger.info("processing_events_streaming", path=str(events_csv))

            # Determine output path (local or S3)
            if s3_streaming and s3_bucket:
                events_output_path: str | Path = f"s3://{s3_bucket}/{s3_prefix}/events.parquet"
            else:
                events_output_path = parquet_dir / "events.parquet"

            events_lookup, events_stats = _stream_events_csv_to_parquet(
                csv_path=events_csv,
                output_path=events_output_path,
                target_schema=EVENTS_SCHEMA,
                validate_batch_fn=validate_events_batch if validate_data else None,
                config=config,
                s3_filesystem=s3_filesystem,
            )

            rows_read["events"] = events_stats["rows_read"]
            rows_written["events"] = events_stats["rows_written"]
            rows_quarantined["events"] = events_stats["rows_quarantined"]
            parquet_files.append(str(events_output_path))

            # Garbage collection after events processing
            gc.collect()
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

        # Process markets using streaming (memory-efficient for large files)
        # The events join is handled within the streaming function
        if markets_csv.exists():
            logger.info("processing_markets_streaming", path=str(markets_csv))

            # Determine output path (local or S3)
            if s3_streaming and s3_bucket:
                markets_output_path: str | Path = f"s3://{s3_bucket}/{s3_prefix}/markets.parquet"
            else:
                markets_output_path = parquet_dir / "markets.parquet"

            markets_stats = _stream_markets_csv_to_parquet(
                csv_path=markets_csv,
                output_path=markets_output_path,
                target_schema=MARKETS_SCHEMA,
                events_lookup=events_lookup,
                validate_batch_fn=validate_markets_batch if validate_data else None,
                config=config,
                s3_filesystem=s3_filesystem,
            )

            rows_read["markets"] = markets_stats["rows_read"]
            rows_written["markets"] = markets_stats["rows_written"]
            rows_quarantined["markets"] = markets_stats["rows_quarantined"]
            parquet_files.append(str(markets_output_path))

            # Clean up events lookup after markets processing (no longer needed)
            if events_lookup is not None:
                events_lookup.close()
                events_lookup = None

            # Garbage collection after markets processing
            gc.collect()
        else:
            logger.warning("csv_not_found", path=str(markets_csv))
            rows_read["markets"] = 0
            rows_written["markets"] = 0
            rows_quarantined["markets"] = 0

        # Process trades using streaming (memory-efficient for large files)
        if trades_csv.exists():
            if s3_streaming and s3_bucket and s3_partitioned:
                # S3 PARTITIONED STREAMING: Memory-efficient, uploads continuously
                # This mode:
                # 1. Reads CSV in batches (never loads entire file)
                # 2. Partitions by year/month/day
                # 3. Uploads to S3 as soon as partition buffer fills
                # 4. Never stores all data locally - prevents disk fill
                logger.info(
                    "processing_trades_partitioned_s3_streaming",
                    path=str(trades_csv),
                    buffer_size=s3_partition_buffer_size,
                )

                trades_stats = _stream_csv_to_partitioned_s3(
                    csv_path=trades_csv,
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix,
                    entity="trades",
                    target_schema=TRADES_SCHEMA,
                    column_mapping=TRADES_COLUMN_MAPPING,
                    validate_batch_fn=validate_trades_batch if validate_data else None,
                    normalize_addresses=normalize_addresses,
                    batch_size=config.batch_size,
                    buffer_size=s3_partition_buffer_size,
                    s3_region=s3_region,
                    timestamp_column="timestamp",
                    config=config,
                )

                rows_read["trades"] = trades_stats["rows_read"]
                rows_written["trades"] = trades_stats["rows_written"]
                rows_quarantined["trades"] = trades_stats["rows_quarantined"]
                parquet_files.append(f"s3://{s3_bucket}/{s3_prefix}/trades/")

            elif partitioning_enabled and not s3_streaming:
                # Use original approach for partitioned writes (local only)
                # WARNING: This loads entire table into memory - avoid for 100GB+ datasets
                logger.warning(
                    "partitioned_trades_memory_warning",
                    message="Partitioned mode loads entire trades CSV into memory. "
                            "Consider using non-partitioned mode for large datasets.",
                )
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
                else:
                    rows_quarantined["trades"] = 0

                output_path = parquet_writer.write_trades(trades_table)
                rows_written["trades"] = trades_table.num_rows
                parquet_files.append(str(output_path))
            else:
                # Use streaming for monolithic writes (memory-efficient, supports S3)
                logger.info("processing_trades_streaming", path=str(trades_csv))

                # Determine output path (local or S3)
                if s3_streaming and s3_bucket:
                    trades_output_path: str | Path = f"s3://{s3_bucket}/{s3_prefix}/trades.parquet"
                else:
                    trades_output_path = parquet_dir / "trades.parquet"

                trades_stats = _stream_csv_to_parquet(
                    csv_path=trades_csv,
                    output_path=trades_output_path,
                    target_schema=TRADES_SCHEMA,
                    column_mapping=TRADES_COLUMN_MAPPING,
                    validate_batch_fn=validate_trades_batch if validate_data else None,
                    normalize_addresses=normalize_addresses,
                    config=config,
                    checkpoint_manager=checkpoint_manager,
                    run_id=run_id,
                    entity="trades",
                    s3_filesystem=s3_filesystem,
                )

                rows_read["trades"] = trades_stats["rows_read"]
                rows_written["trades"] = trades_stats["rows_written"]
                rows_quarantined["trades"] = trades_stats["rows_quarantined"]
                parquet_files.append(str(trades_output_path))

            # Garbage collection after trades processing
            gc.collect()
        else:
            logger.warning("csv_not_found", path=str(trades_csv))
            rows_read["trades"] = 0
            rows_written["trades"] = 0
            rows_quarantined["trades"] = 0

        # Process order filled using streaming (memory-efficient for large files)
        if order_filled_csv.exists():
            if s3_streaming and s3_bucket and s3_partitioned:
                # S3 PARTITIONED STREAMING: Memory-efficient, uploads continuously
                logger.info(
                    "processing_order_filled_partitioned_s3_streaming",
                    path=str(order_filled_csv),
                    buffer_size=s3_partition_buffer_size,
                )

                order_filled_stats = _stream_csv_to_partitioned_s3(
                    csv_path=order_filled_csv,
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix,
                    entity="order_filled",
                    target_schema=ORDER_FILLED_SCHEMA,
                    column_mapping=ORDER_FILLED_COLUMN_MAPPING,
                    validate_batch_fn=validate_order_filled_batch if validate_data else None,
                    normalize_addresses=normalize_addresses,
                    batch_size=config.batch_size,
                    buffer_size=s3_partition_buffer_size,
                    s3_region=s3_region,
                    timestamp_column="timestamp",
                    config=config,
                )

                rows_read["order_filled"] = order_filled_stats["rows_read"]
                rows_written["order_filled"] = order_filled_stats["rows_written"]
                rows_quarantined["order_filled"] = order_filled_stats["rows_quarantined"]
                parquet_files.append(f"s3://{s3_bucket}/{s3_prefix}/order_filled/")

            else:
                # Monolithic streaming (local or S3 non-partitioned)
                logger.info("processing_order_filled_streaming", path=str(order_filled_csv))

                # Determine output path (local or S3)
                if s3_streaming and s3_bucket:
                    order_filled_output_path: str | Path = f"s3://{s3_bucket}/{s3_prefix}/order_filled.parquet"
                else:
                    order_filled_output_path = parquet_dir / "order_filled.parquet"

                order_filled_stats = _stream_csv_to_parquet(
                    csv_path=order_filled_csv,
                    output_path=order_filled_output_path,
                    target_schema=ORDER_FILLED_SCHEMA,
                    column_mapping=ORDER_FILLED_COLUMN_MAPPING,
                    validate_batch_fn=validate_order_filled_batch if validate_data else None,
                    normalize_addresses=normalize_addresses,
                    config=config,
                    checkpoint_manager=checkpoint_manager,
                    run_id=run_id,
                    entity="order_filled",
                    s3_filesystem=s3_filesystem,
                )

                rows_read["order_filled"] = order_filled_stats["rows_read"]
                rows_written["order_filled"] = order_filled_stats["rows_written"]
                rows_quarantined["order_filled"] = order_filled_stats["rows_quarantined"]
                parquet_files.append(str(order_filled_output_path))

            # Garbage collection after order_filled processing
            gc.collect()
        else:
            logger.warning("csv_not_found", path=str(order_filled_csv))
            rows_read["order_filled"] = 0
            rows_written["order_filled"] = 0
            rows_quarantined["order_filled"] = 0

        # Create DuckDB views (only for local Parquet, skip for S3 streaming)
        if not s3_streaming:
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
        else:
            logger.info(
                "skipping_duckdb_views",
                reason="S3 streaming mode - Parquet files are on S3, not local disk",
            )

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

        # Clear checkpoints on successful completion
        if checkpoint_manager is not None:
            checkpoint_manager.clear_checkpoints(run_id)

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
        # Clean up events lookup if still open
        if events_lookup is not None:
            events_lookup.close()

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


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for bootstrap pipeline.

    Usage:
        # Standard local bootstrap
        python -m polymkt.pipeline.bootstrap

        # S3 partitioned streaming (recommended for large datasets)
        python -m polymkt.pipeline.bootstrap --s3-streaming --s3-partitioned

        # With custom buffer size (lower = more frequent S3 uploads, less memory)
        python -m polymkt.pipeline.bootstrap --s3-streaming --s3-partitioned --buffer-size 25000

        # Monolithic S3 streaming (single file, not partitioned)
        python -m polymkt.pipeline.bootstrap --s3-streaming --no-s3-partitioned
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Bootstrap CSV data into Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local bootstrap (default)
  python -m polymkt.pipeline.bootstrap

  # S3 streaming with partitioned writes (memory-efficient, uploads continuously)
  python -m polymkt.pipeline.bootstrap --s3-streaming --s3-partitioned

  # Custom S3 bucket
  python -m polymkt.pipeline.bootstrap --s3-streaming --bucket my-bucket --prefix raw/data

  # Lower buffer size for constrained memory (flushes more often)
  python -m polymkt.pipeline.bootstrap --s3-streaming --buffer-size 25000

  # Process only trades
  python -m polymkt.pipeline.bootstrap --entity trades --s3-streaming

Environment variables:
  POLYMKT_S3_BUCKET       S3 bucket name
  POLYMKT_S3_REGION       AWS region (default: us-east-1)
  POLYMKT_S3_RAW_PREFIX   S3 prefix for raw data
        """,
    )

    parser.add_argument(
        "--s3-streaming",
        action="store_true",
        help="Stream directly to S3 (bypasses local disk storage)",
    )
    parser.add_argument(
        "--s3-partitioned",
        action="store_true",
        default=True,
        help="Write partitioned parquet to S3 (year/month/day). Default: True",
    )
    parser.add_argument(
        "--no-s3-partitioned",
        action="store_true",
        help="Disable partitioned writes (write monolithic files)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50_000,
        help="Rows per partition to buffer before S3 upload (default: 50000). "
             "Lower = more uploads, less memory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows per CSV batch (default: 10000)",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=4000,
        help="Max memory in MB before triggering flush (default: 4000)",
    )
    parser.add_argument(
        "--bucket", "-b",
        help="S3 bucket (uses POLYMKT_S3_BUCKET if not provided)",
    )
    parser.add_argument(
        "--prefix", "-p",
        default="raw/polymarket",
        help="S3 prefix (default: raw/polymarket)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--entity",
        choices=["all", "trades", "order_filled", "markets", "events"],
        default="all",
        help="Entity to process (default: all)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable data validation",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Handle partitioned flag
    s3_partitioned = args.s3_partitioned and not args.no_s3_partitioned

    # Build config
    config = BootstrapConfig(
        batch_size=args.batch_size,
        max_memory_mb=args.max_memory_mb,
        memory_check_interval=10,
        checkpoint_enabled=True,
        checkpoint_interval=100,
        checkpoint_dir=Path(".polymkt_checkpoints"),
        events_max_in_memory=100_000,
    )

    # Determine which CSVs to process
    # Use non-existent paths for entities we want to skip (run_bootstrap checks .exists())
    skip_path = Path("/nonexistent/skip.csv")
    if args.entity == "all":
        trades_csv = settings.trades_csv
        order_filled_csv = settings.order_filled_csv
        markets_csv = settings.markets_csv
        events_csv = settings.events_csv
    elif args.entity == "trades":
        trades_csv = settings.trades_csv
        order_filled_csv = skip_path
        markets_csv = skip_path
        events_csv = skip_path
    elif args.entity == "order_filled":
        trades_csv = skip_path
        order_filled_csv = settings.order_filled_csv
        markets_csv = skip_path
        events_csv = skip_path
    elif args.entity == "markets":
        trades_csv = skip_path
        order_filled_csv = skip_path
        markets_csv = settings.markets_csv
        events_csv = settings.events_csv  # Need events for tags
    elif args.entity == "events":
        trades_csv = skip_path
        order_filled_csv = skip_path
        markets_csv = skip_path
        events_csv = settings.events_csv
    else:
        trades_csv = skip_path
        order_filled_csv = skip_path
        markets_csv = skip_path
        events_csv = skip_path

    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print("Bootstrap Configuration")
        print("=" * 60)
        print(f"S3 Streaming:      {args.s3_streaming}")
        print(f"S3 Partitioned:    {s3_partitioned}")
        if args.s3_streaming:
            print(f"S3 Bucket:         {args.bucket or settings.s3_bucket}")
            print(f"S3 Prefix:         {args.prefix}")
            print(f"S3 Region:         {args.region}")
            print(f"Buffer Size:       {args.buffer_size:,} rows/partition")
        print(f"Batch Size:        {args.batch_size:,} rows")
        print(f"Max Memory:        {args.max_memory_mb:,} MB")
        print(f"Entity:            {args.entity}")
        print(f"Validation:        {'disabled' if args.no_validate else 'enabled'}")
        print("=" * 60)

    # Run bootstrap
    try:
        result = run_bootstrap(
            trades_csv=trades_csv,
            order_filled_csv=order_filled_csv,
            markets_csv=markets_csv,
            events_csv=events_csv,
            s3_streaming=args.s3_streaming,
            s3_bucket=args.bucket,
            s3_prefix=args.prefix,
            s3_region=args.region,
            s3_partitioned=s3_partitioned,
            s3_partition_buffer_size=args.buffer_size,
            validate_data=not args.no_validate,
            config=config,
        )

        # Print results
        if not args.quiet:
            print("\n" + "=" * 60)
            print("Bootstrap Complete")
            print("=" * 60)
            print(f"Run ID:            {result.run_id}")
            print(f"Status:            {result.status}")
            print(f"Duration:          {result.duration_seconds:.1f}s")
            print(f"Trades:            {result.trades_rows:,} rows")
            print(f"Order Filled:      {result.order_filled_rows:,} rows")
            print(f"Markets:           {result.markets_rows:,} rows")
            print(f"Events:            {result.events_rows:,} rows")
            if result.rows_quarantined:
                print(f"Quarantined:       {result.rows_quarantined}")
            print(f"Output files:      {len(result.parquet_files)}")
            for f in result.parquet_files[:5]:
                print(f"  - {f}")
            if len(result.parquet_files) > 5:
                print(f"  ... and {len(result.parquet_files) - 5} more")

    except Exception as e:
        print(f"\nBootstrap failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
