"""Memory-efficient S3 parquet partitioner.

Transform monolithic parquet files on S3 into partitioned datasets without
loading entire files into memory. Streams batches, partitions by timestamp,
and writes to year/month/day structure.

Source structure:
    s3://bucket/raw/polymarket/trades.parquet (single file)

Target structure:
    s3://bucket/raw/polymarket/trades/
    ├── year=2024/month=01/trades_20240101.parquet
    ├── year=2024/month=01/trades_20240102.parquet
    └── ...

Usage:
    python -m polymkt.pipeline.s3_rewrite --entity trades
    python -m polymkt.pipeline.s3_rewrite --entity order_filled
    python -m polymkt.pipeline.s3_rewrite --entity trades --resume
    python -m polymkt.pipeline.s3_rewrite --entity trades --dry-run
"""

import gc
import json
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import structlog

from polymkt.config import RewriteConfig, settings
from polymkt.storage.parquet import ORDER_FILLED_SCHEMA, TRADES_SCHEMA

logger = structlog.get_logger()

# Type alias for transform functions
TransformFn = Callable[[pa.RecordBatch], pa.RecordBatch]
ProgressCallback = Callable[["PartitionProgress"], None]


# =============================================================================
# Transform Functions
# =============================================================================


def fix_order_filled_timestamps(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Fix order_filled timestamps that are stored as seconds but read as microseconds.

    The order_filled.parquet file has timestamps stored as Unix seconds, but the
    schema declares them as timestamp[us] (microseconds). This causes dates to
    appear as 1970 instead of 2022+.

    Fix: multiply by 1,000,000 to convert seconds -> microseconds.
    """
    table = pa.Table.from_batches([batch])

    # Get timestamp column and cast to int64
    ts_col = table.column("timestamp")
    ts_int = ts_col.cast(pa.int64())

    # Multiply by 1,000,000 to convert seconds to microseconds
    ts_fixed = pc.multiply(ts_int, 1_000_000)

    # Cast back to timestamp[us, tz=UTC]
    ts_corrected = ts_fixed.cast(pa.timestamp("us", tz="UTC"))

    # Replace column
    col_idx = table.schema.get_field_index("timestamp")
    table = table.set_column(col_idx, "timestamp", ts_corrected)

    return table.to_batches()[0]


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class PartitionCheckpoint:
    """Checkpoint state for resumable partition operations."""

    run_id: str
    entity: str
    rows_processed: int
    batches_processed: int
    partitions_written: dict[str, int]  # partition_key -> row_count
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PartitionCheckpoint":
        return cls(**data)


@dataclass
class PartitionProgress:
    """Progress information for partition operation."""

    entity: str
    source_file: str
    rows_processed: int
    batches_processed: int
    partitions_created: int
    memory_usage_mb: float
    elapsed_seconds: float
    rows_per_second: float


@dataclass
class PartitionSummary:
    """Summary of a complete partition operation."""

    run_id: str
    entity: str
    source_file: str
    total_rows: int
    partitions_created: int
    files_created: int
    duration_seconds: float
    success: bool
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Memory Monitor
# =============================================================================


class MemoryMonitor:
    """Monitor system memory and trigger GC when needed."""

    def __init__(self, max_memory_mb: int, check_interval: int = 10) -> None:
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.batch_count = 0

    def check_and_throttle(self) -> bool:
        """Check memory and GC if needed. Returns True if critically exceeded."""
        self.batch_count += 1
        if self.batch_count % self.check_interval != 0:
            return False

        mem_mb = self.get_current_usage_mb()

        if mem_mb > self.max_memory_mb * 0.8:
            gc.collect()
            mem_after = self.get_current_usage_mb()
            logger.warning(
                "memory_pressure",
                before_mb=round(mem_mb, 1),
                after_mb=round(mem_after, 1),
                limit_mb=self.max_memory_mb,
            )
            return mem_after > self.max_memory_mb * 0.95

        return False

    def get_current_usage_mb(self) -> float:
        return psutil.Process().memory_info().rss / (1024 * 1024)


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """Manage checkpoints for resume capability."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, run_id: str, entity: str) -> Path:
        return self.checkpoint_dir / f"{run_id}_{entity}.json"

    def save(self, checkpoint: PartitionCheckpoint) -> None:
        path = self._get_path(checkpoint.run_id, checkpoint.entity)
        with open(path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def load(self, run_id: str, entity: str) -> PartitionCheckpoint | None:
        path = self._get_path(run_id, entity)
        if not path.exists():
            return None
        with open(path) as f:
            return PartitionCheckpoint.from_dict(json.load(f))

    def clear(self, run_id: str, entity: str) -> None:
        path = self._get_path(run_id, entity)
        if path.exists():
            path.unlink()

    def get_latest_run_id(self, entity: str) -> str | None:
        checkpoints = list(self.checkpoint_dir.glob(f"*_{entity}.json"))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0].stem.rsplit("_", 1)[0]


# =============================================================================
# Partition Writer - Manages multiple output files
# =============================================================================


class PartitionWriter:
    """Manages writing to multiple partitioned parquet files.

    Buffers rows by partition key (year/month/day), flushes when buffer
    exceeds threshold or when flush_all() is called.
    """

    def __init__(
        self,
        s3fs: pafs.S3FileSystem,
        output_prefix: str,  # e.g., "bucket/staging/polymarket/trades"
        entity: str,
        schema: pa.Schema,
        buffer_size: int = 50_000,  # Rows per partition before flush
    ) -> None:
        self.s3fs = s3fs
        self.output_prefix = output_prefix
        self.entity = entity
        self.schema = schema
        self.buffer_size = buffer_size

        # Buffers: partition_key -> list of RecordBatches
        self.buffers: dict[str, list[pa.RecordBatch]] = defaultdict(list)
        self.buffer_rows: dict[str, int] = defaultdict(int)

        # Track written files
        self.files_written: dict[str, list[str]] = defaultdict(list)  # partition -> [files]
        self.rows_written: dict[str, int] = defaultdict(int)  # partition -> row_count

    def add_batch(self, batch: pa.RecordBatch) -> None:
        """Add a batch to be partitioned and written.

        The batch must have a 'timestamp' column for partitioning.
        """
        if batch.num_rows == 0:
            return

        # Convert to table for easier manipulation
        table = pa.Table.from_batches([batch])

        # Extract partition keys from timestamp
        timestamps = table.column("timestamp")

        # Compute year, month, day arrays
        years = pc.year(timestamps)
        months = pc.month(timestamps)
        days = pc.day(timestamps)

        # Build partition keys for each row
        # We'll group by unique (year, month, day) combinations
        partition_keys = [
            f"year={y}/month={m:02d}/day={d:02d}"
            for y, m, d in zip(
                years.to_pylist(),
                months.to_pylist(),
                days.to_pylist(),
            )
        ]

        # Group rows by partition
        partition_indices: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(partition_keys):
            partition_indices[key].append(i)

        # Add rows to appropriate buffers
        for partition_key, indices in partition_indices.items():
            partition_table = table.take(indices)
            partition_batch = partition_table.to_batches()[0]

            self.buffers[partition_key].append(partition_batch)
            self.buffer_rows[partition_key] += partition_batch.num_rows

            # Flush if buffer is large enough
            if self.buffer_rows[partition_key] >= self.buffer_size:
                self._flush_partition(partition_key)

    def _flush_partition(self, partition_key: str) -> None:
        """Flush buffered data for a partition to S3."""
        if not self.buffers[partition_key]:
            return

        # Combine all batches
        table = pa.Table.from_batches(self.buffers[partition_key], schema=self.schema)
        row_count = table.num_rows

        # Generate filename with timestamp for uniqueness
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.entity}_{ts}_{row_count}.parquet"
        output_path = f"{self.output_prefix}/{partition_key}/{filename}"

        # Ensure directory exists and write
        dir_path = f"{self.output_prefix}/{partition_key}"
        self.s3fs.create_dir(dir_path)

        pq.write_table(
            table,
            output_path,
            filesystem=self.s3fs,
            compression="zstd",
        )

        # Track
        self.files_written[partition_key].append(output_path)
        self.rows_written[partition_key] += row_count

        logger.debug(
            "partition_flushed",
            partition=partition_key,
            rows=row_count,
            file=filename,
        )

        # Clear buffer
        self.buffers[partition_key] = []
        self.buffer_rows[partition_key] = 0

        # Help GC
        del table
        gc.collect()

    def flush_all(self) -> None:
        """Flush all remaining buffered data."""
        for partition_key in list(self.buffers.keys()):
            if self.buffers[partition_key]:
                self._flush_partition(partition_key)

    def get_stats(self) -> dict:
        """Get statistics about written data."""
        return {
            "partitions": len(self.rows_written),
            "files": sum(len(files) for files in self.files_written.values()),
            "rows": sum(self.rows_written.values()),
            "partition_details": dict(self.rows_written),
        }


# =============================================================================
# S3 Parquet Partitioner
# =============================================================================


class S3ParquetPartitioner:
    """Partition monolithic S3 parquet files into year/month/day structure."""

    SCHEMAS = {
        "trades": TRADES_SCHEMA,
        "order_filled": ORDER_FILLED_SCHEMA,
    }

    def __init__(self, config: RewriteConfig) -> None:
        self.config = config
        self.s3fs = pafs.S3FileSystem(region=config.s3_region)
        self.memory_monitor = MemoryMonitor(
            config.max_memory_mb,
            config.memory_check_interval,
        )
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_dir)

    def partition_entity(
        self,
        entity: str,
        transform_fn: TransformFn | None = None,
        resume_run_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        dry_run: bool = False,
    ) -> PartitionSummary:
        """Partition a monolithic parquet file into year/month/day structure.

        Args:
            entity: 'trades' or 'order_filled'
            transform_fn: Optional function to transform each batch
            resume_run_id: Resume from checkpoint (not fully implemented yet)
            progress_callback: Callback for progress updates
            dry_run: If True, analyze source file without writing

        Returns:
            PartitionSummary with stats
        """
        if entity not in self.SCHEMAS:
            raise ValueError(f"Unknown entity: {entity}")

        schema = self.SCHEMAS[entity]
        start_time = perf_counter()
        run_id = resume_run_id or str(uuid.uuid4())[:8]

        # Source file path
        source_path = f"{self.config.s3_bucket}/{self.config.s3_prefix}/{entity}.parquet"
        staging_prefix = f"{self.config.s3_bucket}/{self.config.staging_prefix}/{entity}"

        logger.info(
            "starting_partition",
            run_id=run_id,
            entity=entity,
            source=source_path,
            staging=staging_prefix,
            dry_run=dry_run,
        )

        # Check source file exists
        try:
            file_info = self.s3fs.get_file_info(source_path)
            if file_info.type != pafs.FileType.File:
                raise FileNotFoundError(f"Source file not found: {source_path}")
            source_size_gb = file_info.size / (1024**3)
            logger.info("source_file_found", size_gb=round(source_size_gb, 2))
        except Exception as e:
            return PartitionSummary(
                run_id=run_id,
                entity=entity,
                source_file=source_path,
                total_rows=0,
                partitions_created=0,
                files_created=0,
                duration_seconds=perf_counter() - start_time,
                success=False,
                errors=[f"Source file error: {e}"],
            )

        if dry_run:
            return self._dry_run(run_id, entity, source_path, start_time, transform_fn)

        # Initialize partition writer
        writer = PartitionWriter(
            self.s3fs,
            staging_prefix,
            entity,
            schema,
            buffer_size=50_000,  # Flush every 50k rows per partition
        )

        # Open source file for streaming
        source_file = pq.ParquetFile(source_path, filesystem=self.s3fs)
        total_rows = source_file.metadata.num_rows

        rows_processed = 0
        batches_processed = 0
        errors: list[str] = []

        try:
            for batch in source_file.iter_batches(batch_size=self.config.batch_size):
                # Apply transform if provided
                if transform_fn:
                    batch = transform_fn(batch)

                # Cast to schema
                table = pa.Table.from_batches([batch])
                table = table.cast(schema)
                batch = table.to_batches()[0]

                # Add to partition writer (handles buffering and flushing)
                writer.add_batch(batch)

                rows_processed += batch.num_rows
                batches_processed += 1

                # Progress callback
                if progress_callback and batches_processed % 10 == 0:
                    elapsed = perf_counter() - start_time
                    progress = PartitionProgress(
                        entity=entity,
                        source_file=source_path,
                        rows_processed=rows_processed,
                        batches_processed=batches_processed,
                        partitions_created=len(writer.rows_written),
                        memory_usage_mb=self.memory_monitor.get_current_usage_mb(),
                        elapsed_seconds=elapsed,
                        rows_per_second=rows_processed / elapsed if elapsed > 0 else 0,
                    )
                    progress_callback(progress)

                # Memory check
                if self.memory_monitor.check_and_throttle():
                    logger.warning("memory_pressure_flushing")
                    writer.flush_all()
                    gc.collect()

                # Checkpoint periodically
                if (
                    self.config.checkpoint_enabled
                    and batches_processed % self.config.checkpoint_interval == 0
                ):
                    ckpt = PartitionCheckpoint(
                        run_id=run_id,
                        entity=entity,
                        rows_processed=rows_processed,
                        batches_processed=batches_processed,
                        partitions_written=dict(writer.rows_written),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    self.checkpoint_mgr.save(ckpt)

        except Exception as e:
            errors.append(f"Processing error: {e}")
            logger.error("partition_error", error=str(e))

        # Flush remaining data
        writer.flush_all()

        # Get final stats
        stats = writer.get_stats()
        duration = perf_counter() - start_time

        # Clear checkpoint on success
        if not errors:
            self.checkpoint_mgr.clear(run_id, entity)

        summary = PartitionSummary(
            run_id=run_id,
            entity=entity,
            source_file=source_path,
            total_rows=rows_processed,
            partitions_created=stats["partitions"],
            files_created=stats["files"],
            duration_seconds=duration,
            success=len(errors) == 0,
            errors=errors,
        )

        logger.info(
            "partition_complete",
            run_id=run_id,
            entity=entity,
            rows=rows_processed,
            partitions=stats["partitions"],
            files=stats["files"],
            duration=round(duration, 2),
            rows_per_sec=round(rows_processed / duration) if duration > 0 else 0,
        )

        return summary

    def _dry_run(
        self,
        run_id: str,
        entity: str,
        source_path: str,
        start_time: float,
        transform_fn: TransformFn | None = None,
    ) -> PartitionSummary:
        """Analyze source file without writing."""
        source_file = pq.ParquetFile(source_path, filesystem=self.s3fs)
        metadata = source_file.metadata

        print(f"\n=== Dry Run Analysis: {entity} ===")
        print(f"Source: s3://{source_path}")
        print(f"Total rows: {metadata.num_rows:,}")
        print(f"Row groups: {metadata.num_row_groups}")
        print(f"Schema columns: {metadata.schema.names}")

        # Sample first batch to see date range
        first_batch = next(source_file.iter_batches(batch_size=1000))

        # Apply transform if provided (e.g., timestamp fix)
        if transform_fn:
            first_batch = transform_fn(first_batch)
            print("\n[Transform applied to sample]")

        timestamps = first_batch.column("timestamp")
        min_ts = pc.min(timestamps).as_py()
        max_ts = pc.max(timestamps).as_py()

        print(f"\nSample date range (first 1000 rows):")
        print(f"  Min: {min_ts}")
        print(f"  Max: {max_ts}")

        # Estimate partitions
        if min_ts and max_ts:
            months_span = (max_ts.year - min_ts.year) * 12 + (max_ts.month - min_ts.month) + 1
            print(f"\nEstimated partitions: ~{months_span * 30} (days)")

        print(f"\nTarget: s3://{self.config.s3_bucket}/{self.config.staging_prefix}/{entity}/")
        print("Structure: year=YYYY/month=MM/day=DD/{entity}_YYYYMMDD_HHMMSS_N.parquet")

        return PartitionSummary(
            run_id=run_id,
            entity=entity,
            source_file=source_path,
            total_rows=metadata.num_rows,
            partitions_created=0,
            files_created=0,
            duration_seconds=perf_counter() - start_time,
            success=True,
        )

    def swap_staging_to_final(self, entity: str) -> None:
        """Move staging files to final location.

        1. Delete old partitioned data (if exists)
        2. Move staging to final location
        """
        staging_path = f"{self.config.s3_bucket}/{self.config.staging_prefix}/{entity}"
        final_path = f"{self.config.s3_bucket}/{self.config.s3_prefix}/{entity}"

        logger.info("starting_swap", staging=staging_path, final=final_path)

        # List all staging files
        try:
            selector = pafs.FileSelector(staging_path, recursive=True)
            staging_files = self.s3fs.get_file_info(selector)
        except Exception as e:
            logger.error("failed_to_list_staging", error=str(e))
            raise

        files_moved = 0
        for file_info in staging_files:
            if file_info.type != pafs.FileType.File:
                continue
            if not file_info.path.endswith(".parquet"):
                continue

            # Compute final path
            relative_path = file_info.path.replace(staging_path, "")
            final_file_path = f"{final_path}{relative_path}"

            # Ensure directory exists
            final_dir = "/".join(final_file_path.split("/")[:-1])
            self.s3fs.create_dir(final_dir)

            # Copy to final
            self.s3fs.copy_file(file_info.path, final_file_path)
            files_moved += 1

            # Delete staging file
            if self.config.delete_staging_on_success:
                self.s3fs.delete_file(file_info.path)

        logger.info("swap_complete", files_moved=files_moved)


# =============================================================================
# Entry Points
# =============================================================================


def run_s3_partition(
    entity: str,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    staging_prefix: str | None = None,
    s3_region: str | None = None,
    transform_fn: TransformFn | None = None,
    batch_size: int = 10_000,
    max_memory_mb: int = 6000,
    resume_run_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    dry_run: bool = False,
    auto_swap: bool = False,
) -> PartitionSummary:
    """Main entry point for S3 parquet partitioning.

    Transforms monolithic parquet files into partitioned datasets.

    Args:
        entity: 'trades' or 'order_filled'
        s3_bucket: S3 bucket
        s3_prefix: S3 prefix for source files
        staging_prefix: Staging prefix for output
        s3_region: AWS region
        transform_fn: Optional batch transform function
        batch_size: Rows per batch
        max_memory_mb: Memory limit
        resume_run_id: Resume from checkpoint
        progress_callback: Progress callback
        dry_run: Analyze without writing
        auto_swap: Swap staging to final on success

    Returns:
        PartitionSummary
    """
    base_config = settings.get_rewrite_config()

    config = RewriteConfig(
        s3_bucket=s3_bucket or base_config.s3_bucket,
        s3_prefix=s3_prefix or base_config.s3_prefix,
        staging_prefix=staging_prefix or base_config.staging_prefix,
        s3_region=s3_region or base_config.s3_region,
        batch_size=batch_size,
        max_memory_mb=max_memory_mb,
        checkpoint_enabled=base_config.checkpoint_enabled,
        checkpoint_interval=base_config.checkpoint_interval,
        checkpoint_dir=base_config.checkpoint_dir,
    )

    partitioner = S3ParquetPartitioner(config)
    summary = partitioner.partition_entity(
        entity=entity,
        transform_fn=transform_fn,
        resume_run_id=resume_run_id,
        progress_callback=progress_callback,
        dry_run=dry_run,
    )

    if auto_swap and summary.success and not dry_run:
        partitioner.swap_staging_to_final(entity)

    return summary


def print_progress(progress: PartitionProgress) -> None:
    """Default progress printer."""
    print(
        f"Rows: {progress.rows_processed:>12,} | "
        f"Batches: {progress.batches_processed:>6} | "
        f"Partitions: {progress.partitions_created:>4} | "
        f"Mem: {progress.memory_usage_mb:>6.0f}MB | "
        f"Rate: {progress.rows_per_second:>8,.0f}/s | "
        f"Time: {progress.elapsed_seconds:>6.0f}s",
        end="\r",
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Partition monolithic S3 parquet files into year/month/day structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze without writing (dry run)
  python -m polymkt.pipeline.s3_rewrite --entity trades --dry-run

  # Partition trades to staging
  python -m polymkt.pipeline.s3_rewrite --entity trades

  # Partition and auto-swap to final location
  python -m polymkt.pipeline.s3_rewrite --entity trades --auto-swap

  # Resume interrupted partition
  python -m polymkt.pipeline.s3_rewrite --entity trades --resume
        """,
    )

    parser.add_argument(
        "--entity",
        required=True,
        choices=["trades", "order_filled"],
        help="Entity to partition",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze source file without writing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--resume-run-id",
        help="Resume from specific run ID",
    )
    parser.add_argument(
        "--auto-swap",
        action="store_true",
        help="Swap staging to final location on success",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows per batch (default: 10000)",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=6000,
        help="Memory limit in MB (default: 6000)",
    )
    parser.add_argument(
        "--s3-bucket",
        help="S3 bucket (default: from settings)",
    )
    parser.add_argument(
        "--s3-prefix",
        help="S3 prefix (default: from settings)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Handle resume
    resume_run_id = args.resume_run_id
    if args.resume and not resume_run_id:
        config = settings.get_rewrite_config()
        mgr = CheckpointManager(config.checkpoint_dir)
        resume_run_id = mgr.get_latest_run_id(args.entity)
        if not resume_run_id:
            print(f"No checkpoint found for {args.entity}")
            return

    # Apply timestamp fix for order_filled (timestamps stored as seconds, not microseconds)
    transform_fn = None
    if args.entity == "order_filled":
        print("Applying timestamp fix for order_filled (seconds -> microseconds)")
        transform_fn = fix_order_filled_timestamps

    # Run
    summary = run_s3_partition(
        entity=args.entity,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        batch_size=args.batch_size,
        max_memory_mb=args.max_memory_mb,
        resume_run_id=resume_run_id,
        progress_callback=None if args.quiet else print_progress,
        dry_run=args.dry_run,
        auto_swap=args.auto_swap,
        transform_fn=transform_fn,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"Run ID:     {summary.run_id}")
    print(f"Entity:     {summary.entity}")
    print(f"Source:     {summary.source_file}")
    print(f"Rows:       {summary.total_rows:,}")
    print(f"Partitions: {summary.partitions_created}")
    print(f"Files:      {summary.files_created}")
    print(f"Duration:   {summary.duration_seconds:.1f}s")
    print(f"Status:     {'SUCCESS' if summary.success else 'FAILED'}")

    if summary.errors:
        print(f"\nErrors ({len(summary.errors)}):")
        for error in summary.errors[:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
