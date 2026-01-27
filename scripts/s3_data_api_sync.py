"""Stream trades data from Polymarket Data API directly to S3.

Fetches data incrementally from Polymarket's official Data API and writes directly to S3
partitioned parquet files without accumulating on local disk.

This is a parallel data pipeline to the Goldsky-based order_filled sync.
Data is stored at: raw/polymarket/data_api/trades/YYYY/MM/DD/

Memory-safe design:
- Fetches batches from Polymarket API (up to 10,000 records)
- Buffers by day (max 50k rows per partition)
- Flushes to S3 when buffer threshold hit
- Explicit gc.collect() between batches
- No local disk usage

Usage:
    python scripts/s3_data_api_sync.py
    python scripts/s3_data_api_sync.py --dry-run
    python scripts/s3_data_api_sync.py --max-batches 100
"""

import gc
import time
from collections import defaultdict
from datetime import datetime, timezone

import httpx
import psutil
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger()


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_system_memory_percent() -> float:
    """Get system memory usage as a percentage."""
    return psutil.virtual_memory().percent


def check_memory_pressure(max_memory_mb: float = 2000, max_system_percent: float = 80) -> bool:
    """Check if we're under memory pressure. Returns True if safe to continue."""
    current_mb = get_memory_usage_mb()
    system_percent = get_system_memory_percent()

    if current_mb > max_memory_mb:
        logger.warning(
            "memory_limit_exceeded",
            current_mb=round(current_mb, 1),
            max_mb=max_memory_mb,
        )
        return False

    if system_percent > max_system_percent:
        logger.warning(
            "system_memory_pressure",
            system_percent=round(system_percent, 1),
            max_percent=max_system_percent,
        )
        return False

    return True


def log_memory_usage(context: str = "") -> None:
    """Log current memory usage for monitoring."""
    current_mb = get_memory_usage_mb()
    system_percent = get_system_memory_percent()
    logger.info(
        "memory_status",
        context=context,
        process_mb=round(current_mb, 1),
        system_percent=round(system_percent, 1),
    )
    print(f"  Memory: {current_mb:.0f}MB process, {system_percent:.0f}% system")


# Constants
POLYMARKET_API_URL = "https://data-api.polymarket.com/trades"
S3_BUCKET = "polymarket-bcp892"
S3_PREFIX = "raw/polymarket/data_api"  # Raw API responses go here
BATCH_SIZE = 10000  # Max allowed by API
BUFFER_FLUSH_THRESHOLD = 50000  # Rows per partition before flush
MAX_MEMORY_MB = 2000  # Conservative memory limit
# Polymarket rate limit: 200 requests/10s = 20 req/sec
# Use 0.1s delay = 10 req/sec (50% of limit, safe margin)
REQUEST_DELAY_SECONDS = 0.1
RATE_LIMIT_BACKOFF_SECONDS = 30  # Backoff after hitting 429

# Schema for trades (matches Polymarket Data API response)
TRADES_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("transaction_hash", pa.string()),
    ("side", pa.string()),  # BUY or SELL
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("asset", pa.string()),  # Token ID
    ("condition_id", pa.string()),
    ("outcome", pa.string()),
    ("outcome_index", pa.int32()),
    ("proxy_wallet", pa.string()),
    ("title", pa.string()),
    ("slug", pa.string()),
    ("event_slug", pa.string()),
    ("icon", pa.string()),
])


class S3PartitionBuffer:
    """Buffers data by partition and flushes to S3 when threshold reached."""

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        entity: str,
        schema: pa.Schema,
        flush_threshold: int = 50000,
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.entity = entity
        self.schema = schema
        self.flush_threshold = flush_threshold

        self.s3fs = pafs.S3FileSystem(region="us-east-1")
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.buffer_counts: dict[str, int] = defaultdict(int)
        self.files_written = 0
        self.rows_written = 0

    def add_record(self, record: dict) -> None:
        """Add a single record, flush if threshold reached."""
        ts = record.get("timestamp")
        if ts is None:
            return

        # Parse timestamp (API returns unix timestamp as int)
        if isinstance(ts, int):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
            # Try ISO format first, then unix timestamp
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        else:
            return

        partition_key = f"year={dt.year}/month={dt.month:02d}/day={dt.day:02d}"
        self.buffers[partition_key].append(record)
        self.buffer_counts[partition_key] += 1

        if self.buffer_counts[partition_key] >= self.flush_threshold:
            self._flush_partition(partition_key)

    def _flush_partition(self, partition_key: str) -> None:
        """Flush a single partition to S3."""
        if not self.buffers[partition_key]:
            return

        records = self.buffers[partition_key]
        row_count = len(records)

        # Convert to PyArrow table
        columns = {
            "timestamp": [],
            "transaction_hash": [],
            "side": [],
            "price": [],
            "size": [],
            "asset": [],
            "condition_id": [],
            "outcome": [],
            "outcome_index": [],
            "proxy_wallet": [],
            "title": [],
            "slug": [],
            "event_slug": [],
            "icon": [],
        }

        for r in records:
            ts = r.get("timestamp")
            if isinstance(ts, int):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            elif isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            else:
                dt = None

            columns["timestamp"].append(dt)
            columns["transaction_hash"].append(r.get("transactionHash"))
            columns["side"].append(r.get("side"))
            columns["price"].append(float(r.get("price", 0)) if r.get("price") else None)
            columns["size"].append(float(r.get("size", 0)) if r.get("size") else None)
            columns["asset"].append(r.get("asset"))
            columns["condition_id"].append(r.get("conditionId"))
            columns["outcome"].append(r.get("outcome"))
            columns["outcome_index"].append(
                int(r.get("outcomeIndex")) if r.get("outcomeIndex") is not None else None
            )
            columns["proxy_wallet"].append(r.get("proxyWallet"))
            columns["title"].append(r.get("title"))
            columns["slug"].append(r.get("slug"))
            columns["event_slug"].append(r.get("eventSlug"))
            columns["icon"].append(r.get("icon"))

        # Create table
        table = pa.table({
            "timestamp": pa.array(columns["timestamp"], type=pa.timestamp("us", tz="UTC")),
            "transaction_hash": pa.array(columns["transaction_hash"], type=pa.string()),
            "side": pa.array(columns["side"], type=pa.string()),
            "price": pa.array(columns["price"], type=pa.float64()),
            "size": pa.array(columns["size"], type=pa.float64()),
            "asset": pa.array(columns["asset"], type=pa.string()),
            "condition_id": pa.array(columns["condition_id"], type=pa.string()),
            "outcome": pa.array(columns["outcome"], type=pa.string()),
            "outcome_index": pa.array(columns["outcome_index"], type=pa.int32()),
            "proxy_wallet": pa.array(columns["proxy_wallet"], type=pa.string()),
            "title": pa.array(columns["title"], type=pa.string()),
            "slug": pa.array(columns["slug"], type=pa.string()),
            "event_slug": pa.array(columns["event_slug"], type=pa.string()),
            "icon": pa.array(columns["icon"], type=pa.string()),
        })

        # Generate filename
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.entity}_{ts_str}_{row_count}.parquet"
        s3_path = f"{self.s3_bucket}/{self.s3_prefix}/{self.entity}/{partition_key}/{filename}"

        # Write to S3
        pq.write_table(
            table,
            s3_path,
            filesystem=self.s3fs,
            compression="zstd",
        )

        self.files_written += 1
        self.rows_written += row_count

        logger.info(
            "partition_flushed",
            partition=partition_key,
            rows=row_count,
            file=filename,
            total_files=self.files_written,
            total_rows=self.rows_written,
        )
        print(f"  Flushed {row_count:,} rows to {partition_key}")

        # Clear buffer
        self.buffers[partition_key] = []
        self.buffer_counts[partition_key] = 0

        # Help GC
        del table
        del records
        gc.collect()

    def flush_all(self) -> None:
        """Flush all remaining buffers."""
        for partition_key in list(self.buffers.keys()):
            if self.buffers[partition_key]:
                self._flush_partition(partition_key)

    def get_stats(self) -> dict:
        """Get statistics."""
        return {
            "files_written": self.files_written,
            "rows_written": self.rows_written,
            "pending_partitions": len([k for k, v in self.buffer_counts.items() if v > 0]),
            "pending_rows": sum(self.buffer_counts.values()),
        }


def get_latest_s3_timestamp(s3_bucket: str, s3_prefix: str, entity: str) -> datetime | None:
    """Get the latest timestamp from existing S3 partitions."""
    s3fs = pafs.S3FileSystem(region="us-east-1")
    base_path = f"{s3_bucket}/{s3_prefix}/{entity}"

    logger.info("scanning_s3_for_latest_timestamp", path=base_path)

    try:
        # List all files recursively
        selector = pafs.FileSelector(base_path, recursive=True)
        files = s3fs.get_file_info(selector)

        # Find parquet files
        parquet_files = [f for f in files if f.type == pafs.FileType.File and f.path.endswith(".parquet")]

        if not parquet_files:
            logger.warning("no_existing_files", path=base_path)
            return None

        # Sort by path (year/month/day structure means lexicographic = chronological)
        parquet_files.sort(key=lambda f: f.path, reverse=True)

        # Read most recent file's max timestamp
        latest_file = parquet_files[0].path
        logger.info("reading_latest_file", path=latest_file)

        pf = pq.ParquetFile(latest_file, filesystem=s3fs)
        # Read just timestamp column from last row group
        table = pf.read_row_group(pf.metadata.num_row_groups - 1, columns=["timestamp"])

        if table.num_rows == 0:
            return None

        # Get max timestamp
        ts_col = table.column("timestamp")
        max_ts = None
        for i in range(ts_col.length()):
            val = ts_col[i].as_py()
            if val is not None:
                if max_ts is None or val > max_ts:
                    max_ts = val

        if max_ts:
            logger.info(
                "found_latest_timestamp",
                timestamp=max_ts.isoformat(),
            )
            return max_ts

    except Exception as e:
        logger.error("error_reading_s3", error=str(e))

    return None


def fetch_batch(
    client: httpx.Client,
    offset: int,
    limit: int = BATCH_SIZE,
    start_timestamp: int | None = None,
) -> list[dict]:
    """Fetch a batch of trades from Polymarket Data API."""
    params = {
        "limit": limit,
        "offset": offset,
        "takerOnly": "false",  # Get all trades, not just taker
    }

    if start_timestamp is not None:
        params["start"] = start_timestamp

    try:
        response = client.get(POLYMARKET_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "api_request_failed",
            url=POLYMARKET_API_URL,
            status_code=e.response.status_code,
            error=str(e),
            offset=offset,
            limit=limit,
        )
        raise
    except Exception as e:
        logger.error(
            "api_request_failed",
            url=POLYMARKET_API_URL,
            error_type=type(e).__name__,
            error=str(e),
            offset=offset,
            limit=limit,
        )
        raise


def run_sync(
    dry_run: bool = False,
    max_batches: int | None = None,
    start_timestamp: int | None = None,
) -> dict:
    """Run the S3 sync process.

    Args:
        dry_run: If True, only scan existing data without writing
        max_batches: Maximum number of batches to process (for testing)
        start_timestamp: Override start timestamp (for testing)

    Returns:
        Summary statistics
    """
    start_time = time.time()

    print("=" * 60)
    print("Polymarket Data API Sync: Streaming trades to S3")
    print(f"Target: s3://{S3_BUCKET}/{S3_PREFIX}/trades/")
    print("=" * 60)

    # Get starting timestamp from S3
    if start_timestamp is not None:
        last_dt = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
        print(f"Using provided start timestamp: {last_dt.isoformat()}")
    else:
        last_dt = get_latest_s3_timestamp(S3_BUCKET, S3_PREFIX, "trades")
        if last_dt is None:
            print("No existing data found, starting from beginning")
            start_timestamp = None
        else:
            start_timestamp = int(last_dt.timestamp())
            print(f"Resuming from: {last_dt.isoformat()} (unix: {start_timestamp})")

    if dry_run:
        print("\n[DRY RUN] Would fetch from Polymarket API starting at this timestamp")
        return {"dry_run": True, "start_timestamp": start_timestamp}

    # Initialize
    client = httpx.Client()
    buffer = S3PartitionBuffer(
        s3_bucket=S3_BUCKET,
        s3_prefix=S3_PREFIX,
        entity="trades",
        schema=TRADES_SCHEMA,
        flush_threshold=BUFFER_FLUSH_THRESHOLD,
    )

    batch_count = 0
    total_fetched = 0
    offset = 0

    print("\nStarting fetch loop...")
    log_memory_usage("start")

    while True:
        # Check batch limit
        if max_batches is not None and batch_count >= max_batches:
            print(f"\nReached max batches limit: {max_batches}")
            break

        # Fetch batch with exponential backoff on error
        retry_count = 0
        max_fetch_retries = 5
        records = None

        while retry_count < max_fetch_retries:
            try:
                records = fetch_batch(client, offset, BATCH_SIZE, start_timestamp)
                break  # Success
            except httpx.HTTPStatusError as e:
                retry_count += 1

                # Detect rate limit (429)
                is_rate_limit = e.response.status_code == 429

                if is_rate_limit:
                    delay = RATE_LIMIT_BACKOFF_SECONDS
                    print(f"Rate limited (429) - backing off for {delay}s...")
                else:
                    delay = 2 ** retry_count

                logger.error(
                    "fetch_error",
                    url=POLYMARKET_API_URL,
                    status_code=e.response.status_code,
                    is_rate_limit=is_rate_limit,
                    attempt=retry_count,
                    max_retries=max_fetch_retries,
                    delay=delay,
                    offset=offset,
                    batches_completed=batch_count,
                    records_so_far=total_fetched,
                )
                print(f"Fetch error: {e}, retry {retry_count}/{max_fetch_retries} in {delay}s...")

                if retry_count >= max_fetch_retries:
                    print("\nMax retries exceeded - flushing partial data before exit...")
                    try:
                        buffer.flush_all()
                        logger.info(
                            "partial_data_flushed_on_failure",
                            batches=batch_count,
                            written=buffer.get_stats()["rows_written"],
                        )
                    except Exception as flush_err:
                        logger.error("flush_on_failure_failed", error=str(flush_err))
                    raise RuntimeError(f"Max retries ({max_fetch_retries}) exceeded fetching from Polymarket API") from e
                time.sleep(delay)
                continue
            except Exception as e:
                retry_count += 1
                delay = 2 ** retry_count

                logger.error(
                    "fetch_error",
                    url=POLYMARKET_API_URL,
                    error_type=type(e).__name__,
                    error=str(e),
                    attempt=retry_count,
                    max_retries=max_fetch_retries,
                    delay=delay,
                    offset=offset,
                    batches_completed=batch_count,
                    records_so_far=total_fetched,
                )
                print(f"Fetch error: {e}, retry {retry_count}/{max_fetch_retries} in {delay}s...")

                if retry_count >= max_fetch_retries:
                    print("\nMax retries exceeded - flushing partial data before exit...")
                    try:
                        buffer.flush_all()
                    except Exception as flush_err:
                        logger.error("flush_on_failure_failed", error=str(flush_err))
                    raise RuntimeError(f"Max retries ({max_fetch_retries}) exceeded") from e
                time.sleep(delay)
                continue

        if records is None or len(records) == 0:
            print("\nNo more data (empty response)")
            break

        batch_count += 1
        total_fetched += len(records)
        offset += len(records)

        # Add records to buffer
        for record in records:
            buffer.add_record(record)

        # Progress
        stats = buffer.get_stats()

        # Get timestamp range from this batch
        timestamps = [r.get("timestamp") for r in records if r.get("timestamp")]
        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc)
            max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc)
            ts_info = f"{min_dt.strftime('%Y-%m-%d %H:%M')} to {max_dt.strftime('%Y-%m-%d %H:%M')}"
        else:
            ts_info = ""

        print(
            f"Batch {batch_count}: fetched {len(records):,}, "
            f"total {total_fetched:,}, "
            f"written {stats['rows_written']:,}, "
            f"pending {stats['pending_rows']:,}, "
            f"ts: {ts_info}"
        )

        # If we got fewer records than requested, we've reached the end
        if len(records) < BATCH_SIZE:
            print("\nReached end of data (partial batch)")
            break

        # Periodic GC and memory check
        if batch_count % 10 == 0:
            gc.collect()
            log_memory_usage(f"batch_{batch_count}")

            # Check memory pressure - flush and exit gracefully if needed
            if not check_memory_pressure(MAX_MEMORY_MB):
                print("\nMemory pressure detected - flushing and exiting gracefully")
                logger.warning("memory_pressure_exit", batch=batch_count, fetched=total_fetched)
                break

        # Rate limit: delay between requests
        time.sleep(REQUEST_DELAY_SECONDS)

    # Flush remaining - ensure partial success writes data
    print("\nFlushing remaining buffers...")
    try:
        buffer.flush_all()
    except Exception as flush_error:
        logger.error(
            "flush_failed_partial_success",
            error=str(flush_error),
            batches_processed=batch_count,
            records_fetched=total_fetched,
        )
        print(f"Warning: Failed to flush some buffers: {flush_error}")

    log_memory_usage("final")
    client.close()

    elapsed = time.time() - start_time
    final_stats = buffer.get_stats()

    print("\n" + "=" * 60)
    print("Sync Complete!")
    print("=" * 60)
    print(f"Batches processed: {batch_count}")
    print(f"Records fetched: {total_fetched:,}")
    print(f"Rows written to S3: {final_stats['rows_written']:,}")
    print(f"Files created: {final_stats['files_written']}")
    print(f"Duration: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"Rate: {total_fetched / elapsed:.0f} records/s")

    return {
        "batches": batch_count,
        "fetched": total_fetched,
        "written": final_stats["rows_written"],
        "files": final_stats["files_written"],
        "duration_seconds": elapsed,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Stream trades from Polymarket Data API to S3")
    parser.add_argument("--dry-run", action="store_true", help="Scan S3 without fetching")
    parser.add_argument("--max-batches", type=int, help="Max batches to process (for testing)")
    parser.add_argument("--start-ts", type=int, help="Override start timestamp (unix seconds)")
    args = parser.parse_args()

    try:
        result = run_sync(
            dry_run=args.dry_run,
            max_batches=args.max_batches,
            start_timestamp=args.start_ts,
        )
    except MemoryError as e:
        logger.error("memory_exhausted", error=str(e), memory_mb=get_memory_usage_mb())
        print(f"\nERROR: Memory exhausted - {e}")
        print("Partial progress may have been written to S3.")
        sys.exit(1)
    except Exception as e:
        logger.error("sync_failed", error=str(e))
        print(f"\nERROR: Sync failed - {e}")
        sys.exit(1)
