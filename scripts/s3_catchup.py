"""Stream order_filled data from Goldsky API directly to S3.

Fetches data incrementally from Goldsky GraphQL API and writes directly to S3
partitioned parquet files without accumulating on local disk.

Memory-safe design:
- Fetches small batches from Goldsky (1000 records)
- Buffers by day (max 50k rows per partition)
- Flushes to S3 when buffer threshold hit
- Explicit gc.collect() between batches
- No local disk usage

Usage:
    python scripts/s3_catchup.py
    python scripts/s3_catchup.py --dry-run
    python scripts/s3_catchup.py --max-batches 100
"""

import gc
import time
from collections import defaultdict
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import structlog
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

logger = structlog.get_logger()

# Constants
GOLDSKY_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
S3_BUCKET = "polymarket-bcp892"
S3_PREFIX = "raw/polymarket"
BATCH_SIZE = 1000  # Records per Goldsky query
BUFFER_FLUSH_THRESHOLD = 50000  # Rows per partition before flush
MAX_MEMORY_MB = 2000  # Conservative memory limit

# Schema for order_filled
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

        # Parse timestamp
        if isinstance(ts, int):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
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
            "maker": [],
            "maker_asset_id": [],
            "maker_amount_filled": [],
            "taker": [],
            "taker_asset_id": [],
            "taker_amount_filled": [],
            "transaction_hash": [],
        }

        for r in records:
            ts = r.get("timestamp")
            if isinstance(ts, (int, str)):
                ts_int = int(ts)
                # Convert seconds to microseconds for timestamp[us]
                dt = datetime.fromtimestamp(ts_int, tz=timezone.utc)
            else:
                dt = None

            columns["timestamp"].append(dt)
            columns["maker"].append(r.get("maker"))
            columns["maker_asset_id"].append(r.get("makerAssetId"))
            columns["maker_amount_filled"].append(
                float(r.get("makerAmountFilled", 0)) / 1e6 if r.get("makerAmountFilled") else None
            )
            columns["taker"].append(r.get("taker"))
            columns["taker_asset_id"].append(r.get("takerAssetId"))
            columns["taker_amount_filled"].append(
                float(r.get("takerAmountFilled", 0)) / 1e6 if r.get("takerAmountFilled") else None
            )
            columns["transaction_hash"].append(r.get("transactionHash"))

        # Create table
        table = pa.table({
            "timestamp": pa.array(columns["timestamp"], type=pa.timestamp("us", tz="UTC")),
            "maker": pa.array(columns["maker"], type=pa.string()),
            "maker_asset_id": pa.array(columns["maker_asset_id"], type=pa.string()),
            "maker_amount_filled": pa.array(columns["maker_amount_filled"], type=pa.float64()),
            "taker": pa.array(columns["taker"], type=pa.string()),
            "taker_asset_id": pa.array(columns["taker_asset_id"], type=pa.string()),
            "taker_amount_filled": pa.array(columns["taker_amount_filled"], type=pa.float64()),
            "transaction_hash": pa.array(columns["transaction_hash"], type=pa.string()),
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


def get_latest_s3_timestamp(s3_bucket: str, s3_prefix: str, entity: str) -> int:
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
            return 0

        # Sort by path (year/month/day structure means lexicographic = chronological)
        parquet_files.sort(key=lambda f: f.path, reverse=True)

        # Read most recent file's max timestamp
        latest_file = parquet_files[0].path
        logger.info("reading_latest_file", path=latest_file)

        pf = pq.ParquetFile(latest_file, filesystem=s3fs)
        # Read just timestamp column from first row group
        table = pf.read_row_group(pf.metadata.num_row_groups - 1, columns=["timestamp"])

        if table.num_rows == 0:
            return 0

        # Get max timestamp
        ts_col = table.column("timestamp")
        max_ts = None
        for i in range(ts_col.length()):
            val = ts_col[i].as_py()
            if val is not None:
                if max_ts is None or val > max_ts:
                    max_ts = val

        if max_ts:
            # Convert to unix timestamp (seconds)
            unix_ts = int(max_ts.timestamp())
            logger.info(
                "found_latest_timestamp",
                timestamp=max_ts.isoformat(),
                unix_ts=unix_ts,
            )
            return unix_ts

    except Exception as e:
        logger.error("error_reading_s3", error=str(e))

    return 0


def create_goldsky_client() -> Client:
    """Create GraphQL client for Goldsky."""
    transport = RequestsHTTPTransport(url=GOLDSKY_URL, verify=True, retries=3)
    return Client(transport=transport)


def fetch_batch(client: Client, last_timestamp: int, batch_size: int = 1000) -> list[dict]:
    """Fetch a batch of orderFilledEvents from Goldsky."""
    query_str = f"""query {{
        orderFilledEvents(
            orderBy: timestamp
            first: {batch_size}
            where: {{timestamp_gt: "{last_timestamp}"}}
        ) {{
            timestamp
            maker
            makerAssetId
            makerAmountFilled
            taker
            takerAssetId
            takerAmountFilled
            transactionHash
        }}
    }}"""

    query = gql(query_str)
    result = client.execute(query)
    return result.get("orderFilledEvents", [])


def run_catchup(
    dry_run: bool = False,
    max_batches: int | None = None,
    start_timestamp: int | None = None,
) -> dict:
    """Run the S3 catchup process.

    Args:
        dry_run: If True, only scan existing data without writing
        max_batches: Maximum number of batches to process (for testing)
        start_timestamp: Override start timestamp (for testing)

    Returns:
        Summary statistics
    """
    start_time = time.time()

    print("=" * 60)
    print("S3 Catchup: Streaming order_filled from Goldsky to S3")
    print("=" * 60)

    # Get starting timestamp from S3
    if start_timestamp is not None:
        last_timestamp = start_timestamp
        print(f"Using provided start timestamp: {last_timestamp}")
    else:
        last_timestamp = get_latest_s3_timestamp(S3_BUCKET, S3_PREFIX, "order_filled")
        if last_timestamp == 0:
            print("No existing data found, starting from beginning")
        else:
            dt = datetime.fromtimestamp(last_timestamp, tz=timezone.utc)
            print(f"Resuming from: {dt.isoformat()} (unix: {last_timestamp})")

    if dry_run:
        print("\n[DRY RUN] Would fetch from Goldsky starting at this timestamp")
        return {"dry_run": True, "start_timestamp": last_timestamp}

    # Initialize
    client = create_goldsky_client()
    buffer = S3PartitionBuffer(
        s3_bucket=S3_BUCKET,
        s3_prefix=S3_PREFIX,
        entity="order_filled",
        schema=ORDER_FILLED_SCHEMA,
        flush_threshold=BUFFER_FLUSH_THRESHOLD,
    )

    batch_count = 0
    total_fetched = 0
    consecutive_empty = 0

    print("\nStarting fetch loop...")

    while True:
        # Check batch limit
        if max_batches is not None and batch_count >= max_batches:
            print(f"\nReached max batches limit: {max_batches}")
            break

        # Fetch batch
        try:
            records = fetch_batch(client, last_timestamp, BATCH_SIZE)
        except Exception as e:
            logger.error("fetch_error", error=str(e))
            print(f"Fetch error: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not records:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("\nNo more data (3 consecutive empty batches)")
                break
            print("Empty batch, waiting 2s...")
            time.sleep(2)
            continue

        consecutive_empty = 0
        batch_count += 1
        total_fetched += len(records)

        # Update last timestamp
        max_ts = max(int(r["timestamp"]) for r in records)
        last_timestamp = max_ts

        # Add records to buffer
        for record in records:
            buffer.add_record(record)

        # Progress
        dt = datetime.fromtimestamp(last_timestamp, tz=timezone.utc)
        stats = buffer.get_stats()
        print(
            f"Batch {batch_count}: fetched {len(records)}, "
            f"total {total_fetched:,}, "
            f"written {stats['rows_written']:,}, "
            f"pending {stats['pending_rows']:,}, "
            f"ts: {dt.strftime('%Y-%m-%d %H:%M')}"
        )

        # Periodic GC
        if batch_count % 50 == 0:
            gc.collect()

        # Small delay to be nice to API
        if len(records) == BATCH_SIZE:
            time.sleep(0.1)

    # Flush remaining
    print("\nFlushing remaining buffers...")
    buffer.flush_all()

    elapsed = time.time() - start_time
    final_stats = buffer.get_stats()

    print("\n" + "=" * 60)
    print("Catchup Complete!")
    print("=" * 60)
    print(f"Batches processed: {batch_count}")
    print(f"Records fetched: {total_fetched:,}")
    print(f"Rows written to S3: {final_stats['rows_written']:,}")
    print(f"Files created: {final_stats['files_written']}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Rate: {total_fetched / elapsed:.0f} records/s" if elapsed > 0 else "")

    return {
        "batches": batch_count,
        "fetched": total_fetched,
        "written": final_stats["rows_written"],
        "files": final_stats["files_written"],
        "duration_seconds": elapsed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream order_filled from Goldsky to S3")
    parser.add_argument("--dry-run", action="store_true", help="Scan S3 without fetching")
    parser.add_argument("--max-batches", type=int, help="Max batches to process (for testing)")
    parser.add_argument("--start-ts", type=int, help="Override start timestamp")
    args = parser.parse_args()

    run_catchup(
        dry_run=args.dry_run,
        max_batches=args.max_batches,
        start_timestamp=args.start_ts,
    )
