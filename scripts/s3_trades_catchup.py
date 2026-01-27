"""Process order_filled data from S3 into trades and write back to S3.

Reads order_filled parquets from S3 in batches, joins with markets data to compute
trade attributes (price, direction, market_id), and writes partitioned
trades parquets back to S3.

Resumable: Reads latest trades timestamp from S3 and continues from there.

Usage:
    python scripts/s3_trades_catchup.py
    python scripts/s3_trades_catchup.py --dry-run
    python scripts/s3_trades_catchup.py --batch-size 50
"""

import gc
import time
from datetime import datetime, timezone

import polars as pl
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger()

# Constants
S3_BUCKET = "polymarket-bcp892"
S3_PREFIX = "raw/polymarket"
FILES_PER_BATCH = 100  # Process this many files at a time


def get_markets_df() -> pl.DataFrame:
    """Load markets data for joining."""
    logger.info("loading_markets_data")

    # Load from local cache with proper schema
    markets_df = pl.read_csv(
        "data/markets.csv",
        schema_overrides={"token1": pl.Utf8, "token2": pl.Utf8},
    )
    logger.info("loaded_markets", rows=len(markets_df))
    return markets_df


def process_trades(df: pl.DataFrame, markets_df: pl.DataFrame) -> pl.DataFrame:
    """Transform order_filled data into trades."""

    # Make markets long: (market_id, side, asset_id)
    markets_long = (
        markets_df
        .select(["id", "token1", "token2"])
        .rename({"id": "market_id"})
        .melt(id_vars="market_id", value_vars=["token1", "token2"],
              variable_name="side", value_name="asset_id")
    )

    # Identify the non-USDC asset for each trade
    df = df.with_columns(
        pl.when(pl.col("maker_asset_id") != "0")
        .then(pl.col("maker_asset_id"))
        .otherwise(pl.col("taker_asset_id"))
        .alias("nonusdc_asset_id")
    )

    # Join to get market_id and side
    df = df.join(
        markets_long,
        left_on="nonusdc_asset_id",
        right_on="asset_id",
        how="left",
    )

    # Compute directions
    df = df.with_columns([
        pl.when(pl.col("taker_asset_id") == "0")
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("SELL"))
        .alias("taker_direction"),

        pl.when(pl.col("taker_asset_id") == "0")
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("BUY"))
        .alias("maker_direction"),
    ])

    # Compute nonusdc_side
    df = df.with_columns([
        pl.col("side").alias("nonusdc_side"),
    ])

    # Compute amounts and price
    df = df.with_columns([
        pl.when(pl.col("taker_asset_id") == "0")
        .then(pl.col("taker_amount_filled"))
        .otherwise(pl.col("maker_amount_filled"))
        .alias("usd_amount"),

        pl.when(pl.col("taker_asset_id") != "0")
        .then(pl.col("taker_amount_filled"))
        .otherwise(pl.col("maker_amount_filled"))
        .alias("token_amount"),
    ])

    df = df.with_columns([
        (pl.col("usd_amount") / pl.col("token_amount"))
        .cast(pl.Float64)
        .alias("price")
    ])

    # Select final columns
    df = df.select([
        "timestamp",
        "market_id",
        "maker",
        "taker",
        "nonusdc_side",
        "maker_direction",
        "taker_direction",
        "price",
        "usd_amount",
        "token_amount",
        "transaction_hash",
    ])

    # Filter out rows with null market_id (unmatched)
    df = df.filter(pl.col("market_id").is_not_null())

    return df


def get_latest_trades_timestamp() -> datetime | None:
    """Get the latest timestamp from existing trades in S3."""
    s3fs = pafs.S3FileSystem(region="us-east-1")
    base_path = f"{S3_BUCKET}/{S3_PREFIX}/trades"

    logger.info("scanning_trades_for_latest_timestamp", path=base_path)

    try:
        selector = pafs.FileSelector(base_path, recursive=True)
        files = s3fs.get_file_info(selector)
        parquet_files = [f for f in files if f.type == pafs.FileType.File and f.path.endswith(".parquet")]

        if not parquet_files:
            logger.warning("no_existing_trades")
            return None

        parquet_files.sort(key=lambda f: f.path, reverse=True)
        latest_file = parquet_files[0].path

        logger.info("reading_latest_trades_file", path=latest_file)

        pf = pq.ParquetFile(latest_file, filesystem=s3fs)
        table = pf.read_row_group(pf.metadata.num_row_groups - 1, columns=["timestamp"])

        if table.num_rows == 0:
            return None

        ts_col = table.column("timestamp")
        max_ts = None
        for i in range(ts_col.length()):
            val = ts_col[i].as_py()
            if val is not None:
                if max_ts is None or val > max_ts:
                    max_ts = val

        if max_ts:
            logger.info("found_latest_trades_timestamp", timestamp=max_ts.isoformat())
            return max_ts

    except Exception as e:
        logger.error("error_reading_trades", error=str(e))

    return None


def get_order_filled_files(start_ts: datetime | None) -> list[str]:
    """Get list of order_filled files from S3, optionally filtered by partition date."""
    s3fs = pafs.S3FileSystem(region="us-east-1")
    base_path = f"{S3_BUCKET}/{S3_PREFIX}/order_filled"

    logger.info("listing_order_filled_files", path=base_path)

    selector = pafs.FileSelector(base_path, recursive=True)
    files = s3fs.get_file_info(selector)
    parquet_files = [f.path for f in files if f.type == pafs.FileType.File and f.path.endswith(".parquet")]
    parquet_files.sort()

    logger.info("found_order_filled_files", count=len(parquet_files))

    # Filter by partition path if we have a start timestamp
    if start_ts:
        start_partition = f"year={start_ts.year}/month={start_ts.month:02d}/day={start_ts.day:02d}"
        # Keep files where partition >= start_partition lexicographically
        filtered = [f for f in parquet_files if extract_partition(f) >= start_partition]
        parquet_files = filtered
        logger.info("filtered_to_files", count=len(parquet_files), start_partition=start_partition)

    return parquet_files


def extract_partition(path: str) -> str:
    """Extract year=X/month=Y/day=Z from path."""
    import re
    match = re.search(r'year=\d+/month=\d+/day=\d+', path)
    return match.group(0) if match else ""


def read_files_batch(files: list[str], start_ts: datetime | None) -> pl.DataFrame:
    """Read a batch of parquet files and filter by timestamp."""
    s3fs = pafs.S3FileSystem(region="us-east-1")

    all_dfs = []
    for path in files:
        try:
            table = pq.read_table(path, filesystem=s3fs)
            df = pl.from_arrow(table)

            if start_ts is not None:
                df = df.filter(pl.col("timestamp") > start_ts)

            if len(df) > 0:
                all_dfs.append(df)

        except Exception as e:
            logger.warning("error_reading_file", path=path, error=str(e))
            continue

    if not all_dfs:
        return pl.DataFrame()

    combined = pl.concat(all_dfs)
    return combined.sort("timestamp")


def write_trades_to_s3(df: pl.DataFrame) -> dict:
    """Write trades dataframe to S3 partitioned by day."""
    s3fs = pafs.S3FileSystem(region="us-east-1")

    files_written = 0
    rows_written = 0

    if len(df) == 0:
        return {"files_written": 0, "rows_written": 0}

    # Group by day
    df = df.with_columns([
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.day().alias("day"),
    ])

    for (year, month, day), group in df.group_by(["year", "month", "day"]):
        partition_key = f"year={year}/month={month:02d}/day={day:02d}"

        # Convert to arrow
        group_clean = group.drop(["year", "month", "day"])
        table = group_clean.to_arrow()

        # Generate filename
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        row_count = len(group)
        filename = f"trades_{ts_str}_{row_count}.parquet"
        s3_path = f"{S3_BUCKET}/{S3_PREFIX}/trades/{partition_key}/{filename}"

        # Write
        pq.write_table(table, s3_path, filesystem=s3fs, compression="zstd")

        files_written += 1
        rows_written += row_count

        logger.info("partition_written", partition=partition_key, rows=row_count)

    return {"files_written": files_written, "rows_written": rows_written}


def run_catchup(dry_run: bool = False, batch_size: int = FILES_PER_BATCH) -> dict:
    """Run the trades catchup process in batches."""
    start_time = time.time()

    print("=" * 60)
    print("Trades Catchup: Processing order_filled -> trades (batched)")
    print("=" * 60)

    # Get latest trades timestamp
    latest_trades_ts = get_latest_trades_timestamp()
    if latest_trades_ts:
        print(f"Resuming from: {latest_trades_ts.isoformat()}")
    else:
        print("No existing trades - processing all order_filled data")

    # Load markets once
    print("\nLoading markets data...")
    markets_df = get_markets_df()
    print(f"Loaded {len(markets_df):,} markets")

    # Get list of files to process
    print("\nListing order_filled files...")
    all_files = get_order_filled_files(latest_trades_ts)
    print(f"Found {len(all_files):,} files to process")

    if dry_run:
        print(f"\n[DRY RUN] Would process {len(all_files)} files in batches of {batch_size}")
        return {"dry_run": True, "files_to_process": len(all_files)}

    if not all_files:
        print("No files to process - already up to date!")
        return {"rows_processed": 0, "trades_written": 0}

    # Process in batches
    total_order_filled = 0
    total_trades = 0
    total_files_written = 0
    current_ts = latest_trades_ts

    num_batches = (len(all_files) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(all_files))
        batch_files = all_files[batch_start:batch_end]

        print(f"\n--- Batch {batch_num + 1}/{num_batches} ({len(batch_files)} files) ---")

        # Read batch
        order_filled_df = read_files_batch(batch_files, current_ts)

        if len(order_filled_df) == 0:
            print(f"  No new rows in this batch")
            continue

        print(f"  Read {len(order_filled_df):,} order_filled rows")
        total_order_filled += len(order_filled_df)

        # Process into trades
        trades_df = process_trades(order_filled_df, markets_df)
        print(f"  Generated {len(trades_df):,} trades")

        if len(trades_df) == 0:
            continue

        # Update current timestamp for next batch filtering
        max_ts = trades_df.select(pl.col("timestamp").max()).item()
        if max_ts:
            current_ts = max_ts

        # Write to S3
        stats = write_trades_to_s3(trades_df)
        total_trades += stats["rows_written"]
        total_files_written += stats["files_written"]

        print(f"  Wrote {stats['rows_written']:,} trades to {stats['files_written']} files")
        print(f"  Progress: {batch_end}/{len(all_files)} files ({100*batch_end/len(all_files):.1f}%)")

        # Clean up memory
        del order_filled_df
        del trades_df
        gc.collect()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Trades Catchup Complete!")
    print("=" * 60)
    print(f"Order filled rows processed: {total_order_filled:,}")
    print(f"Trades written: {total_trades:,}")
    print(f"Files created: {total_files_written}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Rate: {total_order_filled / elapsed:.0f} rows/s" if elapsed > 0 else "")

    return {
        "order_filled_processed": total_order_filled,
        "trades_written": total_trades,
        "files": total_files_written,
        "duration_seconds": elapsed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process order_filled into trades")
    parser.add_argument("--dry-run", action="store_true", help="Scan without processing")
    parser.add_argument("--batch-size", type=int, default=FILES_PER_BATCH, help="Files per batch")
    args = parser.parse_args()

    run_catchup(dry_run=args.dry_run, batch_size=args.batch_size)
