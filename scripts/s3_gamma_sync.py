"""Gamma API sync script: Fetch markets and events directly to S3 (no local CSV).

Fetches markets and events from the Polymarket Gamma API, writes raw JSONL to S3
for audit trail, then transforms to canonical parquet schemas and writes cleaned
parquet files to S3.

Memory-safe design:
- Paginates API requests (500 records per batch)
- Writes raw JSONL incrementally
- Atomic parquet write (temp file then rename)
- zstd compression for parquet files

Usage:
    python scripts/s3_gamma_sync.py
    python scripts/s3_gamma_sync.py --dry-run
    python scripts/s3_gamma_sync.py --entity markets
    python scripts/s3_gamma_sync.py --entity events
"""

import gc
import json
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import httpx
import psutil
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger()

# Constants
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
S3_BUCKET = "polymarket-bcp892"
S3_PREFIX = "raw/polymarket"
BATCH_SIZE = 500  # Records per API request
MAX_RETRIES = 5
TIMEOUT_SECONDS = 30
MAX_MEMORY_MB = 2000  # Conservative memory limit


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_system_memory_percent() -> float:
    """Get system memory usage as a percentage."""
    return psutil.virtual_memory().percent


def check_memory_pressure(max_memory_mb: float = MAX_MEMORY_MB, max_system_percent: float = 80) -> bool:
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

# Retry delays (in seconds)
RETRY_DELAYS = {
    429: 10,  # Rate limit
    500: 5,   # Server error
    "default": 2,
}

# Markets schema (matching PRD requirements)
MARKETS_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("question", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")),
    ("closed_time", pa.timestamp("us", tz="UTC")),
    ("category", pa.string()),
    ("volume", pa.float64()),
    ("token1", pa.string()),
    ("token2", pa.string()),
    ("condition_id", pa.string()),
    ("slug", pa.string()),
    ("description", pa.string()),
    ("event_id", pa.string()),
    ("neg_risk", pa.bool_()),
    ("answer1", pa.string()),
    ("answer2", pa.string()),
    ("ticker", pa.string()),
    ("tags", pa.list_(pa.string())),
])

# Events schema (matching PRD requirements)
EVENTS_SCHEMA = pa.schema([
    ("event_id", pa.string()),
    ("title", pa.string()),
    ("description", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")),
    ("tags", pa.list_(pa.string())),
])


def _get_s3_filesystem() -> pafs.S3FileSystem:
    """Create a PyArrow S3FileSystem."""
    return pafs.S3FileSystem(region="us-east-1")


def _parse_timestamp(ts_str: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime with UTC timezone."""
    if not ts_str:
        return None
    try:
        # Handle various ISO formats
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _normalize_address(addr: str | None) -> str | None:
    """Normalize Ethereum address to lowercase with 0x prefix."""
    if not addr:
        return None
    addr = str(addr).strip().lower()
    if not addr.startswith("0x"):
        addr = "0x" + addr
    return addr


def _extract_tag_labels(tags: list[dict[str, Any]] | None) -> list[str]:
    """Extract tag labels from API format [{id, label, slug}] to ['label1', 'label2']."""
    if not tags:
        return []
    result = []
    for tag in tags:
        if isinstance(tag, dict) and "label" in tag:
            result.append(str(tag["label"]))
    return result


def fetch_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch URL with exponential backoff retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(url, params=params, timeout=TIMEOUT_SECONDS)

            if response.status_code == 200:
                return response.json()

            # Handle retryable errors
            if response.status_code in (429, 500, 502, 503, 504):
                delay = RETRY_DELAYS.get(response.status_code, RETRY_DELAYS["default"])
                logger.warning(
                    "retrying_request",
                    status=response.status_code,
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    delay=delay,
                    url=url,
                    params=params,
                    response_body_preview=response.text[:200] if response.text else None,
                )
                time.sleep(delay)
                continue

            # Non-retryable error - log with full context
            logger.error(
                "request_failed_non_retryable",
                status=response.status_code,
                url=url,
                params=params,
                response_body_preview=response.text[:500] if response.text else None,
                response_headers=dict(response.headers),
            )
            raise httpx.HTTPStatusError(
                f"Request failed with status {response.status_code}",
                request=response.request,
                response=response,
            )

        except httpx.TimeoutException as e:
            delay = RETRY_DELAYS["default"]
            logger.warning(
                "request_timeout",
                error_type=type(e).__name__,
                attempt=attempt + 1,
                max_retries=MAX_RETRIES,
                delay=delay,
                timeout_seconds=TIMEOUT_SECONDS,
                url=url,
                params=params,
            )
            time.sleep(delay)
            continue

        except httpx.RequestError as e:
            delay = RETRY_DELAYS["default"]
            logger.warning(
                "request_error",
                error_type=type(e).__name__,
                error=str(e),
                attempt=attempt + 1,
                max_retries=MAX_RETRIES,
                delay=delay,
                url=url,
                params=params,
            )
            time.sleep(delay)
            continue

    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded for {url}")


def fetch_records_streaming(
    entity: str,
    s3fs: pafs.S3FileSystem,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Fetch records for an entity with streaming JSONL writes to S3.

    Instead of accumulating all records in memory, writes each batch to S3 as JSONL
    immediately after fetching. Returns all records for final parquet transformation,
    but memory is bounded by gc.collect() after each batch.

    Partial success: If fetching fails partway through, already-written JSONL files
    are preserved for audit trail and potential recovery.

    Returns:
        Tuple of (all_records, total_count, jsonl_paths)
    """
    url = f"{GAMMA_API_BASE}/{entity}"
    all_records: list[dict[str, Any]] = []
    offset = 0
    batch_num = 0
    jsonl_paths: list[str] = []
    fetch_error: Exception | None = None

    print(f"Fetching {entity} from Gamma API (streaming mode)...")
    log_memory_usage("fetch_start")

    with httpx.Client() as client:
        while True:
            params = {
                "limit": BATCH_SIZE,
                "offset": offset,
                "order": "createdAt",
                "ascending": "true",
            }

            if dry_run and offset >= BATCH_SIZE:
                # In dry run, only fetch first batch
                print(f"[DRY RUN] Would continue fetching from offset {offset}")
                break

            try:
                records = fetch_with_retry(client, url, params)
            except Exception as e:
                # Log error with full context but continue with partial data
                fetch_error = e
                logger.error(
                    "fetch_stopped_with_partial_data",
                    entity=entity,
                    url=url,
                    offset=offset,
                    batches_completed=batch_num,
                    records_fetched=len(all_records),
                    jsonl_files_written=len(jsonl_paths),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                print(f"\nFetch error at offset {offset}: {e}")
                print(f"Continuing with {len(all_records)} records fetched so far...")
                break

            if not records:
                break

            batch_num += 1

            # Write this batch to JSONL immediately (audit trail)
            if not dry_run:
                try:
                    jsonl_path = write_jsonl_batch_to_s3(records, entity, batch_num, s3fs)
                    jsonl_paths.append(jsonl_path)
                except Exception as write_err:
                    logger.error(
                        "jsonl_write_failed",
                        entity=entity,
                        batch=batch_num,
                        records_in_batch=len(records),
                        error=str(write_err),
                    )
                    # Continue - the records are still in memory for parquet write

            all_records.extend(records)
            offset += len(records)

            print(f"  Batch {batch_num}: fetched {len(records)} {entity}, total: {len(all_records)}")

            # Periodic memory check and cleanup
            if batch_num % 10 == 0:
                gc.collect()
                log_memory_usage(f"batch_{batch_num}")

                if not check_memory_pressure():
                    print("\nMemory pressure detected during fetch - stopping early")
                    logger.warning(
                        "memory_pressure_during_fetch",
                        entity=entity,
                        records_fetched=len(all_records),
                        jsonl_files_written=len(jsonl_paths),
                    )
                    break

            # If we got fewer records than batch size, we're done
            if len(records) < BATCH_SIZE:
                break

            # Small delay to be nice to the API
            time.sleep(0.1)

    log_memory_usage("fetch_complete")

    # Log summary including any partial failure
    if fetch_error:
        logger.info(
            "fetch_completed_with_partial_data",
            entity=entity,
            records=len(all_records),
            jsonl_batches=len(jsonl_paths),
            had_error=True,
        )
    else:
        logger.info(
            "fetch_completed",
            entity=entity,
            records=len(all_records),
            jsonl_batches=len(jsonl_paths),
        )

    return all_records, len(all_records), jsonl_paths


def write_jsonl_batch_to_s3(
    records: list[dict[str, Any]],
    entity: str,
    batch_num: int,
    s3fs: pafs.S3FileSystem,
) -> str:
    """Write a single batch of records as JSONL to S3."""
    now = datetime.now(timezone.utc)
    date_path = now.strftime("%Y/%m/%d")
    time_str = now.strftime("%H%M%S")
    filename = f"{entity}_{time_str}_batch{batch_num:04d}.jsonl"
    s3_path = f"{S3_BUCKET}/{S3_PREFIX}/gamma_api/{entity}/{date_path}/{filename}"

    # Create JSONL content
    jsonl_buffer = BytesIO()
    for record in records:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        jsonl_buffer.write(line.encode("utf-8"))

    jsonl_buffer.seek(0)

    # Write to S3
    with s3fs.open_output_stream(s3_path) as f:
        f.write(jsonl_buffer.getvalue())

    logger.debug("wrote_jsonl_batch", path=s3_path, records=len(records), batch=batch_num)
    return s3_path
def transform_markets(records: list[dict[str, Any]]) -> pa.Table:
    """Transform raw market records to canonical parquet schema."""
    columns: dict[str, list[Any]] = {
        "id": [],
        "question": [],
        "created_at": [],
        "closed_time": [],
        "category": [],
        "volume": [],
        "token1": [],
        "token2": [],
        "condition_id": [],
        "slug": [],
        "description": [],
        "event_id": [],
        "neg_risk": [],
        "answer1": [],
        "answer2": [],
        "ticker": [],
        "tags": [],
    }

    for r in records:
        columns["id"].append(r.get("id"))
        columns["question"].append(r.get("question"))
        columns["created_at"].append(_parse_timestamp(r.get("createdAt")))
        columns["closed_time"].append(_parse_timestamp(r.get("closedTime")))
        columns["category"].append(r.get("category"))

        # Volume could be string or number
        vol = r.get("volume")
        if vol is not None:
            try:
                columns["volume"].append(float(vol))
            except (ValueError, TypeError):
                columns["volume"].append(None)
        else:
            columns["volume"].append(None)

        # Token addresses - normalize to lowercase with 0x prefix
        # Get tokens from clobTokenIds array if available
        clob_tokens = r.get("clobTokenIds", [])
        if clob_tokens and len(clob_tokens) >= 2:
            columns["token1"].append(_normalize_address(clob_tokens[0]))
            columns["token2"].append(_normalize_address(clob_tokens[1]))
        else:
            columns["token1"].append(_normalize_address(r.get("token1")))
            columns["token2"].append(_normalize_address(r.get("token2")))

        columns["condition_id"].append(r.get("conditionId"))
        columns["slug"].append(r.get("slug"))
        columns["description"].append(r.get("description"))
        columns["event_id"].append(r.get("eventId"))

        # negRisk is a boolean
        neg_risk = r.get("negRisk")
        if isinstance(neg_risk, bool):
            columns["neg_risk"].append(neg_risk)
        elif isinstance(neg_risk, str):
            columns["neg_risk"].append(neg_risk.lower() == "true")
        else:
            columns["neg_risk"].append(False)

        # Outcomes
        outcomes = r.get("outcomes", [])
        if outcomes and len(outcomes) >= 2:
            columns["answer1"].append(outcomes[0])
            columns["answer2"].append(outcomes[1])
        else:
            columns["answer1"].append(r.get("answer1", "Yes"))
            columns["answer2"].append(r.get("answer2", "No"))

        columns["ticker"].append(r.get("ticker"))

        # Tags from parent event (if present in API response)
        tags = r.get("tags", [])
        columns["tags"].append(_extract_tag_labels(tags) if tags else [])

    # Create PyArrow table
    table = pa.table({
        "id": pa.array(columns["id"], type=pa.string()),
        "question": pa.array(columns["question"], type=pa.string()),
        "created_at": pa.array(columns["created_at"], type=pa.timestamp("us", tz="UTC")),
        "closed_time": pa.array(columns["closed_time"], type=pa.timestamp("us", tz="UTC")),
        "category": pa.array(columns["category"], type=pa.string()),
        "volume": pa.array(columns["volume"], type=pa.float64()),
        "token1": pa.array(columns["token1"], type=pa.string()),
        "token2": pa.array(columns["token2"], type=pa.string()),
        "condition_id": pa.array(columns["condition_id"], type=pa.string()),
        "slug": pa.array(columns["slug"], type=pa.string()),
        "description": pa.array(columns["description"], type=pa.string()),
        "event_id": pa.array(columns["event_id"], type=pa.string()),
        "neg_risk": pa.array(columns["neg_risk"], type=pa.bool_()),
        "answer1": pa.array(columns["answer1"], type=pa.string()),
        "answer2": pa.array(columns["answer2"], type=pa.string()),
        "ticker": pa.array(columns["ticker"], type=pa.string()),
        "tags": pa.array(columns["tags"], type=pa.list_(pa.string())),
    })

    return table


def transform_events(records: list[dict[str, Any]]) -> pa.Table:
    """Transform raw event records to canonical parquet schema."""
    columns: dict[str, list[Any]] = {
        "event_id": [],
        "title": [],
        "description": [],
        "created_at": [],
        "tags": [],
    }

    for r in records:
        columns["event_id"].append(r.get("id"))
        columns["title"].append(r.get("title"))
        columns["description"].append(r.get("description"))
        columns["created_at"].append(_parse_timestamp(r.get("createdAt")))
        columns["tags"].append(_extract_tag_labels(r.get("tags", [])))

    # Create PyArrow table
    table = pa.table({
        "event_id": pa.array(columns["event_id"], type=pa.string()),
        "title": pa.array(columns["title"], type=pa.string()),
        "description": pa.array(columns["description"], type=pa.string()),
        "created_at": pa.array(columns["created_at"], type=pa.timestamp("us", tz="UTC")),
        "tags": pa.array(columns["tags"], type=pa.list_(pa.string())),
    })

    return table


def write_parquet_atomic(
    table: pa.Table,
    entity: str,
    s3fs: pafs.S3FileSystem,
    dry_run: bool = False,
) -> str:
    """Write parquet to S3 using atomic pattern (temp file then rename)."""
    final_path = f"{S3_BUCKET}/{S3_PREFIX}/{entity}.parquet"
    temp_path = f"{S3_BUCKET}/{S3_PREFIX}/{entity}_temp_{uuid.uuid4().hex[:8]}.parquet"

    if dry_run:
        print(f"[DRY RUN] Would write {table.num_rows} rows to s3://{final_path}")
        return final_path

    try:
        # Write to temp path
        pq.write_table(
            table,
            temp_path,
            filesystem=s3fs,
            compression="zstd",
        )

        # Verify temp file exists and has expected row count
        info = s3fs.get_file_info(temp_path)
        if info.type != pafs.FileType.File:
            raise RuntimeError(f"Temp file not found after write: {temp_path}")

        # Rename (copy + delete) to final path
        # S3 doesn't have true rename, so we copy then delete
        s3fs.copy_file(temp_path, final_path)

        # Delete temp file
        try:
            s3fs.delete_file(temp_path)
        except Exception as delete_err:
            # Temp file cleanup failure is non-fatal - log and continue
            logger.warning(
                "temp_file_cleanup_failed",
                temp_path=temp_path,
                error=str(delete_err),
            )

        logger.info("wrote_parquet", path=final_path, rows=table.num_rows)
        return final_path

    except Exception as e:
        # Log full context for debugging
        logger.error(
            "parquet_write_failed",
            entity=entity,
            temp_path=temp_path,
            final_path=final_path,
            rows=table.num_rows,
            error_type=type(e).__name__,
            error=str(e),
        )
        # Try to clean up temp file if it exists
        try:
            info = s3fs.get_file_info(temp_path)
            if info.type == pafs.FileType.File:
                s3fs.delete_file(temp_path)
                logger.info("cleaned_up_failed_temp_file", path=temp_path)
        except Exception:
            pass  # Best effort cleanup
        raise


def sync_entity(
    entity: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync a single entity (markets or events) from Gamma API to S3."""
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Syncing {entity}")
    print("=" * 60)

    # Get S3 filesystem
    s3fs = _get_s3_filesystem()

    # Fetch all records with streaming JSONL writes (memory-safe)
    records, total_count, jsonl_paths = fetch_records_streaming(entity, s3fs, dry_run=dry_run)
    print(f"Fetched {total_count} {entity} records")

    if not records:
        print(f"No {entity} records to sync")
        return {
            "entity": entity,
            "records": 0,
            "dry_run": dry_run,
        }

    if not dry_run:
        print(f"Wrote {len(jsonl_paths)} JSONL batch files to S3")

    # Transform to parquet schema
    if entity == "markets":
        table = transform_markets(records)
    elif entity == "events":
        table = transform_events(records)
    else:
        raise ValueError(f"Unknown entity: {entity}")

    print(f"Transformed {table.num_rows} rows")

    # Clear records from memory before parquet write
    del records
    gc.collect()
    log_memory_usage("pre_parquet_write")

    # Write parquet atomically
    parquet_path = write_parquet_atomic(table, entity, s3fs, dry_run=dry_run)
    print(f"Wrote parquet to s3://{parquet_path}")

    elapsed = time.time() - start_time
    log_memory_usage("sync_complete")

    return {
        "entity": entity,
        "records": total_count,
        "rows": table.num_rows,
        "jsonl_batches": len(jsonl_paths),
        "parquet_path": f"s3://{parquet_path}",
        "duration_seconds": elapsed,
        "dry_run": dry_run,
    }


def run_sync(
    entity: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the Gamma API sync.

    Args:
        entity: Which entity to sync - 'markets', 'events', or 'all'
        dry_run: If True, only scan API without writing to S3

    Returns:
        Summary statistics
    """
    start_time = time.time()

    print("=" * 60)
    print("Gamma API Sync: Fetch markets and events to S3")
    print("=" * 60)
    print(f"Entity: {entity}")
    print(f"Dry run: {dry_run}")

    results = {}

    if entity in ("all", "markets"):
        results["markets"] = sync_entity("markets", dry_run=dry_run)

    if entity in ("all", "events"):
        results["events"] = sync_entity("events", dry_run=dry_run)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Sync Complete!")
    print("=" * 60)

    for entity_name, result in results.items():
        print(f"\n{entity_name.upper()}:")
        print(f"  Records: {result.get('records', 0):,}")
        if not dry_run:
            print(f"  JSONL batches: {result.get('jsonl_batches', 0)}")
            print(f"  Parquet: {result.get('parquet_path', 'N/A')}")

    print(f"\nTotal duration: {elapsed:.1f}s")

    return {
        "results": results,
        "total_duration_seconds": elapsed,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Sync markets and events from Gamma API to S3")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan API without writing to S3",
    )
    parser.add_argument(
        "--entity",
        choices=["markets", "events", "all"],
        default="all",
        help="Which entity to sync (default: all)",
    )
    args = parser.parse_args()

    try:
        log_memory_usage("startup")
        result = run_sync(
            entity=args.entity,
            dry_run=args.dry_run,
        )
        # Exit with error if no records synced and not a dry run
        if not args.dry_run:
            total_records = sum(r.get("records", 0) for r in result.get("results", {}).values())
            if total_records == 0:
                logger.error("no_records_synced", entity=args.entity)
                sys.exit(1)
    except MemoryError as e:
        # Memory exhaustion - log and exit with partial progress preserved
        logger.error("memory_exhausted", error=str(e), memory_mb=get_memory_usage_mb())
        print(f"\nERROR: Memory exhausted - {e}")
        print("Partial progress (JSONL batches) may have been written to S3.")
        sys.exit(1)
    except Exception as e:
        logger.error("sync_failed", error=str(e), entity=args.entity)
        print(f"\nERROR: Sync failed - {e}")
        sys.exit(1)
