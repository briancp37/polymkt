"""Cloud data lake bootstrap for S3-backed Parquet storage.

Implements the PRD requirements for cloud data lake bootstrap:
- Detect whether S3 Parquet datasets already exist
- Convert local CSVs to partitioned Parquet
- Upload to S3 with raw/curated layer structure
- Generate verification reports comparing CSV vs Parquet
- Block CSV deletion until verification passes
"""

import hashlib
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import structlog

from polymkt.config import settings
from polymkt.models.schemas import (
    CloudBootstrapResult,
    CloudBootstrapVerificationReport,
    DatasetVerificationStats,
    S3DataLakeStatus,
    S3DatasetStatus,
)
from polymkt.pipeline.bootstrap import (
    EVENTS_COLUMN_MAPPING,
    MARKETS_COLUMN_MAPPING,
    ORDER_FILLED_COLUMN_MAPPING,
    TRADES_COLUMN_MAPPING,
    _read_csv_with_schema,
    _read_events_csv,
    _join_events_tags_to_markets,
    validate_schema_requirements,
)
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.parquet import (
    EVENTS_SCHEMA,
    MARKETS_SCHEMA,
    ORDER_FILLED_SCHEMA,
    TRADES_SCHEMA,
    ParquetWriter,
)

logger = structlog.get_logger()


def _try_import_boto3() -> Any:
    """Try to import boto3, return None if not available."""
    try:
        import boto3
        return boto3
    except ImportError:
        return None


def check_s3_dataset_exists(
    bucket: str,
    prefix: str,
    dataset: str,
    region: str = "us-east-1",
) -> S3DatasetStatus:
    """
    Check if a dataset exists in S3.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., raw/polymarket)
        dataset: Dataset name (trades, events, markets)
        region: AWS region

    Returns:
        S3DatasetStatus with existence and metadata
    """
    boto3 = _try_import_boto3()
    if boto3 is None:
        return S3DatasetStatus(
            dataset=dataset,
            exists_in_s3=False,
            s3_key=None,
            s3_row_count=None,
            last_modified=None,
        )

    s3_key = f"{prefix}/{dataset}/"
    try:
        s3 = boto3.client("s3", region_name=region)

        # List objects under the dataset prefix
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=s3_key,
            MaxKeys=1,
        )

        exists = response.get("KeyCount", 0) > 0

        if exists and response.get("Contents"):
            last_modified = response["Contents"][0].get("LastModified")
            return S3DatasetStatus(
                dataset=dataset,
                exists_in_s3=True,
                s3_key=s3_key,
                s3_row_count=None,  # Would need to read Parquet to get this
                last_modified=last_modified,
            )

        return S3DatasetStatus(
            dataset=dataset,
            exists_in_s3=False,
            s3_key=s3_key,
            s3_row_count=None,
            last_modified=None,
        )

    except Exception as e:
        logger.warning("s3_check_failed", dataset=dataset, error=str(e))
        return S3DatasetStatus(
            dataset=dataset,
            exists_in_s3=False,
            s3_key=s3_key,
            s3_row_count=None,
            last_modified=None,
        )


def get_s3_data_lake_status() -> S3DataLakeStatus:
    """
    Get the current status of the S3 data lake.

    Returns:
        S3DataLakeStatus with per-dataset existence info
    """
    bucket = settings.s3_bucket
    region = settings.s3_region
    raw_prefix = settings.s3_raw_prefix
    curated_prefix = settings.s3_curated_prefix
    s3_enabled = settings.s3_enabled and bucket is not None

    datasets_to_check = ["trades", "markets", "events", "order_filled"]
    dataset_statuses = []

    if s3_enabled and bucket:
        for dataset in datasets_to_check:
            status = check_s3_dataset_exists(bucket, raw_prefix, dataset, region)
            dataset_statuses.append(status)

    # Bootstrap is required if S3 is enabled and any dataset is missing
    bootstrap_required = s3_enabled and any(
        not ds.exists_in_s3 for ds in dataset_statuses
    )

    return S3DataLakeStatus(
        bucket=bucket,
        region=region,
        raw_prefix=raw_prefix,
        curated_prefix=curated_prefix,
        s3_enabled=s3_enabled,
        datasets=dataset_statuses,
        bootstrap_required=bootstrap_required,
    )


def _count_csv_rows(csv_path: Path) -> int:
    """Count rows in a CSV file without loading it all into memory."""
    if not csv_path.exists():
        return 0

    count = 0
    with open(csv_path, "r") as f:
        # Skip header
        next(f, None)
        for _ in f:
            count += 1
    return count


def _get_parquet_row_count(parquet_path: Path) -> int:
    """Get row count from Parquet file or directory."""
    if not parquet_path.exists():
        return 0

    if parquet_path.is_dir():
        # Partitioned dataset - count all files
        total = 0
        for pq_file in parquet_path.rglob("*.parquet"):
            metadata = pq.read_metadata(pq_file)
            total += metadata.num_rows
        return total
    else:
        # Single file
        metadata = pq.read_metadata(parquet_path)
        return int(metadata.num_rows)


def _get_timestamp_range(
    parquet_path: Path, timestamp_col: str
) -> tuple[datetime | None, datetime | None]:
    """Get min/max timestamps from Parquet file."""
    if not parquet_path.exists():
        return None, None

    try:
        if parquet_path.is_dir():
            # Read all files in directory
            import pyarrow.dataset as ds
            dataset = ds.dataset(parquet_path, format="parquet")
            table = dataset.to_table(columns=[timestamp_col])
        else:
            table = pq.read_table(parquet_path, columns=[timestamp_col])

        if table.num_rows == 0:
            return None, None

        ts_col = table.column(timestamp_col)
        min_ts = ts_col.to_pandas().min()
        max_ts = ts_col.to_pandas().max()

        return min_ts.to_pydatetime(), max_ts.to_pydatetime()
    except Exception as e:
        logger.warning("timestamp_range_failed", path=str(parquet_path), error=str(e))
        return None, None


def _count_distinct_market_ids(parquet_path: Path) -> int | None:
    """Count distinct market IDs in Parquet file."""
    if not parquet_path.exists():
        return None

    try:
        if parquet_path.is_dir():
            import pyarrow.dataset as ds
            dataset = ds.dataset(parquet_path, format="parquet")
            table = dataset.to_table(columns=["market_id"])
        else:
            table = pq.read_table(parquet_path, columns=["market_id"])

        unique_ids = table.column("market_id").unique()
        return len(unique_ids)
    except Exception:
        return None


def _sample_verify_dataset(
    csv_path: Path,
    parquet_path: Path,
    key_column: str,
    sample_size: int = 100,
) -> tuple[int, int, int]:
    """
    Verify dataset by sampling rows and comparing key columns.

    Args:
        csv_path: Path to source CSV
        parquet_path: Path to Parquet output
        key_column: Column to use for matching (e.g., transaction_hash, id)
        sample_size: Number of rows to sample

    Returns:
        (sample_size, matches, mismatches)
    """
    if not csv_path.exists() or not parquet_path.exists():
        return 0, 0, 0

    try:
        # Read Parquet keys
        if parquet_path.is_dir():
            import pyarrow.dataset as ds
            dataset = ds.dataset(parquet_path, format="parquet")
            pq_table = dataset.to_table(columns=[key_column])
        else:
            pq_table = pq.read_table(parquet_path, columns=[key_column])

        pq_keys = set(pq_table.column(key_column).to_pylist())

        # Read CSV and sample
        csv_table = csv.read_csv(csv_path)

        # Map column name if needed
        csv_key_col = key_column
        for old_name, new_name in {
            **TRADES_COLUMN_MAPPING,
            **MARKETS_COLUMN_MAPPING,
            **EVENTS_COLUMN_MAPPING,
        }.items():
            if new_name == key_column and old_name in csv_table.column_names:
                csv_key_col = old_name
                break

        if csv_key_col not in csv_table.column_names:
            return 0, 0, 0

        csv_keys = csv_table.column(csv_key_col).to_pylist()

        # Sample random keys from CSV
        actual_sample_size = min(sample_size, len(csv_keys))
        if actual_sample_size == 0:
            return 0, 0, 0

        sampled_keys = random.sample(csv_keys, actual_sample_size)

        # Count matches
        matches = sum(1 for k in sampled_keys if k in pq_keys)
        mismatches = actual_sample_size - matches

        return actual_sample_size, matches, mismatches

    except Exception as e:
        logger.warning(
            "sample_verification_failed",
            csv_path=str(csv_path),
            parquet_path=str(parquet_path),
            error=str(e),
        )
        return 0, 0, 0


def generate_verification_report(
    run_id: str,
    parquet_dir: Path,
    markets_csv: Path | None = None,
    trades_csv: Path | None = None,
    events_csv: Path | None = None,
    order_filled_csv: Path | None = None,
    s3_bucket: str | None = None,
    s3_raw_prefix: str | None = None,
    s3_curated_prefix: str | None = None,
    sample_size: int = 100,
) -> CloudBootstrapVerificationReport:
    """
    Generate a verification report comparing CSVs to Parquet output.

    Args:
        run_id: Bootstrap run ID
        parquet_dir: Directory containing Parquet files
        markets_csv: Path to markets CSV
        trades_csv: Path to trades CSV
        events_csv: Path to events CSV
        order_filled_csv: Path to order_filled CSV
        s3_bucket: S3 bucket if used
        s3_raw_prefix: S3 raw prefix if used
        s3_curated_prefix: S3 curated prefix if used
        sample_size: Number of rows to sample for verification

    Returns:
        CloudBootstrapVerificationReport
    """
    markets_csv = markets_csv or settings.markets_csv
    trades_csv = trades_csv or settings.trades_csv
    events_csv = events_csv or settings.events_csv
    order_filled_csv = order_filled_csv or settings.order_filled_csv

    dataset_stats = []

    # Verify trades
    trades_parquet = parquet_dir / "trades.parquet"
    if not trades_parquet.exists():
        trades_parquet = parquet_dir / "trades"  # Partitioned

    trades_csv_count = _count_csv_rows(trades_csv)
    trades_pq_count = _get_parquet_row_count(trades_parquet)
    trades_min_ts, trades_max_ts = _get_timestamp_range(trades_parquet, "timestamp")
    trades_market_ids = _count_distinct_market_ids(trades_parquet)
    trades_samples, trades_matches, trades_mismatches = _sample_verify_dataset(
        trades_csv, trades_parquet, "transaction_hash", sample_size
    )

    trades_passed = (
        trades_csv_count == trades_pq_count
        and (trades_samples == 0 or trades_mismatches == 0)
    )

    dataset_stats.append(DatasetVerificationStats(
        dataset="trades",
        csv_row_count=trades_csv_count,
        parquet_row_count=trades_pq_count,
        row_count_match=trades_csv_count == trades_pq_count,
        min_timestamp=trades_min_ts,
        max_timestamp=trades_max_ts,
        distinct_market_ids=trades_market_ids,
        sample_size=trades_samples,
        sample_matches=trades_matches,
        sample_mismatches=trades_mismatches,
        verification_passed=trades_passed,
    ))

    # Verify markets
    markets_parquet = parquet_dir / "markets.parquet"
    markets_csv_count = _count_csv_rows(markets_csv)
    markets_pq_count = _get_parquet_row_count(markets_parquet)
    markets_samples, markets_matches, markets_mismatches = _sample_verify_dataset(
        markets_csv, markets_parquet, "id", sample_size
    )

    markets_passed = (
        markets_csv_count == markets_pq_count
        and (markets_samples == 0 or markets_mismatches == 0)
    )

    dataset_stats.append(DatasetVerificationStats(
        dataset="markets",
        csv_row_count=markets_csv_count,
        parquet_row_count=markets_pq_count,
        row_count_match=markets_csv_count == markets_pq_count,
        min_timestamp=None,
        max_timestamp=None,
        distinct_market_ids=markets_pq_count,  # Each market is unique
        sample_size=markets_samples,
        sample_matches=markets_matches,
        sample_mismatches=markets_mismatches,
        verification_passed=markets_passed,
    ))

    # Verify events
    events_parquet = parquet_dir / "events.parquet"
    events_csv_count = _count_csv_rows(events_csv)
    events_pq_count = _get_parquet_row_count(events_parquet)
    events_samples, events_matches, events_mismatches = _sample_verify_dataset(
        events_csv, events_parquet, "event_id", sample_size
    )

    events_passed = (
        events_csv_count == events_pq_count
        and (events_samples == 0 or events_mismatches == 0)
    )

    dataset_stats.append(DatasetVerificationStats(
        dataset="events",
        csv_row_count=events_csv_count,
        parquet_row_count=events_pq_count,
        row_count_match=events_csv_count == events_pq_count,
        min_timestamp=None,
        max_timestamp=None,
        distinct_market_ids=None,
        sample_size=events_samples,
        sample_matches=events_matches,
        sample_mismatches=events_mismatches,
        verification_passed=events_passed,
    ))

    # Verify order_filled
    order_filled_parquet = parquet_dir / "order_filled.parquet"
    of_csv_count = _count_csv_rows(order_filled_csv)
    of_pq_count = _get_parquet_row_count(order_filled_parquet)
    of_min_ts, of_max_ts = _get_timestamp_range(order_filled_parquet, "timestamp")
    of_samples, of_matches, of_mismatches = _sample_verify_dataset(
        order_filled_csv, order_filled_parquet, "transaction_hash", sample_size
    )

    of_passed = (
        of_csv_count == of_pq_count
        and (of_samples == 0 or of_mismatches == 0)
    )

    dataset_stats.append(DatasetVerificationStats(
        dataset="order_filled",
        csv_row_count=of_csv_count,
        parquet_row_count=of_pq_count,
        row_count_match=of_csv_count == of_pq_count,
        min_timestamp=of_min_ts,
        max_timestamp=of_max_ts,
        distinct_market_ids=None,
        sample_size=of_samples,
        sample_matches=of_matches,
        sample_mismatches=of_mismatches,
        verification_passed=of_passed,
    ))

    # Overall pass if all non-empty datasets pass
    overall_passed = all(
        ds.verification_passed
        for ds in dataset_stats
        if ds.csv_row_count > 0 or ds.parquet_row_count > 0
    )

    return CloudBootstrapVerificationReport(
        run_id=run_id,
        verification_time=datetime.now(timezone.utc),
        datasets=dataset_stats,
        overall_passed=overall_passed,
        s3_bucket=s3_bucket,
        s3_raw_prefix=s3_raw_prefix,
        s3_curated_prefix=s3_curated_prefix,
        local_parquet_dir=str(parquet_dir),
        csv_deletion_blocked=not overall_passed,
    )


def upload_parquet_to_s3(
    local_path: Path,
    bucket: str,
    s3_key: str,
    region: str = "us-east-1",
) -> bool:
    """
    Upload a Parquet file or directory to S3.

    Args:
        local_path: Local Parquet file or directory
        bucket: S3 bucket
        s3_key: S3 key prefix
        region: AWS region

    Returns:
        True if upload succeeded
    """
    boto3 = _try_import_boto3()
    if boto3 is None:
        logger.warning("boto3_not_available", message="S3 upload skipped")
        return False

    try:
        s3 = boto3.client("s3", region_name=region)

        if local_path.is_dir():
            # Upload all files in directory
            for pq_file in local_path.rglob("*.parquet"):
                relative_path = pq_file.relative_to(local_path)
                file_key = f"{s3_key}/{relative_path}"
                s3.upload_file(str(pq_file), bucket, file_key)
                logger.info("s3_file_uploaded", local=str(pq_file), s3_key=file_key)
        else:
            # Upload single file
            s3.upload_file(str(local_path), bucket, s3_key)
            logger.info("s3_file_uploaded", local=str(local_path), s3_key=s3_key)

        return True

    except Exception as e:
        logger.error("s3_upload_failed", local=str(local_path), error=str(e))
        return False


def run_cloud_bootstrap(
    markets_csv: Path | None = None,
    trades_csv: Path | None = None,
    order_filled_csv: Path | None = None,
    events_csv: Path | None = None,
    parquet_dir: Path | None = None,
    duckdb_path: Path | None = None,
    metadata_db_path: Path | None = None,
    s3_bucket: str | None = None,
    s3_region: str | None = None,
    s3_raw_prefix: str | None = None,
    s3_curated_prefix: str | None = None,
    upload_to_s3: bool | None = None,
    verification_sample_size: int = 100,
    fail_on_verification_error: bool = True,
) -> CloudBootstrapResult:
    """
    Run cloud data lake bootstrap with verification.

    This is the main entry point for bootstrapping the cloud data lake:
    1. Checks if S3 datasets already exist (skip if present)
    2. Converts local CSVs to partitioned Parquet
    3. Generates verification report
    4. Uploads to S3 if enabled and verification passes
    5. Blocks CSV deletion until verification passes

    Args:
        markets_csv: Path to markets CSV
        trades_csv: Path to trades CSV
        order_filled_csv: Path to order_filled CSV
        events_csv: Path to events CSV
        parquet_dir: Output directory for Parquet files
        duckdb_path: Path to DuckDB database
        metadata_db_path: Path to metadata database
        s3_bucket: S3 bucket for upload
        s3_region: AWS region
        s3_raw_prefix: S3 prefix for raw data
        s3_curated_prefix: S3 prefix for curated data
        upload_to_s3: Whether to upload to S3
        verification_sample_size: Number of rows to sample for verification
        fail_on_verification_error: If True, fail if verification doesn't pass

    Returns:
        CloudBootstrapResult with status and verification report
    """
    # Apply defaults from settings
    markets_csv = markets_csv or settings.markets_csv
    trades_csv = trades_csv or settings.trades_csv
    order_filled_csv = order_filled_csv or settings.order_filled_csv
    events_csv = events_csv or settings.events_csv
    parquet_dir = parquet_dir or settings.parquet_dir
    duckdb_path = duckdb_path or settings.duckdb_path
    metadata_db_path = metadata_db_path or settings.metadata_db_path
    s3_bucket = s3_bucket or settings.s3_bucket
    s3_region = s3_region or settings.s3_region
    s3_raw_prefix = s3_raw_prefix or settings.s3_raw_prefix
    s3_curated_prefix = s3_curated_prefix or settings.s3_curated_prefix
    upload_to_s3 = upload_to_s3 if upload_to_s3 is not None else settings.s3_enabled

    run_id = str(uuid.uuid4())
    start_time = perf_counter()
    local_parquet_files: list[str] = []
    s3_uploads: list[str] = []

    logger.info(
        "cloud_bootstrap_starting",
        run_id=run_id,
        s3_enabled=upload_to_s3,
        s3_bucket=s3_bucket,
    )

    try:
        # Step 1: Check if S3 datasets already exist
        if upload_to_s3 and s3_bucket:
            status = get_s3_data_lake_status()
            if not status.bootstrap_required:
                logger.info(
                    "cloud_bootstrap_skipped",
                    reason="S3 datasets already exist",
                )
                return CloudBootstrapResult(
                    run_id=run_id,
                    status="skipped",
                    duration_seconds=perf_counter() - start_time,
                    local_parquet_files=[],
                    s3_uploads=[],
                    verification_report=None,
                    csv_safe_to_delete=False,
                    error_message="S3 datasets already exist, bootstrap skipped",
                )

        # Step 2: Run the standard bootstrap to create local Parquet
        from polymkt.pipeline.bootstrap import run_bootstrap

        bootstrap_result = run_bootstrap(
            markets_csv=markets_csv,
            trades_csv=trades_csv,
            order_filled_csv=order_filled_csv,
            events_csv=events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_db_path,
            normalize_addresses=True,
            validate_data=True,
            partitioning_enabled=True,  # Always partition for cloud
            hash_bucket_count=8,
        )

        local_parquet_files = bootstrap_result.parquet_files

        # Step 3: Generate verification report
        verification_report = generate_verification_report(
            run_id=run_id,
            parquet_dir=parquet_dir,
            markets_csv=markets_csv,
            trades_csv=trades_csv,
            events_csv=events_csv,
            order_filled_csv=order_filled_csv,
            s3_bucket=s3_bucket,
            s3_raw_prefix=s3_raw_prefix,
            s3_curated_prefix=s3_curated_prefix,
            sample_size=verification_sample_size,
        )

        logger.info(
            "verification_completed",
            run_id=run_id,
            overall_passed=verification_report.overall_passed,
            datasets_verified=len(verification_report.datasets),
        )

        # Step 4: Check verification result
        if not verification_report.overall_passed:
            if fail_on_verification_error:
                return CloudBootstrapResult(
                    run_id=run_id,
                    status="verification_failed",
                    duration_seconds=perf_counter() - start_time,
                    local_parquet_files=local_parquet_files,
                    s3_uploads=[],
                    verification_report=verification_report,
                    csv_safe_to_delete=False,
                    error_message="Verification failed - CSV deletion blocked",
                )
            else:
                logger.warning(
                    "verification_failed_continuing",
                    run_id=run_id,
                    message="Verification failed but continuing per configuration",
                )

        # Step 5: Upload to S3 if enabled and verification passed
        if upload_to_s3 and s3_bucket and verification_report.overall_passed:
            datasets = [
                ("trades", parquet_dir / "trades"),
                ("markets", parquet_dir / "markets.parquet"),
                ("events", parquet_dir / "events.parquet"),
                ("order_filled", parquet_dir / "order_filled.parquet"),
            ]

            for dataset_name, local_path in datasets:
                if local_path.exists():
                    # Upload to raw layer
                    raw_key = f"{s3_raw_prefix}/{dataset_name}"
                    if upload_parquet_to_s3(local_path, s3_bucket, raw_key, s3_region):
                        s3_uploads.append(f"s3://{s3_bucket}/{raw_key}")

                    # Upload to curated layer (same data for now)
                    curated_key = f"{s3_curated_prefix}/{dataset_name}"
                    if upload_parquet_to_s3(
                        local_path, s3_bucket, curated_key, s3_region
                    ):
                        s3_uploads.append(f"s3://{s3_bucket}/{curated_key}")

        duration = perf_counter() - start_time

        logger.info(
            "cloud_bootstrap_completed",
            run_id=run_id,
            duration_seconds=duration,
            local_files=len(local_parquet_files),
            s3_uploads=len(s3_uploads),
            verification_passed=verification_report.overall_passed,
        )

        return CloudBootstrapResult(
            run_id=run_id,
            status="completed",
            duration_seconds=duration,
            local_parquet_files=local_parquet_files,
            s3_uploads=s3_uploads,
            verification_report=verification_report,
            csv_safe_to_delete=verification_report.overall_passed,
            error_message=None,
        )

    except Exception as e:
        duration = perf_counter() - start_time
        logger.error("cloud_bootstrap_failed", run_id=run_id, error=str(e))

        return CloudBootstrapResult(
            run_id=run_id,
            status="failed",
            duration_seconds=duration,
            local_parquet_files=local_parquet_files,
            s3_uploads=s3_uploads,
            verification_report=None,
            csv_safe_to_delete=False,
            error_message=str(e),
        )
