"""Bootstrap import pipeline for loading CSV data into Parquet."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pyarrow as pa
import pyarrow.csv as csv
import structlog

from polymkt.config import settings
from polymkt.models.schemas import BootstrapSummary, RunRecord
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.parquet import (
    MARKETS_SCHEMA,
    ORDER_FILLED_SCHEMA,
    TRADES_SCHEMA,
    ParquetWriter,
)

logger = structlog.get_logger()

# Column mappings from CSV to our internal schema
MARKETS_COLUMN_MAPPING = {
    "createdAt": "created_at",
    "closedTime": "closed_time",
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


def _rename_columns(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    """Rename columns in a PyArrow table."""
    new_names = [mapping.get(name, name) for name in table.column_names]
    return table.rename_columns(new_names)


def _read_csv_with_schema(
    csv_path: Path,
    target_schema: pa.Schema,
    column_mapping: dict[str, str],
) -> pa.Table:
    """Read a CSV file and normalize to target schema."""
    logger.info("reading_csv", path=str(csv_path))

    # Read CSV with type inference
    read_options = csv.ReadOptions()
    parse_options = csv.ParseOptions()
    convert_options = csv.ConvertOptions(
        timestamp_parsers=[
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ],
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
                    col = pa.compute.strptime(col, format="%Y-%m-%d %H:%M:%S", unit="us")
                    col = col.cast(field.type)
                else:
                    raise
            columns.append(col)
        else:
            # Create null column for missing fields
            columns.append(pa.nulls(table.num_rows, type=field.type))

    return pa.table(dict(zip([f.name for f in target_schema], columns)), schema=target_schema)


def run_bootstrap(
    markets_csv: Path | None = None,
    trades_csv: Path | None = None,
    order_filled_csv: Path | None = None,
    parquet_dir: Path | None = None,
    duckdb_path: Path | None = None,
    metadata_db_path: Path | None = None,
) -> BootstrapSummary:
    """
    Run the bootstrap import process.

    Reads existing CSV files and converts them to Parquet format,
    then creates DuckDB views over the Parquet files.
    """
    # Use settings defaults if not provided
    markets_csv = markets_csv or settings.markets_csv
    trades_csv = trades_csv or settings.trades_csv
    order_filled_csv = order_filled_csv or settings.order_filled_csv
    parquet_dir = parquet_dir or settings.parquet_dir
    duckdb_path = duckdb_path or settings.duckdb_path
    metadata_db_path = metadata_db_path or settings.metadata_db_path

    # Initialize components
    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    timer_start = perf_counter()

    metadata_store = MetadataStore(metadata_db_path)
    parquet_writer = ParquetWriter(parquet_dir)

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

    try:
        # Process markets
        if markets_csv.exists():
            logger.info("processing_markets", path=str(markets_csv))
            markets_table = _read_csv_with_schema(
                markets_csv, MARKETS_SCHEMA, MARKETS_COLUMN_MAPPING
            )
            rows_read["markets"] = markets_table.num_rows
            output_path = parquet_writer.write_markets(markets_table)
            rows_written["markets"] = markets_table.num_rows
            parquet_files.append(str(output_path))
        else:
            logger.warning("csv_not_found", path=str(markets_csv))
            rows_read["markets"] = 0
            rows_written["markets"] = 0

        # Process trades
        if trades_csv.exists():
            logger.info("processing_trades", path=str(trades_csv))
            trades_table = _read_csv_with_schema(
                trades_csv, TRADES_SCHEMA, TRADES_COLUMN_MAPPING
            )
            rows_read["trades"] = trades_table.num_rows
            output_path = parquet_writer.write_trades(trades_table)
            rows_written["trades"] = trades_table.num_rows
            parquet_files.append(str(output_path))
        else:
            logger.warning("csv_not_found", path=str(trades_csv))
            rows_read["trades"] = 0
            rows_written["trades"] = 0

        # Process order filled
        if order_filled_csv.exists():
            logger.info("processing_order_filled", path=str(order_filled_csv))
            order_filled_table = _read_csv_with_schema(
                order_filled_csv, ORDER_FILLED_SCHEMA, ORDER_FILLED_COLUMN_MAPPING
            )
            rows_read["order_filled"] = order_filled_table.num_rows
            output_path = parquet_writer.write_order_filled(order_filled_table)
            rows_written["order_filled"] = order_filled_table.num_rows
            parquet_files.append(str(output_path))
        else:
            logger.warning("csv_not_found", path=str(order_filled_csv))
            rows_read["order_filled"] = 0
            rows_written["order_filled"] = 0

        # Create DuckDB views
        logger.info("creating_duckdb_views")
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            created_views = duckdb_layer.create_views()
            view_counts = duckdb_layer.verify_views()
            logger.info("views_created", views=created_views, counts=view_counts)
        finally:
            duckdb_layer.close()

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

        logger.info(
            "bootstrap_completed",
            run_id=run_id,
            duration_seconds=duration,
            rows_read=rows_read,
            rows_written=rows_written,
        )

        return BootstrapSummary(
            run_id=run_id,
            status="completed",
            duration_seconds=duration,
            markets_rows=rows_written.get("markets", 0),
            trades_rows=rows_written.get("trades", 0),
            order_filled_rows=rows_written.get("order_filled", 0),
            schema_version="1.0.0",
            parquet_files=parquet_files,
        )

    except Exception as e:
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
