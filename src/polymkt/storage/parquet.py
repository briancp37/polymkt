"""Parquet storage layer for analytics data."""

from pathlib import Path
from typing import Iterator

import pyarrow as pa
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


class ParquetWriter:
    """Writes DataFrames to Parquet with proper schema and compression."""

    def __init__(
        self,
        output_dir: Path,
        compression: str = "zstd",
        row_group_size: int = 100_000,
    ) -> None:
        self.output_dir = output_dir
        self.compression = compression
        self.row_group_size = row_group_size
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
        """Write trades data to Parquet."""
        output_path = self.output_dir / "trades.parquet"
        table = table.cast(TRADES_SCHEMA)
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
        )
        return output_path

    def write_order_filled(self, table: pa.Table) -> Path:
        """Write order filled data to Parquet."""
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
