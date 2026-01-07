"""Tests for the bootstrap import pipeline."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_markets_csv(temp_dir: Path) -> Path:
    """Create a sample markets CSV file."""
    csv_path = temp_dir / "markets.csv"
    csv_path.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category
2024-01-01 00:00:00,market1,Will it rain?,Yes,No,false,will-it-rain,token1a,token1b,cond1,1000.0,RAIN,2024-12-31 23:59:59,Weather prediction,Weather
2024-01-02 00:00:00,market2,Who wins?,A,B,true,who-wins,token2a,token2b,cond2,5000.0,WIN,2024-06-30 23:59:59,Election prediction,Politics
"""
    )
    return csv_path


@pytest.fixture
def sample_trades_csv(temp_dir: Path) -> Path:
    """Create a sample trades CSV file."""
    csv_path = temp_dir / "trades.csv"
    csv_path.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
"""
    )
    return csv_path


@pytest.fixture
def sample_order_filled_csv(temp_dir: Path) -> Path:
    """Create a sample order filled CSV file."""
    csv_path = temp_dir / "orderFilled.csv"
    csv_path.write_text(
        """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash1
2024-01-15 11:00:00,0xmaker2,asset3,50.0,0xtaker2,asset4,142.85,0xhash2
"""
    )
    return csv_path


def test_bootstrap_creates_parquet_files(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> None:
    """Test that bootstrap creates Parquet files from CSVs."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    summary = run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    # Verify summary
    assert summary.status == "completed"
    assert summary.markets_rows == 2
    assert summary.trades_rows == 3
    assert summary.order_filled_rows == 2
    assert len(summary.parquet_files) == 3

    # Verify Parquet files exist
    assert (parquet_dir / "markets.parquet").exists()
    assert (parquet_dir / "trades.parquet").exists()
    assert (parquet_dir / "order_filled.parquet").exists()


def test_bootstrap_parquet_schema(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> None:
    """Test that Parquet files have correct schema."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    # Check markets schema
    markets_table = pq.read_table(parquet_dir / "markets.parquet")
    assert "id" in markets_table.column_names
    assert "question" in markets_table.column_names
    assert "created_at" in markets_table.column_names
    assert "closed_time" in markets_table.column_names
    assert "category" in markets_table.column_names

    # Check trades schema
    trades_table = pq.read_table(parquet_dir / "trades.parquet")
    assert "timestamp" in trades_table.column_names
    assert "market_id" in trades_table.column_names
    assert "price" in trades_table.column_names
    assert "transaction_hash" in trades_table.column_names


def test_bootstrap_creates_run_record(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> None:
    """Test that bootstrap creates a run record in metadata store."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    summary = run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    # Check run record
    metadata_store = MetadataStore(metadata_path)
    run_record = metadata_store.get_run(summary.run_id)

    assert run_record is not None
    assert run_record.run_type == "bootstrap"
    assert run_record.status == "completed"
    assert run_record.rows_read["markets"] == 2
    assert run_record.rows_written["trades"] == 3
    assert run_record.duration_seconds is not None
    assert run_record.duration_seconds > 0


def test_bootstrap_creates_duckdb_views(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> None:
    """Test that bootstrap creates DuckDB views."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    # Verify views work
    duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
    try:
        counts = duckdb_layer.verify_views()
        assert counts["v_markets"] == 2
        assert counts["v_trades"] == 3
        assert counts["v_order_filled"] == 2
        assert counts["v_trades_with_markets"] == 3
    finally:
        duckdb_layer.close()


def test_bootstrap_sets_watermarks(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> None:
    """Test that bootstrap sets watermarks after completion."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    metadata_store = MetadataStore(metadata_path)
    watermarks = metadata_store.get_all_watermarks()

    assert "trades" in watermarks
    assert "markets" in watermarks
    assert "order_filled" in watermarks
    assert "bootstrap_completed" in watermarks["trades"]


def test_duckdb_query_trades(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> None:
    """Test querying trades through DuckDB layer."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
    try:
        duckdb_layer.create_views()

        # Query single market
        trades = duckdb_layer.query_trades(market_id="market1")
        assert len(trades) == 2
        assert all(t["market_id"] == "market1" for t in trades)

        # Query multiple markets
        trades = duckdb_layer.query_trades(market_ids=["market1", "market2"])
        assert len(trades) == 3
    finally:
        duckdb_layer.close()


def test_bootstrap_fails_with_missing_csv(temp_dir: Path) -> None:
    """Test that bootstrap fails when CSV files don't exist."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    # Non-existent CSV files
    markets_csv = temp_dir / "markets.csv"
    trades_csv = temp_dir / "trades.csv"
    order_filled_csv = temp_dir / "orderFilled.csv"

    # Bootstrap should raise an error when CSVs don't exist
    with pytest.raises(Exception):
        run_bootstrap(
            markets_csv=markets_csv,
            trades_csv=trades_csv,
            order_filled_csv=order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )
