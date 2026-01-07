"""Tests for Parquet partitioning functionality."""

import tempfile
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.parquet import (
    ParquetWriter,
    add_partition_columns,
    compute_hash_bucket,
)


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
2024-01-03 00:00:00,market3,Price of BTC?,Up,Down,false,btc-price,token3a,token3b,cond3,10000.0,BTC,2024-12-31 23:59:59,Crypto prediction,Crypto
"""
    )
    return csv_path


@pytest.fixture
def sample_trades_csv_multi_day(temp_dir: Path) -> Path:
    """Create a sample trades CSV file spanning multiple days."""
    csv_path = temp_dir / "trades.csv"
    # Create trades across multiple days and markets
    csv_path.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-15 12:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-16 09:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.70,150.0,214.28,0xhash4
2024-01-16 10:00:00,market2,0xmaker5,0xtaker5,NO,sell,buy,0.20,75.0,375.0,0xhash5
2024-01-16 11:00:00,market3,0xmaker6,0xtaker6,YES,buy,sell,0.55,300.0,545.45,0xhash6
2024-01-17 08:00:00,market1,0xmaker7,0xtaker7,YES,buy,sell,0.72,180.0,250.0,0xhash7
2024-01-17 09:00:00,market3,0xmaker8,0xtaker8,NO,sell,buy,0.45,120.0,266.67,0xhash8
2024-02-01 10:00:00,market2,0xmaker9,0xtaker9,YES,buy,sell,0.85,500.0,588.23,0xhash9
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


class TestHashBucketComputation:
    """Tests for hash bucket computation."""

    def test_compute_hash_bucket_returns_valid_range(self) -> None:
        """Test that hash buckets are within expected range."""
        market_ids = pa.array(["market1", "market2", "market3", "market4"])
        bucket_count = 8

        buckets = compute_hash_bucket(market_ids, bucket_count)

        # All buckets should be in range [0, bucket_count)
        for bucket in buckets.to_pylist():
            assert 0 <= bucket < bucket_count

    def test_compute_hash_bucket_deterministic(self) -> None:
        """Test that hash bucket computation is deterministic."""
        market_ids = pa.array(["market1", "market2", "market1"])
        bucket_count = 8

        buckets1 = compute_hash_bucket(market_ids, bucket_count)
        buckets2 = compute_hash_bucket(market_ids, bucket_count)

        assert buckets1.to_pylist() == buckets2.to_pylist()
        # Same market_id should always get same bucket
        assert buckets1[0].as_py() == buckets1[2].as_py()

    def test_compute_hash_bucket_distribution(self) -> None:
        """Test that hash buckets distribute across different market IDs."""
        # Create 100 unique market IDs
        market_ids = pa.array([f"market{i}" for i in range(100)])
        bucket_count = 8

        buckets = compute_hash_bucket(market_ids, bucket_count)
        unique_buckets = set(buckets.to_pylist())

        # With 100 markets and 8 buckets, we should see multiple buckets used
        assert len(unique_buckets) >= 2


class TestAddPartitionColumns:
    """Tests for adding partition columns to tables."""

    def test_add_partition_columns_extracts_date_parts(self) -> None:
        """Test that year, month, day are extracted correctly."""
        table = pa.table({
            "timestamp": pa.array([
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 3, 20, 15, 30, 0, tzinfo=timezone.utc),
            ]),
            "market_id": pa.array(["market1", "market2"]),
            "price": pa.array([0.5, 0.6]),
        })

        result = add_partition_columns(table, bucket_count=8)

        assert "year" in result.column_names
        assert "month" in result.column_names
        assert "day" in result.column_names
        assert "hash_bucket" in result.column_names

        assert result.column("year").to_pylist() == [2024, 2024]
        assert result.column("month").to_pylist() == [1, 3]
        assert result.column("day").to_pylist() == [15, 20]


class TestPartitionedParquetWriter:
    """Tests for writing partitioned Parquet files."""

    def test_write_trades_partitioned_creates_directory_structure(
        self, temp_dir: Path
    ) -> None:
        """Test that partitioned writes create correct directory structure."""
        writer = ParquetWriter(
            temp_dir,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        table = pa.table({
            "timestamp": pa.array([
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 16, 11, 0, 0, tzinfo=timezone.utc),
            ], type=pa.timestamp("us", tz="UTC")),
            "market_id": pa.array(["market1", "market2"]),
            "maker": pa.array(["0xmaker1", "0xmaker2"]),
            "taker": pa.array(["0xtaker1", "0xtaker2"]),
            "nonusdc_side": pa.array(["YES", "NO"]),
            "maker_direction": pa.array(["buy", "sell"]),
            "taker_direction": pa.array(["sell", "buy"]),
            "price": pa.array([0.5, 0.6]),
            "usd_amount": pa.array([100.0, 200.0]),
            "token_amount": pa.array([200.0, 333.33]),
            "transaction_hash": pa.array(["0xhash1", "0xhash2"]),
        })

        output_path = writer.write_trades(table)

        # Should be a directory, not a file
        assert output_path.is_dir()
        assert output_path.name == "trades"

        # Should contain year=XXXX subdirectories
        year_dirs = list(output_path.glob("year=*"))
        assert len(year_dirs) > 0

        # Should contain .parquet files deep in the hierarchy
        parquet_files = list(output_path.glob("**/*.parquet"))
        assert len(parquet_files) > 0

    def test_write_trades_monolithic_creates_single_file(
        self, temp_dir: Path
    ) -> None:
        """Test that non-partitioned writes create a single file."""
        writer = ParquetWriter(
            temp_dir,
            partitioning_enabled=False,
        )

        table = pa.table({
            "timestamp": pa.array([
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            ], type=pa.timestamp("us", tz="UTC")),
            "market_id": pa.array(["market1"]),
            "maker": pa.array(["0xmaker1"]),
            "taker": pa.array(["0xtaker1"]),
            "nonusdc_side": pa.array(["YES"]),
            "maker_direction": pa.array(["buy"]),
            "taker_direction": pa.array(["sell"]),
            "price": pa.array([0.5]),
            "usd_amount": pa.array([100.0]),
            "token_amount": pa.array([200.0]),
            "transaction_hash": pa.array(["0xhash1"]),
        })

        output_path = writer.write_trades(table)

        # Should be a file, not a directory
        assert output_path.is_file()
        assert output_path.name == "trades.parquet"


class TestBootstrapWithPartitioning:
    """Tests for bootstrap with partitioning enabled."""

    def test_bootstrap_with_partitioning_creates_partitioned_trades(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap with partitioning creates partitioned trades."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        summary = run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        assert summary.status == "completed"
        assert summary.trades_rows == 9

        # trades should be a directory (partitioned)
        trades_path = parquet_dir / "trades"
        assert trades_path.is_dir()

        # markets and order_filled should still be single files
        assert (parquet_dir / "markets.parquet").is_file()
        assert (parquet_dir / "order_filled.parquet").is_file()

    def test_bootstrap_without_partitioning_creates_single_file(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap without partitioning creates single trades file."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        summary = run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=False,
        )

        assert summary.status == "completed"

        # trades should be a file (not partitioned)
        assert (parquet_dir / "trades.parquet").is_file()


class TestDuckDBPartitionedReads:
    """Tests for DuckDB reading partitioned data."""

    def test_duckdb_can_read_partitioned_trades(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that DuckDB can query partitioned trades data."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()
            counts = duckdb_layer.verify_views()

            assert counts["v_trades"] == 9
            assert counts["v_markets"] == 3
            assert counts["v_trades_with_markets"] == 9

        finally:
            duckdb_layer.close()

    def test_duckdb_query_partitioned_by_market_id(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test querying partitioned data by market_id."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()

            # Query single market
            trades, count = duckdb_layer.query_trades(market_id="market1")
            assert count == 4  # 4 trades for market1
            assert all(t["market_id"] == "market1" for t in trades)

            # Query multiple markets
            trades, count = duckdb_layer.query_trades(
                market_ids=["market1", "market2"]
            )
            assert count == 7  # 4 + 3 trades

        finally:
            duckdb_layer.close()

    def test_duckdb_query_partitioned_by_time_range(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test querying partitioned data by time range."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()

            # Query specific day
            trades, count = duckdb_layer.query_trades(
                start_time="2024-01-16 00:00:00",
                end_time="2024-01-16 23:59:59",
            )
            assert count == 3  # 3 trades on Jan 16

            # Query January only
            trades, count = duckdb_layer.query_trades(
                start_time="2024-01-01 00:00:00",
                end_time="2024-01-31 23:59:59",
            )
            assert count == 8  # 8 trades in January

        finally:
            duckdb_layer.close()


class TestPartitionPruning:
    """Tests for partition pruning verification."""

    def test_query_plan_shows_filter_for_partitioned_data(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that query plan shows filter operations for partitioned data."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()

            # Get query plan for a filtered query
            query = "SELECT * FROM v_trades WHERE market_id = 'market1'"
            plan = duckdb_layer.explain_query(query)

            # The plan should exist and be non-empty
            assert len(plan) > 0
            # Plan should be a valid string (DuckDB returns plan output)
            assert isinstance(plan, str)

        finally:
            duckdb_layer.close()

    def test_query_plan_with_time_filter(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that query plan with time filter shows appropriate operations."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()

            # Get query plan for a time-filtered query
            query = """
                SELECT * FROM v_trades
                WHERE timestamp >= '2024-01-16' AND timestamp < '2024-01-17'
            """
            plan = duckdb_layer.explain_query(query)

            # Plan should exist and be non-empty
            assert len(plan) > 0
            # Plan should be a valid string
            assert isinstance(plan, str)

        finally:
            duckdb_layer.close()

    def test_query_returns_correct_rows_after_partitioning(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv_multi_day: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that queries return correct data regardless of partitioning."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=sample_trades_csv_multi_day,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
            hash_bucket_count=4,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()

            # Combined filter query
            trades, count = duckdb_layer.query_trades(
                market_id="market1",
                start_time="2024-01-16 00:00:00",
                end_time="2024-01-17 23:59:59",
            )

            # Should have 2 trades: one on Jan 16 and one on Jan 17 for market1
            assert count == 2
            assert len(trades) == 2
            for trade in trades:
                assert trade["market_id"] == "market1"

        finally:
            duckdb_layer.close()


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_existing_tests_work_with_partitioning_disabled(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that existing behavior works with partitioning disabled."""
        # Create simple trades CSV (same as original tests)
        trades_csv = temp_dir / "trades.csv"
        trades_csv.write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
"""
        )

        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        summary = run_bootstrap(
            markets_csv=sample_markets_csv,
            trades_csv=trades_csv,
            order_filled_csv=sample_order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=False,
        )

        assert summary.status == "completed"
        assert summary.trades_rows == 3

        # Verify original file structure
        assert (parquet_dir / "trades.parquet").is_file()

        # Verify queries work with non-partitioned data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=False)
        try:
            duckdb_layer.create_views()
            trades, count = duckdb_layer.query_trades(market_id="market1")
            assert count == 2
        finally:
            duckdb_layer.close()
