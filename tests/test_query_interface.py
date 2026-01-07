"""Tests for the query interface (filter, sort, pagination)."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer


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
    """Create a sample trades CSV file with multiple records for testing."""
    csv_path = temp_dir / "trades.csv"
    csv_path.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-15 12:00:00,market1,0xmaker3,0xtaker3,YES,buy,sell,0.70,75.0,107.14,0xhash3
2024-01-16 09:00:00,market2,0xmaker4,0xtaker4,YES,buy,sell,0.80,200.0,250.0,0xhash4
2024-01-16 10:00:00,market2,0xmaker5,0xtaker5,NO,sell,buy,0.20,30.0,150.0,0xhash5
2024-01-17 08:00:00,market1,0xmaker6,0xtaker6,YES,buy,sell,0.72,120.0,166.67,0xhash6
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
"""
    )
    return csv_path


@pytest.fixture
def setup_test_data(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> tuple[Path, Path]:
    """Set up test data by running bootstrap."""
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

    return duckdb_path, parquet_dir


class TestQueryFiltering:
    """Tests for query filtering by market_id and time range."""

    def test_filter_by_single_market_id(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by a single market_id."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(market_id="market1")

            assert total_count == 4
            assert len(trades) == 4
            assert all(t["market_id"] == "market1" for t in trades)
        finally:
            duckdb_layer.close()

    def test_filter_by_time_range(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test filtering by time range."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(
                start_time="2024-01-16 00:00:00",
                end_time="2024-01-16 23:59:59",
            )

            assert total_count == 2
            assert len(trades) == 2
            assert all(t["market_id"] == "market2" for t in trades)
        finally:
            duckdb_layer.close()

    def test_filter_by_market_id_and_time_range(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by both market_id and time range."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(
                market_id="market1",
                start_time="2024-01-15 00:00:00",
                end_time="2024-01-15 23:59:59",
            )

            assert total_count == 3
            assert len(trades) == 3
            assert all(t["market_id"] == "market1" for t in trades)
        finally:
            duckdb_layer.close()


class TestQueryOrdering:
    """Tests for query result ordering."""

    def test_default_order_by_timestamp_asc(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test default ordering by timestamp ascending."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, _ = duckdb_layer.query_trades()

            # Verify ascending order
            timestamps = [t["timestamp"] for t in trades]
            assert timestamps == sorted(timestamps)
        finally:
            duckdb_layer.close()

    def test_order_by_timestamp_desc(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test ordering by timestamp descending."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, _ = duckdb_layer.query_trades(order_dir="DESC")

            # Verify descending order
            timestamps = [t["timestamp"] for t in trades]
            assert timestamps == sorted(timestamps, reverse=True)
        finally:
            duckdb_layer.close()

    def test_order_by_price(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test ordering by price."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, _ = duckdb_layer.query_trades(order_by="price", order_dir="ASC")

            prices = [t["price"] for t in trades]
            assert prices == sorted(prices)
        finally:
            duckdb_layer.close()

    def test_stable_ordering_with_composite_key(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test stable ordering with composite key (timestamp,transaction_hash)."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, _ = duckdb_layer.query_trades(
                order_by="timestamp,transaction_hash", order_dir="ASC"
            )

            # Run the same query multiple times to verify deterministic ordering
            for _ in range(3):
                trades2, _ = duckdb_layer.query_trades(
                    order_by="timestamp,transaction_hash", order_dir="ASC"
                )
                hashes1 = [t["transaction_hash"] for t in trades]
                hashes2 = [t["transaction_hash"] for t in trades2]
                assert hashes1 == hashes2, "Ordering should be deterministic"
        finally:
            duckdb_layer.close()

    def test_invalid_order_by_column_raises_error(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that invalid order_by column raises ValueError."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            with pytest.raises(ValueError, match="Invalid order_by column"):
                duckdb_layer.query_trades(order_by="invalid_column")
        finally:
            duckdb_layer.close()

    def test_invalid_order_dir_raises_error(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that invalid order_dir raises ValueError."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            with pytest.raises(ValueError, match="Invalid order_dir"):
                duckdb_layer.query_trades(order_dir="INVALID")
        finally:
            duckdb_layer.close()


class TestQueryPagination:
    """Tests for query pagination."""

    def test_pagination_limit(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test pagination limit."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(limit=2)

            assert len(trades) == 2
            assert total_count == 6  # Total rows in test data
        finally:
            duckdb_layer.close()

    def test_pagination_offset(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test pagination offset."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Get first page
            page1, total1 = duckdb_layer.query_trades(limit=2, offset=0)
            # Get second page
            page2, total2 = duckdb_layer.query_trades(limit=2, offset=2)

            assert len(page1) == 2
            assert len(page2) == 2
            assert total1 == total2 == 6

            # Verify different records
            hashes1 = {t["transaction_hash"] for t in page1}
            hashes2 = {t["transaction_hash"] for t in page2}
            assert hashes1.isdisjoint(hashes2), "Pages should have different records"
        finally:
            duckdb_layer.close()

    def test_total_count_for_pagination(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that total_count is returned correctly for pagination UI."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query with filter and pagination
            trades, total_count = duckdb_layer.query_trades(
                market_id="market1", limit=2, offset=0
            )

            assert len(trades) == 2
            assert total_count == 4  # Total market1 trades
        finally:
            duckdb_layer.close()

    def test_offset_beyond_total_returns_empty(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that offset beyond total returns empty list."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(limit=10, offset=100)

            assert len(trades) == 0
            assert total_count == 6  # Total still returned
        finally:
            duckdb_layer.close()


class TestQueryIntegration:
    """Integration tests combining all query features."""

    def test_full_query_workflow(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test a full query workflow with filter, sort, and pagination."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query market1 trades, sorted by price descending, with pagination
            trades, total_count = duckdb_layer.query_trades(
                market_id="market1",
                start_time="2024-01-15 00:00:00",
                end_time="2024-01-17 23:59:59",
                order_by="price",
                order_dir="DESC",
                limit=2,
                offset=0,
            )

            assert len(trades) == 2
            assert total_count == 4
            # Verify descending price order
            assert trades[0]["price"] >= trades[1]["price"]
            # All from market1
            assert all(t["market_id"] == "market1" for t in trades)
        finally:
            duckdb_layer.close()
