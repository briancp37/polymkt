"""Tests for the days_to_exp derived field functionality."""

import tempfile
from collections.abc import Generator
from datetime import datetime, timezone
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
    """Create a sample markets CSV file with known closedTime values."""
    csv_path = temp_dir / "markets.csv"
    # Market1: closedTime = 2024-12-31 (366 days from 2024-01-01)
    # Market2: closedTime = 2024-06-30 (181 days from 2024-01-01)
    # Market3: No closedTime (null)
    csv_path.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category
2024-01-01 00:00:00,market1,Will it rain?,Yes,No,false,will-it-rain,token1a,token1b,cond1,1000.0,RAIN,2024-12-31 00:00:00,Weather prediction,Weather
2024-01-02 00:00:00,market2,Who wins?,A,B,true,who-wins,token2a,token2b,cond2,5000.0,WIN,2024-06-30 00:00:00,Election prediction,Politics
2024-01-03 00:00:00,market3,No expiry,Yes,No,false,no-expiry,token3a,token3b,cond3,100.0,NOEXP,,No expiry market,Other
"""
    )
    return csv_path


@pytest.fixture
def sample_trades_csv(temp_dir: Path) -> Path:
    """Create a sample trades CSV file with trades at known timestamps."""
    csv_path = temp_dir / "trades.csv"
    # Trade at 90 days to expiry for market1:
    #   closedTime = 2024-12-31, trade at 2024-10-02 => ~90 days
    # Trade at 180 days to expiry for market1:
    #   closedTime = 2024-12-31, trade at 2024-07-04 => ~180 days
    # Trade at 30 days to expiry for market2:
    #   closedTime = 2024-06-30, trade at 2024-05-31 => 30 days
    csv_path.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-10-02 00:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-07-04 00:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-05-31 00:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-15 00:00:00,market2,0xmaker4,0xtaker4,YES,buy,sell,0.75,150.0,200.0,0xhash4
2024-06-15 00:00:00,market3,0xmaker5,0xtaker5,NO,sell,buy,0.50,100.0,200.0,0xhash5
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


class TestDaysToExpCorrectness:
    """Tests verifying days_to_exp is computed correctly."""

    def test_days_to_exp_formula(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test that days_to_exp = (closedTime - trade_timestamp) in days."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query trade at 2024-10-02 for market1 (closedTime=2024-12-31)
            trades, _ = duckdb_layer.query_trades_with_markets(
                market_id="market1",
                start_time="2024-10-01 00:00:00",
                end_time="2024-10-03 00:00:00",
            )

            assert len(trades) == 1
            trade = trades[0]

            # Expected: 2024-12-31 00:00:00 - 2024-10-02 00:00:00 = 90 days
            assert trade["days_to_exp"] is not None
            assert 89 <= trade["days_to_exp"] <= 91  # Allow for rounding

        finally:
            duckdb_layer.close()

    def test_days_to_exp_180_days(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test days_to_exp for a trade ~180 days before expiry."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query trade at 2024-07-04 for market1 (closedTime=2024-12-31)
            trades, _ = duckdb_layer.query_trades_with_markets(
                market_id="market1",
                start_time="2024-07-03 00:00:00",
                end_time="2024-07-05 00:00:00",
            )

            assert len(trades) == 1
            trade = trades[0]

            # Expected: 2024-12-31 00:00:00 - 2024-07-04 00:00:00 = 180 days
            assert trade["days_to_exp"] is not None
            assert 179 <= trade["days_to_exp"] <= 181

        finally:
            duckdb_layer.close()

    def test_days_to_exp_30_days(self, setup_test_data: tuple[Path, Path]) -> None:
        """Test days_to_exp for a trade ~30 days before expiry."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query trade at 2024-05-31 for market2 (closedTime=2024-06-30)
            trades, _ = duckdb_layer.query_trades_with_markets(
                market_id="market2",
                start_time="2024-05-30 00:00:00",
                end_time="2024-06-01 00:00:00",
            )

            assert len(trades) == 1
            trade = trades[0]

            # Expected: 2024-06-30 00:00:00 - 2024-05-31 00:00:00 = 30 days
            assert trade["days_to_exp"] is not None
            assert 29 <= trade["days_to_exp"] <= 31

        finally:
            duckdb_layer.close()

    def test_days_to_exp_null_when_no_closed_time(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that days_to_exp is NULL when market has no closedTime."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query trade for market3 (no closedTime)
            trades, _ = duckdb_layer.query_trades_with_markets(market_id="market3")

            assert len(trades) == 1
            trade = trades[0]
            assert trade["days_to_exp"] is None

        finally:
            duckdb_layer.close()

    def test_days_to_exp_includes_market_columns(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that query_trades_with_markets includes market columns."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            trades, _ = duckdb_layer.query_trades_with_markets(market_id="market1")

            assert len(trades) == 2
            trade = trades[0]

            # Should include trade columns
            assert "timestamp" in trade
            assert "market_id" in trade
            assert "price" in trade
            assert "transaction_hash" in trade

            # Should include market columns
            assert "question" in trade
            assert "category" in trade
            assert "closed_time" in trade

            # Should include derived field
            assert "days_to_exp" in trade

        finally:
            duckdb_layer.close()


class TestDaysToExpFiltering:
    """Tests for filtering by days_to_exp range."""

    def test_filter_by_days_to_exp_min(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by minimum days_to_exp."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Get trades with days_to_exp >= 100
            trades, total_count = duckdb_layer.query_trades_with_markets(
                days_to_exp_min=100
            )

            # Should get trades with ~180 days (market1) and ~166 days (market2)
            assert total_count >= 1
            for trade in trades:
                if trade["days_to_exp"] is not None:
                    assert trade["days_to_exp"] >= 100

        finally:
            duckdb_layer.close()

    def test_filter_by_days_to_exp_max(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by maximum days_to_exp."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Get trades with days_to_exp <= 50
            trades, total_count = duckdb_layer.query_trades_with_markets(
                days_to_exp_max=50
            )

            # Should include the 30-day trade for market2
            assert total_count >= 1
            for trade in trades:
                if trade["days_to_exp"] is not None:
                    assert trade["days_to_exp"] <= 50

        finally:
            duckdb_layer.close()

    def test_filter_by_days_to_exp_range_89_to_91(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by days_to_exp range (89-91) - PRD requirement."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # PRD: Verify days_to_exp can be filtered efficiently (e.g., between 89 and 91)
            trades, total_count = duckdb_layer.query_trades_with_markets(
                days_to_exp_min=89,
                days_to_exp_max=91,
            )

            # Should get the ~90 day trade for market1
            assert total_count >= 1
            for trade in trades:
                assert trade["days_to_exp"] is not None
                assert 89 <= trade["days_to_exp"] <= 91

        finally:
            duckdb_layer.close()

    def test_filter_days_to_exp_combined_with_market_id(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by days_to_exp and market_id together."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            trades, total_count = duckdb_layer.query_trades_with_markets(
                market_id="market1",
                days_to_exp_min=89,
                days_to_exp_max=91,
            )

            assert total_count == 1
            assert len(trades) == 1
            assert trades[0]["market_id"] == "market1"
            assert 89 <= trades[0]["days_to_exp"] <= 91

        finally:
            duckdb_layer.close()

    def test_filter_days_to_exp_combined_with_time_range(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by days_to_exp and time range together."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            trades, total_count = duckdb_layer.query_trades_with_markets(
                start_time="2024-01-01 00:00:00",
                end_time="2024-12-31 23:59:59",
                days_to_exp_min=25,
                days_to_exp_max=35,
            )

            # Should get the ~30 day trade
            for trade in trades:
                assert trade["days_to_exp"] is not None
                assert 25 <= trade["days_to_exp"] <= 35

        finally:
            duckdb_layer.close()

    def test_filter_excludes_null_days_to_exp(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that filtering by days_to_exp excludes NULL values."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Filter should not include market3 (NULL days_to_exp)
            trades, _ = duckdb_layer.query_trades_with_markets(
                days_to_exp_min=0,
                days_to_exp_max=1000,
            )

            for trade in trades:
                assert trade["market_id"] != "market3"
                assert trade["days_to_exp"] is not None

        finally:
            duckdb_layer.close()


class TestDaysToExpOrdering:
    """Tests for ordering by days_to_exp."""

    def test_order_by_days_to_exp_asc(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test ordering by days_to_exp ascending."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Get all trades ordered by days_to_exp ASC
            trades, _ = duckdb_layer.query_trades_with_markets(
                order_by="days_to_exp",
                order_dir="ASC",
            )

            # Extract non-null values
            values = [t["days_to_exp"] for t in trades if t["days_to_exp"] is not None]
            assert values == sorted(values)

        finally:
            duckdb_layer.close()

    def test_order_by_days_to_exp_desc(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test ordering by days_to_exp descending."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            trades, _ = duckdb_layer.query_trades_with_markets(
                order_by="days_to_exp",
                order_dir="DESC",
            )

            # NULLs will sort to the end in DESC order, check non-null values
            values = [t["days_to_exp"] for t in trades if t["days_to_exp"] is not None]
            assert values == sorted(values, reverse=True)

        finally:
            duckdb_layer.close()


class TestDaysToExpPagination:
    """Tests for pagination with days_to_exp filtering."""

    def test_pagination_with_days_to_exp_filter(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test pagination works correctly with days_to_exp filtering."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Get total matching days_to_exp range
            _, total_count = duckdb_layer.query_trades_with_markets(
                days_to_exp_min=0,
                days_to_exp_max=200,
            )

            # Get first page
            page1, total1 = duckdb_layer.query_trades_with_markets(
                days_to_exp_min=0,
                days_to_exp_max=200,
                limit=2,
                offset=0,
                order_by="days_to_exp",
            )

            assert total1 == total_count
            assert len(page1) <= 2

        finally:
            duckdb_layer.close()

    def test_total_count_with_days_to_exp_filter(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that total_count is correct with days_to_exp filtering."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query with tight range that should match 1 trade
            trades, total_count = duckdb_layer.query_trades_with_markets(
                days_to_exp_min=89,
                days_to_exp_max=91,
                limit=10,
            )

            # The 90-day trade should be the only match
            assert total_count == 1
            assert len(trades) == 1

        finally:
            duckdb_layer.close()


class TestDaysToExpValidation:
    """Tests for parameter validation in days_to_exp queries."""

    def test_invalid_order_by_column_raises_error(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that invalid order_by column raises ValueError."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            with pytest.raises(ValueError, match="Invalid order_by column"):
                duckdb_layer.query_trades_with_markets(order_by="invalid_column")
        finally:
            duckdb_layer.close()

    def test_extended_order_by_columns_available(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that extended order_by columns (question, category) work."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Order by question
            trades, _ = duckdb_layer.query_trades_with_markets(order_by="question")
            assert len(trades) > 0

            # Order by category
            trades, _ = duckdb_layer.query_trades_with_markets(order_by="category")
            assert len(trades) > 0

            # Order by closed_time
            trades, _ = duckdb_layer.query_trades_with_markets(order_by="closed_time")
            assert len(trades) > 0

        finally:
            duckdb_layer.close()
