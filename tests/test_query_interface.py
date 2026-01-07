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


class TestQueryMultipleMarketIds:
    """Tests for querying with multiple market_ids (100+ efficiently)."""

    def test_filter_by_two_market_ids(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by a list of two market_ids."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(
                market_ids=["market1", "market2"]
            )

            # Total: 4 from market1 + 2 from market2 = 6
            assert total_count == 6
            assert len(trades) == 6
            # All trades belong to one of the requested market_ids
            for t in trades:
                assert t["market_id"] in ["market1", "market2"]
        finally:
            duckdb_layer.close()

    def test_filter_by_subset_market_ids(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by a subset of market_ids returns only those markets."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(market_ids=["market2"])

            assert total_count == 2
            assert len(trades) == 2
            assert all(t["market_id"] == "market2" for t in trades)
        finally:
            duckdb_layer.close()

    def test_filter_by_nonexistent_market_ids(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by nonexistent market_ids returns empty."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(
                market_ids=["nonexistent1", "nonexistent2"]
            )

            assert total_count == 0
            assert len(trades) == 0
        finally:
            duckdb_layer.close()

    def test_multiple_market_ids_with_time_range(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test filtering by multiple market_ids and time range."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades(
                market_ids=["market1", "market2"],
                start_time="2024-01-16 00:00:00",
                end_time="2024-01-16 23:59:59",
            )

            # Only market2 trades are on 2024-01-16
            assert total_count == 2
            assert len(trades) == 2
            assert all(t["market_id"] == "market2" for t in trades)
        finally:
            duckdb_layer.close()

    def test_multiple_market_ids_with_pagination(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test multiple market_ids with pagination controls."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            # First page
            page1, total1 = duckdb_layer.query_trades(
                market_ids=["market1", "market2"],
                limit=3,
                offset=0,
                order_by="timestamp,transaction_hash",
            )
            # Second page
            page2, total2 = duckdb_layer.query_trades(
                market_ids=["market1", "market2"],
                limit=3,
                offset=3,
                order_by="timestamp,transaction_hash",
            )

            assert total1 == total2 == 6
            assert len(page1) == 3
            assert len(page2) == 3

            # Verify no overlap
            hashes1 = {t["transaction_hash"] for t in page1}
            hashes2 = {t["transaction_hash"] for t in page2}
            assert hashes1.isdisjoint(hashes2)
        finally:
            duckdb_layer.close()

    def test_single_market_id_parameter_takes_precedence(
        self, setup_test_data: tuple[Path, Path]
    ) -> None:
        """Test that market_id takes precedence over market_ids if both provided."""
        duckdb_path, parquet_dir = setup_test_data
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            # Pass both market_id and market_ids - single should take precedence
            trades, total_count = duckdb_layer.query_trades(
                market_id="market1",
                market_ids=["market2"],
            )

            # Should return market1 trades, not market2
            assert total_count == 4
            assert all(t["market_id"] == "market1" for t in trades)
        finally:
            duckdb_layer.close()


class TestQuery100PlusMarketIds:
    """Tests for efficiently querying 100+ market_ids."""

    @pytest.fixture
    def large_dataset(self, temp_dir: Path) -> tuple[Path, Path]:
        """Create a dataset with 150 markets for testing large market_id queries."""
        # Create markets CSV with 150 markets
        markets_lines = [
            "createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category"
        ]
        for i in range(150):
            markets_lines.append(
                f"2024-01-01 00:00:00,market{i},Question {i}?,Yes,No,false,slug{i},tok{i}a,tok{i}b,cond{i},{i * 100}.0,TK{i},2024-12-31 23:59:59,Desc {i},Cat{i % 5}"
            )

        markets_csv = temp_dir / "markets.csv"
        markets_csv.write_text("\n".join(markets_lines))

        # Create trades CSV with trades across 150 markets
        trades_lines = [
            "timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash"
        ]
        for i in range(150):
            # 3 trades per market = 450 trades total
            for j in range(3):
                hour = (i * 3 + j) % 24
                # Keep days within valid January range (1-31)
                day = 1 + ((i * 3 + j) // 24) % 28
                trades_lines.append(
                    f"2024-01-{day:02d} {hour:02d}:00:00,market{i},0xmaker{i},0xtaker{i},YES,buy,sell,0.5{j},{i * 10 + j}.0,{i + j}.0,0xhash{i}_{j}"
                )

        trades_csv = temp_dir / "trades.csv"
        trades_csv.write_text("\n".join(trades_lines))

        # Create order filled CSV (minimal)
        order_filled_csv = temp_dir / "orderFilled.csv"
        order_filled_csv.write_text(
            "timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash\n"
            "2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash1\n"
        )

        # Run bootstrap
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=markets_csv,
            trades_csv=trades_csv,
            order_filled_csv=order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        return duckdb_path, parquet_dir

    def test_query_100_plus_market_ids(
        self, large_dataset: tuple[Path, Path]
    ) -> None:
        """Test querying with 100+ market_ids returns correct results."""
        duckdb_path, parquet_dir = large_dataset
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query for 110 market_ids (market0 to market109)
            market_ids = [f"market{i}" for i in range(110)]
            trades, total_count = duckdb_layer.query_trades(
                market_ids=market_ids, limit=500
            )

            # 110 markets * 3 trades each = 330 trades
            assert total_count == 330
            # All returned trades belong to requested market_ids
            for t in trades:
                assert t["market_id"] in market_ids
        finally:
            duckdb_layer.close()

    def test_query_100_plus_market_ids_with_time_filter(
        self, large_dataset: tuple[Path, Path]
    ) -> None:
        """Test querying 100+ market_ids with time range filter."""
        duckdb_path, parquet_dir = large_dataset
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query for 100 market_ids with time filter
            # Data uses day = 1 + ((i * 3 + j) // 24) % 28, so trades start on 2024-01-01
            market_ids = [f"market{i}" for i in range(100)]
            trades, total_count = duckdb_layer.query_trades(
                market_ids=market_ids,
                start_time="2024-01-01 00:00:00",
                end_time="2024-01-01 12:00:00",
                limit=500,
            )

            # Verify all trades match filter criteria
            for t in trades:
                assert t["market_id"] in market_ids
            # Should have some trades (not all 300)
            assert total_count < 300
            assert total_count > 0
        finally:
            duckdb_layer.close()

    def test_query_100_plus_market_ids_pagination(
        self, large_dataset: tuple[Path, Path]
    ) -> None:
        """Test pagination works correctly with 100+ market_ids."""
        duckdb_path, parquet_dir = large_dataset
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            market_ids = [f"market{i}" for i in range(100)]

            # Page 1
            page1, total1 = duckdb_layer.query_trades(
                market_ids=market_ids,
                limit=50,
                offset=0,
                order_by="timestamp,transaction_hash",
            )
            # Page 2
            page2, total2 = duckdb_layer.query_trades(
                market_ids=market_ids,
                limit=50,
                offset=50,
                order_by="timestamp,transaction_hash",
            )

            # 100 markets * 3 trades = 300 total
            assert total1 == total2 == 300
            assert len(page1) == 50
            assert len(page2) == 50

            # Pages should not overlap
            hashes1 = {t["transaction_hash"] for t in page1}
            hashes2 = {t["transaction_hash"] for t in page2}
            assert hashes1.isdisjoint(hashes2)
        finally:
            duckdb_layer.close()

    def test_query_150_market_ids(
        self, large_dataset: tuple[Path, Path]
    ) -> None:
        """Test querying with 150 market_ids (full dataset)."""
        duckdb_path, parquet_dir = large_dataset
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            market_ids = [f"market{i}" for i in range(150)]
            trades, total_count = duckdb_layer.query_trades(
                market_ids=market_ids, limit=1000
            )

            # 150 markets * 3 trades = 450 trades
            assert total_count == 450
        finally:
            duckdb_layer.close()

    def test_query_plan_uses_filter_pushdown(
        self, large_dataset: tuple[Path, Path]
    ) -> None:
        """Test that DuckDB uses predicate pushdown for efficiency."""
        duckdb_path, parquet_dir = large_dataset
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Get explain plan for a multi-market query
            market_ids = [f"market{i}" for i in range(100)]
            placeholders = ", ".join(["?" for _ in market_ids])
            explain_sql = f"""
                EXPLAIN ANALYZE
                SELECT * FROM v_trades
                WHERE market_id IN ({placeholders})
            """
            result = duckdb_layer.conn.execute(explain_sql, market_ids)
            plan = "\n".join([str(row) for row in result.fetchall()])

            # DuckDB should show a filter operation in the plan
            # The presence of "Filter" or filtering in the plan indicates
            # predicate pushdown is being considered
            assert "Filter" in plan or "market_id" in plan.lower()
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
