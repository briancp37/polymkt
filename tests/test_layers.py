"""Tests for raw/analytics layer separation functionality."""

import hashlib
import tempfile
from collections.abc import Generator
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.pipeline.curate import run_curate
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


def _compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for comparison."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@pytest.fixture
def setup_raw_layer(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> tuple[Path, Path, Path, Path]:
    """Set up raw layer by running bootstrap."""
    raw_dir = temp_dir / "parquet" / "raw"
    analytics_dir = temp_dir / "parquet" / "analytics"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=raw_dir,  # Write to raw directory
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    return raw_dir, analytics_dir, duckdb_path, metadata_path


class TestRawLayerImmutability:
    """Tests verifying raw layer is unchanged after curate step."""

    def test_raw_layer_unchanged_after_curate(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that raw layer files are not modified by curate step."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Get hashes of raw layer files before curate
        raw_trades_path = raw_dir / "trades.parquet"
        raw_markets_path = raw_dir / "markets.parquet"
        raw_order_filled_path = raw_dir / "order_filled.parquet"

        trades_hash_before = _compute_file_hash(raw_trades_path)
        markets_hash_before = _compute_file_hash(raw_markets_path)
        order_filled_hash_before = _compute_file_hash(raw_order_filled_path)

        # Run curate step
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Verify raw layer files unchanged
        trades_hash_after = _compute_file_hash(raw_trades_path)
        markets_hash_after = _compute_file_hash(raw_markets_path)
        order_filled_hash_after = _compute_file_hash(raw_order_filled_path)

        assert trades_hash_before == trades_hash_after, "Raw trades modified"
        assert markets_hash_before == markets_hash_after, "Raw markets modified"
        assert order_filled_hash_before == order_filled_hash_after, "Raw order_filled modified"

    def test_raw_layer_row_count_preserved(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that raw layer row counts are preserved after curate."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Count rows before curate
        raw_trades = pq.read_table(raw_dir / "trades.parquet")
        raw_markets = pq.read_table(raw_dir / "markets.parquet")
        rows_before = {
            "trades": raw_trades.num_rows,
            "markets": raw_markets.num_rows,
        }

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Count rows after curate
        raw_trades_after = pq.read_table(raw_dir / "trades.parquet")
        raw_markets_after = pq.read_table(raw_dir / "markets.parquet")
        rows_after = {
            "trades": raw_trades_after.num_rows,
            "markets": raw_markets_after.num_rows,
        }

        assert rows_before == rows_after


class TestCurateStep:
    """Tests for the curate step."""

    def test_curate_creates_analytics_parquet(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that curate creates analytics Parquet files."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        summary = run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Verify analytics trades file created
        analytics_trades_path = analytics_dir / "trades_analytics.parquet"
        assert analytics_trades_path.exists(), "Analytics trades file not created"
        assert summary.status == "completed"
        assert summary.trades_rows_written > 0

    def test_curate_summary_has_correct_counts(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that curate summary reports correct row counts."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Get raw row count
        raw_trades = pq.read_table(raw_dir / "trades.parquet")
        expected_rows = raw_trades.num_rows

        # Run curate
        summary = run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        assert summary.trades_rows_read == expected_rows
        assert summary.trades_rows_written == expected_rows


class TestAnalyticsLayerDerivedFields:
    """Tests for derived fields in the analytics layer."""

    def test_analytics_layer_has_days_to_exp(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that analytics layer includes days_to_exp derived field."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Read analytics parquet
        analytics_trades = pq.read_table(analytics_dir / "trades_analytics.parquet")

        # Verify days_to_exp column exists
        assert "days_to_exp" in analytics_trades.column_names

    def test_analytics_days_to_exp_computed_correctly(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that days_to_exp is computed correctly in analytics layer."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Read analytics parquet
        analytics_trades = pq.read_table(analytics_dir / "trades_analytics.parquet")

        # Find the trade at 2024-10-02 for market1 (should have ~90 days to exp)
        df = analytics_trades.to_pandas()
        market1_trade = df[
            (df["market_id"] == "market1") &
            (df["transaction_hash"] == "0xhash1")
        ]

        assert len(market1_trade) == 1
        days_to_exp = market1_trade["days_to_exp"].values[0]
        assert days_to_exp is not None
        assert 89 <= days_to_exp <= 91  # ~90 days

    def test_analytics_days_to_exp_null_for_no_closed_time(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that days_to_exp is NULL when market has no closedTime."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Read analytics parquet
        analytics_trades = pq.read_table(analytics_dir / "trades_analytics.parquet")
        df = analytics_trades.to_pandas()

        # Find trade for market3 (no closedTime)
        market3_trade = df[df["market_id"] == "market3"]
        assert len(market3_trade) == 1
        assert market3_trade["days_to_exp"].isna().values[0]


class TestLayeredDuckDBViews:
    """Tests for DuckDB views in layered mode."""

    def test_layered_views_created(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that layered mode creates all required views."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate to create analytics layer
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Create DuckDB layer in layered mode
        duckdb_layer = DuckDBLayer(
            duckdb_path,
            parquet_dir=raw_dir,  # Not used in layered mode
            layered=True,
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
        )
        try:
            created_views = duckdb_layer.create_views()

            # Verify all expected views created
            expected_views = {
                "v_markets_raw",
                "v_trades_raw",
                "v_order_filled_raw",
                "v_trades_analytics",
                "v_trades_with_markets",
                "v_markets",
                "v_trades",
                "v_order_filled",
            }
            assert set(created_views) == expected_views
        finally:
            duckdb_layer.close()

    def test_raw_views_have_no_derived_fields(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that raw views don't include derived fields."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Create DuckDB layer in layered mode
        duckdb_layer = DuckDBLayer(
            duckdb_path,
            parquet_dir=raw_dir,
            layered=True,
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
        )
        try:
            duckdb_layer.create_views()

            # Query raw trades view
            result = duckdb_layer.execute("SELECT * FROM v_trades_raw LIMIT 1")
            columns = [desc[0] for desc in result.description]

            # days_to_exp should NOT be in raw view
            assert "days_to_exp" not in columns
        finally:
            duckdb_layer.close()

    def test_analytics_view_has_derived_fields(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that analytics view includes derived fields."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Create DuckDB layer in layered mode
        duckdb_layer = DuckDBLayer(
            duckdb_path,
            parquet_dir=raw_dir,
            layered=True,
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
        )
        try:
            duckdb_layer.create_views()

            # Query analytics trades view
            result = duckdb_layer.execute("SELECT * FROM v_trades_analytics LIMIT 1")
            columns = [desc[0] for desc in result.description]

            # days_to_exp SHOULD be in analytics view
            assert "days_to_exp" in columns
        finally:
            duckdb_layer.close()

    def test_both_layers_queryable_independently(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that both raw and analytics layers can be queried independently."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Create DuckDB layer in layered mode
        duckdb_layer = DuckDBLayer(
            duckdb_path,
            parquet_dir=raw_dir,
            layered=True,
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
        )
        try:
            duckdb_layer.create_views()
            counts = duckdb_layer.verify_views()

            # All views should have rows
            assert counts["v_trades_raw"] == 5
            assert counts["v_trades_analytics"] == 5
            assert counts["v_markets_raw"] == 3
            assert counts["v_trades_with_markets"] == 5
        finally:
            duckdb_layer.close()


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_legacy_mode_still_works(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that non-layered mode still works."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Create DuckDB layer in legacy mode (layered=False)
        duckdb_layer = DuckDBLayer(
            duckdb_path,
            parquet_dir=raw_dir,
            layered=False,
        )
        try:
            created_views = duckdb_layer.create_views()

            # Should have the standard 4 views
            expected_views = {"v_markets", "v_trades", "v_order_filled", "v_trades_with_markets"}
            assert set(created_views) == expected_views

            # Views should be queryable
            counts = duckdb_layer.verify_views()
            assert counts["v_trades"] == 5
            assert counts["v_markets"] == 3
        finally:
            duckdb_layer.close()

    def test_alias_views_work_in_layered_mode(
        self, setup_raw_layer: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that alias views (v_trades, v_markets) work in layered mode."""
        raw_dir, analytics_dir, duckdb_path, metadata_path = setup_raw_layer

        # Run curate
        run_curate(
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
            metadata_db_path=metadata_path,
        )

        # Create DuckDB layer in layered mode
        duckdb_layer = DuckDBLayer(
            duckdb_path,
            parquet_dir=raw_dir,
            layered=True,
            raw_dir=raw_dir,
            analytics_dir=analytics_dir,
        )
        try:
            duckdb_layer.create_views()

            # Query using alias views (should work like raw views)
            result = duckdb_layer.execute("SELECT COUNT(*) FROM v_trades").fetchone()
            assert result is not None and result[0] == 5

            result = duckdb_layer.execute("SELECT COUNT(*) FROM v_markets").fetchone()
            assert result is not None and result[0] == 3
        finally:
            duckdb_layer.close()
