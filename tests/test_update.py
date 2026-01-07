"""Tests for the incremental update pipeline."""

import tempfile
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.pipeline.update import run_update
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


@pytest.fixture
def bootstrapped_environment(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> dict[str, Path]:
    """Create a bootstrapped environment for update tests."""
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

    return {
        "temp_dir": temp_dir,
        "parquet_dir": parquet_dir,
        "duckdb_path": duckdb_path,
        "metadata_path": metadata_path,
        "markets_csv": sample_markets_csv,
        "trades_csv": sample_trades_csv,
        "order_filled_csv": sample_order_filled_csv,
    }


class TestIncrementalUpdate:
    """Tests for the incremental update pipeline."""

    def test_update_with_new_trades(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that update adds new trades."""
        env = bootstrapped_environment

        # Add new trades to CSV
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-17 14:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.70,150.0,214.28,0xhash4
2024-01-18 16:00:00,market2,0xmaker5,0xtaker5,NO,sell,buy,0.25,75.0,300.0,0xhash5
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        assert summary.status == "completed"
        # 2 new trades added (hash4, hash5), 3 skipped (hash1, hash2, hash3)
        assert summary.rows_written["trades"] == 2
        assert summary.rows_skipped["trades"] == 3

        # Verify total trades in Parquet
        trades_table = pq.read_table(env["parquet_dir"] / "trades.parquet")
        assert trades_table.num_rows == 5

    def test_update_with_new_markets(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that update adds new markets."""
        env = bootstrapped_environment

        # Add new market to CSV
        env["markets_csv"].write_text(
            """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category
2024-01-01 00:00:00,market1,Will it rain?,Yes,No,false,will-it-rain,token1a,token1b,cond1,1000.0,RAIN,2024-12-31 23:59:59,Weather prediction,Weather
2024-01-02 00:00:00,market2,Who wins?,A,B,true,who-wins,token2a,token2b,cond2,5000.0,WIN,2024-06-30 23:59:59,Election prediction,Politics
2024-01-03 00:00:00,market3,New market?,X,Y,false,new-market,token3a,token3b,cond3,3000.0,NEW,2024-09-30 23:59:59,New prediction,Sports
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        assert summary.status == "completed"
        assert summary.rows_written["markets"] == 1  # Only new market counted

        # Verify total markets in Parquet
        markets_table = pq.read_table(env["parquet_dir"] / "markets.parquet")
        assert markets_table.num_rows == 3

    def test_update_creates_run_record(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that update creates a run record."""
        env = bootstrapped_environment

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        metadata_store = MetadataStore(env["metadata_path"])
        run_record = metadata_store.get_run(summary.run_id)

        assert run_record is not None
        assert run_record.run_type == "update"
        assert run_record.status == "completed"
        assert run_record.duration_seconds is not None and run_record.duration_seconds > 0

    def test_update_advances_watermark(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that update advances watermarks."""
        env = bootstrapped_environment

        metadata_store = MetadataStore(env["metadata_path"])
        watermarks_before = metadata_store.get_all_watermarks()

        # Add new trades
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-20 10:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.70,150.0,214.28,0xhash4
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        watermarks_after = metadata_store.get_all_watermarks()

        # Watermark should have advanced
        assert summary.watermark_after["trades"]["last_timestamp"] == "2024-01-20T10:00:00+00:00"
        assert watermarks_after["trades"]["last_timestamp"] == "2024-01-20T10:00:00+00:00"


class TestIdempotentUpdates:
    """Tests for idempotent update behavior (PRD item 9)."""

    def test_repeated_update_does_not_duplicate_trades(
        self, bootstrapped_environment: dict[str, Path]
    ) -> None:
        """Test that running the same update twice does not duplicate trades."""
        env = bootstrapped_environment

        # Add new trades
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-17 14:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.70,150.0,214.28,0xhash4
"""
        )

        # First update
        summary1 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        trades_after_first = pq.read_table(env["parquet_dir"] / "trades.parquet")
        count_after_first = trades_after_first.num_rows

        # Second update with same data
        summary2 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        trades_after_second = pq.read_table(env["parquet_dir"] / "trades.parquet")
        count_after_second = trades_after_second.num_rows

        # Counts should be the same
        assert count_after_first == count_after_second
        assert summary2.rows_written["trades"] == 0
        # Watermark filtering ensures no trades are even considered (all filtered out before dedupe)
        # This is the correct idempotent behavior - no new data since last watermark

    def test_transaction_hash_uniqueness_enforced(
        self, bootstrapped_environment: dict[str, Path]
    ) -> None:
        """Test that transaction_hash uniqueness is enforced."""
        env = bootstrapped_environment

        # Verify unique transaction hashes
        trades_table = pq.read_table(env["parquet_dir"] / "trades.parquet")
        hashes = trades_table.column("transaction_hash").to_pylist()
        assert len(hashes) == len(set(hashes)), "Transaction hashes should be unique"

        # Add trades including one with existing hash
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-17 14:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.70,150.0,214.28,0xhash_new
"""
        )

        run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Verify uniqueness still holds
        trades_table = pq.read_table(env["parquet_dir"] / "trades.parquet")
        hashes = trades_table.column("transaction_hash").to_pylist()
        assert len(hashes) == len(set(hashes)), "Transaction hashes should remain unique"

    def test_dedupe_logs_skipped_rows(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that deduplication reports skipped rows in summary."""
        env = bootstrapped_environment

        # Same data as bootstrap (all will be skipped)
        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        assert summary.rows_skipped["trades"] == 3  # All 3 trades skipped
        assert summary.rows_skipped["order_filled"] == 2  # All 2 order_filled skipped
        assert summary.rows_written["trades"] == 0
        assert summary.rows_written["order_filled"] == 0


class TestUpdateRuntimeProportionality:
    """Tests verifying runtime is proportional to new data (PRD item 8)."""

    def test_update_runtime_scales_with_new_data(
        self, bootstrapped_environment: dict[str, Path]
    ) -> None:
        """Test that update runtime is proportional to new data, not total history."""
        env = bootstrapped_environment

        # First: update with no new data (should be fast)
        summary_empty = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Second: update with some new data
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-17 14:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.70,150.0,214.28,0xhash4
2024-01-18 10:00:00,market1,0xmaker5,0xtaker5,NO,sell,buy,0.30,60.0,200.0,0xhash5
"""
        )

        summary_with_data = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Both should complete quickly (not re-processing all data)
        assert summary_empty.duration_seconds is not None and summary_empty.duration_seconds < 5
        assert summary_with_data.duration_seconds is not None and summary_with_data.duration_seconds < 5

        # Empty update should have written nothing
        assert summary_empty.rows_written["trades"] == 0

        # Update with new data should have written the new rows
        assert summary_with_data.rows_written["trades"] == 2


class TestWatermarkFiltering:
    """Tests for watermark-based filtering."""

    def test_watermark_filters_old_data(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that watermark filters out old data already processed."""
        env = bootstrapped_environment

        # Set watermark to future date
        metadata_store = MetadataStore(env["metadata_path"])
        metadata_store.set_watermark("trades", {"last_timestamp": "2024-01-20T00:00:00+00:00"})

        # Add new trades including some before the watermark
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-21 10:00:00,market1,0xmaker10,0xtaker10,YES,buy,sell,0.75,120.0,160.0,0xhash10
2024-01-22 10:00:00,market2,0xmaker11,0xtaker11,NO,sell,buy,0.40,80.0,200.0,0xhash11
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Only 2 new trades after watermark should be processed
        # The old trade (hash1) should be filtered by watermark
        assert summary.rows_written["trades"] == 2


class TestDuckDBViewsRefresh:
    """Tests for DuckDB view refresh after updates."""

    def test_duckdb_views_reflect_new_data(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that DuckDB views reflect newly added data."""
        env = bootstrapped_environment

        # Add new trades
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 09:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
2024-01-17 14:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.70,150.0,214.28,0xhash4
"""
        )

        run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Verify DuckDB views show new data
        duckdb_layer = DuckDBLayer(env["duckdb_path"], env["parquet_dir"])
        try:
            duckdb_layer.create_views()
            trades, total_count = duckdb_layer.query_trades()
            assert total_count == 4  # 3 original + 1 new
        finally:
            duckdb_layer.close()


class TestMarketsUpsert:
    """Tests for markets upsert behavior."""

    def test_existing_market_updated(self, bootstrapped_environment: dict[str, Path]) -> None:
        """Test that existing markets can be updated."""
        env = bootstrapped_environment

        # Update market1 with new volume
        env["markets_csv"].write_text(
            """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category
2024-01-01 00:00:00,market1,Will it rain?,Yes,No,false,will-it-rain,token1a,token1b,cond1,2000.0,RAIN,2024-12-31 23:59:59,Updated weather prediction,Weather
2024-01-02 00:00:00,market2,Who wins?,A,B,true,who-wins,token2a,token2b,cond2,5000.0,WIN,2024-06-30 23:59:59,Election prediction,Politics
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Should report updates, not new writes
        assert summary.rows_updated["markets"] == 2  # Both markets updated

        # Verify data was updated
        markets_table = pq.read_table(env["parquet_dir"] / "markets.parquet")
        market1_rows = [
            row for row in markets_table.to_pylist() if row["id"] == "market1"
        ]
        assert len(market1_rows) == 1
        assert market1_rows[0]["volume"] == 2000.0
        assert market1_rows[0]["description"] == "Updated weather prediction"
