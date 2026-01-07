"""Tests for poly_data integration feature.

This module tests the poly_data-like update logic including:
- Events update module (modeled after poly_data/poly_utils/update_markets)
- Events tags stored and joined onto markets deterministically
- Markets updates preserve category/closedTime and refresh tags via events join
- Outputs are Parquet (not CSV) and partitioned per configuration
- Pipeline is resumable after interrupted run
"""

import tempfile
from collections.abc import Generator
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
def sample_events_csv(temp_dir: Path) -> Path:
    """Create a sample events CSV file."""
    csv_path = temp_dir / "events.csv"
    csv_path.write_text(
        """eventId,tags,title,description,createdAt
event1,"[""politics"",""election""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
"""
    )
    return csv_path


@pytest.fixture
def sample_markets_with_events_csv(temp_dir: Path) -> Path:
    """Create a sample markets CSV file with event_id references."""
    csv_path = temp_dir / "markets.csv"
    csv_path.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category,eventId
2024-01-01 00:00:00,market1,Who wins president?,Trump,Biden,false,president,token1a,token1b,cond1,1000.0,PRES,2024-11-05 23:59:59,Presidential election market,Politics,event1
2024-01-02 00:00:00,market2,Super Bowl winner?,Team A,Team B,true,superbowl,token2a,token2b,cond2,5000.0,BOWL,2024-02-11 23:59:59,Super Bowl prediction,Sports,event2
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
2024-01-15 11:00:00,market2,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
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
def bootstrapped_with_events(
    temp_dir: Path,
    sample_events_csv: Path,
    sample_markets_with_events_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> dict[str, Path]:
    """Create a bootstrapped environment with events."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_with_events_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        events_csv=sample_events_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
        require_events_for_tags=True,
    )

    return {
        "temp_dir": temp_dir,
        "parquet_dir": parquet_dir,
        "duckdb_path": duckdb_path,
        "metadata_path": metadata_path,
        "markets_csv": sample_markets_with_events_csv,
        "trades_csv": sample_trades_csv,
        "order_filled_csv": sample_order_filled_csv,
        "events_csv": sample_events_csv,
    }


class TestEventsUpdateModule:
    """Tests for the events update module (PRD: poly_data integration step 2)."""

    def test_events_update_adds_new_events(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that events update adds new events."""
        env = bootstrapped_with_events

        # Add new event to CSV
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
event3,"[""crypto"",""bitcoin""]",Bitcoin ETF,Bitcoin ETF Approval,2024-01-10 00:00:00
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        assert summary.status == "completed"
        assert summary.rows_written["events"] == 1  # Only new event counted
        assert summary.rows_updated["events"] == 2  # 2 existing events updated

        # Verify total events in Parquet
        events_table = pq.read_table(env["parquet_dir"] / "events.parquet")
        assert events_table.num_rows == 3

    def test_events_update_updates_existing_events(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that events update updates existing event tags."""
        env = bootstrapped_with_events

        # Update event1 with new tags
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election"",""2024""]",Presidential Election,US Presidential Election 2024 Updated,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        assert summary.status == "completed"
        assert summary.rows_updated["events"] == 2

        # Verify event1 was updated
        events_table = pq.read_table(env["parquet_dir"] / "events.parquet")
        event1_rows = [
            row for row in events_table.to_pylist() if row["event_id"] == "event1"
        ]
        assert len(event1_rows) == 1
        assert "2024" in event1_rows[0]["tags"]

    def test_events_update_advances_watermark(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that events update advances watermarks."""
        env = bootstrapped_with_events

        # Add new event with later created_at
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
event3,"[""crypto""]",New Event,New event,2024-01-15 00:00:00
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        assert "events" in summary.watermark_after
        # Watermark should be at the latest event created_at
        assert "2024-01-15" in summary.watermark_after["events"]["last_timestamp"]


class TestEventTagsJoinDeterminism:
    """Tests for deterministic event tags joining (PRD: poly_data integration step 3)."""

    def test_events_tags_joined_to_markets_deterministically(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that event tags are joined to markets deterministically."""
        env = bootstrapped_with_events

        # Run the same update multiple times
        for _ in range(3):
            run_update(
                markets_csv=env["markets_csv"],
                trades_csv=env["trades_csv"],
                order_filled_csv=env["order_filled_csv"],
                events_csv=env["events_csv"],
                parquet_dir=env["parquet_dir"],
                duckdb_path=env["duckdb_path"],
                metadata_db_path=env["metadata_path"],
            )

        # Verify markets have tags from events
        markets_table = pq.read_table(env["parquet_dir"] / "markets.parquet")
        markets_dict = {row["id"]: row for row in markets_table.to_pylist()}

        # market1 should have event1 tags
        assert "politics" in markets_dict["market1"]["tags"]
        assert "election" in markets_dict["market1"]["tags"]

        # market2 should have event2 tags
        assert "sports" in markets_dict["market2"]["tags"]
        assert "football" in markets_dict["market2"]["tags"]


class TestMarketsPreserveCategoryClosedTimeRefreshTags:
    """Tests for market updates preserving category/closedTime and refreshing tags (PRD: poly_data integration step 4)."""

    def test_markets_updates_preserve_category_and_closed_time(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that markets updates preserve category and closedTime."""
        env = bootstrapped_with_events

        # Update event tags
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election"",""updated""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
"""
        )

        run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Verify category and closed_time are preserved
        markets_table = pq.read_table(env["parquet_dir"] / "markets.parquet")
        markets_dict = {row["id"]: row for row in markets_table.to_pylist()}

        assert markets_dict["market1"]["category"] == "Politics"
        assert markets_dict["market2"]["category"] == "Sports"

        # closed_time should be preserved
        assert markets_dict["market1"]["closed_time"] is not None
        assert markets_dict["market2"]["closed_time"] is not None

    def test_markets_tags_refreshed_via_events_join(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that markets.tags are refreshed when events are updated."""
        env = bootstrapped_with_events

        # Verify initial tags
        markets_table = pq.read_table(env["parquet_dir"] / "markets.parquet")
        market1_initial = [
            row for row in markets_table.to_pylist() if row["id"] == "market1"
        ][0]
        assert "updated" not in market1_initial["tags"]

        # Update event1 tags
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election"",""updated""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
"""
        )

        run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Verify market1 has refreshed tags
        markets_table = pq.read_table(env["parquet_dir"] / "markets.parquet")
        market1_updated = [
            row for row in markets_table.to_pylist() if row["id"] == "market1"
        ][0]
        assert "updated" in market1_updated["tags"]


class TestParquetOutputAndPartitioning:
    """Tests for Parquet output and partitioning (PRD: poly_data integration step 5)."""

    def test_outputs_are_parquet_not_csv(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that final outputs are Parquet (not CSV)."""
        env = bootstrapped_with_events

        # Run update
        run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Verify Parquet files exist
        assert (env["parquet_dir"] / "markets.parquet").exists()
        assert (env["parquet_dir"] / "trades.parquet").exists()
        assert (env["parquet_dir"] / "events.parquet").exists()

        # Verify they are valid Parquet files
        pq.read_table(env["parquet_dir"] / "markets.parquet")
        pq.read_table(env["parquet_dir"] / "trades.parquet")
        pq.read_table(env["parquet_dir"] / "events.parquet")

    def test_partitioned_output_when_enabled(self, temp_dir: Path) -> None:
        """Test that trades are partitioned when partitioning is enabled."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        # Create sample CSV files
        events_csv = temp_dir / "events.csv"
        events_csv.write_text(
            """eventId,tags,title,description,createdAt
event1,"[""test""]",Test Event,Test,2024-01-01 00:00:00
"""
        )

        markets_csv = temp_dir / "markets.csv"
        markets_csv.write_text(
            """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category,eventId
2024-01-01 00:00:00,market1,Test?,Yes,No,false,test,token1a,token1b,cond1,1000.0,TEST,2024-12-31 23:59:59,Test market,Test,event1
"""
        )

        trades_csv = temp_dir / "trades.csv"
        trades_csv.write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
"""
        )

        order_filled_csv = temp_dir / "orderFilled.csv"
        order_filled_csv.write_text(
            """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash1
"""
        )

        # Bootstrap with partitioning enabled
        run_bootstrap(
            markets_csv=markets_csv,
            trades_csv=trades_csv,
            order_filled_csv=order_filled_csv,
            events_csv=events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            partitioning_enabled=True,
        )

        # Verify trades directory exists (partitioned structure)
        trades_dir = parquet_dir / "trades"
        assert trades_dir.exists()
        assert trades_dir.is_dir()

        # Verify partition structure (year/month/day/hash_bucket)
        # Should have at least one parquet file in the partitioned structure
        parquet_files = list(trades_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0


class TestPipelineResumability:
    """Tests for pipeline resumability (PRD: poly_data integration step 6)."""

    def test_pipeline_is_resumable_after_interruption(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that pipeline is resumable after an interrupted run."""
        env = bootstrapped_with_events

        # Simulate first partial update
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
event3,"[""crypto""]",New Event 1,New event 1,2024-01-10 00:00:00
"""
        )

        summary1 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        events_after_first = pq.read_table(env["parquet_dir"] / "events.parquet")
        count_after_first = events_after_first.num_rows

        # Add more events and run again (simulating resume)
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
event3,"[""crypto""]",New Event 1,New event 1,2024-01-10 00:00:00
event4,"[""tech""]",New Event 2,New event 2,2024-01-15 00:00:00
"""
        )

        summary2 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        events_after_second = pq.read_table(env["parquet_dir"] / "events.parquet")
        count_after_second = events_after_second.num_rows

        # Should have added only the new event
        assert count_after_first == 3
        assert count_after_second == 4
        assert summary2.rows_written["events"] == 1  # Only event4 is new

    def test_no_duplicates_after_resume(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that resume does not create duplicate data."""
        env = bootstrapped_with_events

        # Run update multiple times with same data
        for _ in range(3):
            run_update(
                markets_csv=env["markets_csv"],
                trades_csv=env["trades_csv"],
                order_filled_csv=env["order_filled_csv"],
                events_csv=env["events_csv"],
                parquet_dir=env["parquet_dir"],
                duckdb_path=env["duckdb_path"],
                metadata_db_path=env["metadata_path"],
            )

        # Verify no duplicates
        events_table = pq.read_table(env["parquet_dir"] / "events.parquet")
        event_ids = events_table.column("event_id").to_pylist()
        assert len(event_ids) == len(set(event_ids)), "No duplicate event IDs"

        markets_table = pq.read_table(env["parquet_dir"] / "markets.parquet")
        market_ids = markets_table.column("id").to_pylist()
        assert len(market_ids) == len(set(market_ids)), "No duplicate market IDs"

        trades_table = pq.read_table(env["parquet_dir"] / "trades.parquet")
        tx_hashes = trades_table.column("transaction_hash").to_pylist()
        assert len(tx_hashes) == len(set(tx_hashes)), "No duplicate transaction hashes"


class TestDuckDBViewsWithEvents:
    """Tests for DuckDB views with events integration."""

    def test_duckdb_views_reflect_event_updates(
        self, bootstrapped_with_events: dict[str, Path]
    ) -> None:
        """Test that DuckDB views reflect event-derived market tags."""
        env = bootstrapped_with_events

        # Update event tags
        env["events_csv"].write_text(
            """eventId,tags,title,description,createdAt
event1,"[""politics"",""election"",""refreshed""]",Presidential Election,US Presidential Election 2024,2024-01-01 00:00:00
event2,"[""sports"",""football""]",Super Bowl,NFL Super Bowl 2024,2024-01-02 00:00:00
"""
        )

        run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Verify DuckDB views show updated tags
        duckdb_layer = DuckDBLayer(env["duckdb_path"], env["parquet_dir"])
        try:
            duckdb_layer.create_views()
            # Query markets view
            result = duckdb_layer.conn.execute(
                "SELECT id, tags FROM v_markets WHERE id = 'market1'"
            ).fetchone()
            assert result is not None
            assert "refreshed" in result[1]
        finally:
            duckdb_layer.close()
