"""Tests for events bootstrap and schema validation with market tags derivation."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from polymkt.pipeline.bootstrap import (
    SchemaValidationError,
    _join_events_tags_to_markets,
    _read_events_csv,
    run_bootstrap,
    validate_schema_requirements,
)
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.parquet import EVENTS_SCHEMA, MARKETS_SCHEMA


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_events_csv(temp_dir: Path) -> Path:
    """Create a sample events CSV file with tags."""
    csv_path = temp_dir / "events.csv"
    csv_path.write_text(
        '''eventId,tags,title,description,createdAt
evt_001,"[""politics"",""election"",""2024""]",US Election 2024,Presidential election markets,2024-01-01 00:00:00
evt_002,"[""sports"",""basketball""]",NBA Finals 2024,Basketball championship markets,2024-01-02 00:00:00
evt_003,"[""weather""]",Hurricane Season,Weather prediction markets,2024-01-03 00:00:00
'''
    )
    return csv_path


@pytest.fixture
def sample_markets_csv_with_event_id(temp_dir: Path) -> Path:
    """Create a sample markets CSV file with event_id for joining."""
    csv_path = temp_dir / "markets.csv"
    csv_path.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category,eventId
2024-01-01 00:00:00,market1,Who will win presidency?,Biden,Trump,false,president-2024,token1a,token1b,cond1,10000.0,PRES,2024-11-05 23:59:59,Election prediction,Politics,evt_001
2024-01-02 00:00:00,market2,Senate control?,Dem,Rep,true,senate-control,token2a,token2b,cond2,5000.0,SEN,2024-11-05 23:59:59,Senate prediction,Politics,evt_001
2024-01-03 00:00:00,market3,Lakers win Finals?,Yes,No,false,lakers-finals,token3a,token3b,cond3,3000.0,LAK,2024-06-30 23:59:59,NBA prediction,Sports,evt_002
2024-01-04 00:00:00,market4,Cat 5 hurricane?,Yes,No,false,cat5-hurricane,token4a,token4b,cond4,1000.0,HURR,2024-09-30 23:59:59,Weather prediction,Weather,evt_003
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
2024-01-16 09:00:00,market3,0xmaker3,0xtaker3,YES,buy,sell,0.80,200.0,250.0,0xhash3
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


class TestReadEventsCSV:
    """Tests for reading events CSV with tags parsing."""

    def test_read_events_csv_parses_tags(self, sample_events_csv: Path) -> None:
        """Test that events CSV is read with tags parsed as list."""
        events_table = _read_events_csv(sample_events_csv)

        assert events_table.num_rows == 3
        assert "event_id" in events_table.column_names
        assert "tags" in events_table.column_names
        assert "title" in events_table.column_names

    def test_read_events_csv_tags_are_lists(self, sample_events_csv: Path) -> None:
        """Test that tags are parsed as lists of strings."""
        events_table = _read_events_csv(sample_events_csv)

        # Convert to Python for inspection
        rows = events_table.to_pylist()

        # First event should have 3 tags
        assert rows[0]["event_id"] == "evt_001"
        assert rows[0]["tags"] == ["politics", "election", "2024"]

        # Second event should have 2 tags
        assert rows[1]["event_id"] == "evt_002"
        assert rows[1]["tags"] == ["sports", "basketball"]

        # Third event should have 1 tag
        assert rows[2]["event_id"] == "evt_003"
        assert rows[2]["tags"] == ["weather"]

    def test_read_events_csv_empty_tags(self, temp_dir: Path) -> None:
        """Test that empty tags are handled as empty list."""
        csv_path = temp_dir / "events_empty_tags.csv"
        csv_path.write_text(
            """eventId,tags,title,description,createdAt
evt_empty,,Empty Event,No tags,2024-01-01 00:00:00
"""
        )

        events_table = _read_events_csv(csv_path)
        rows = events_table.to_pylist()

        assert rows[0]["tags"] == []


class TestJoinEventsToMarkets:
    """Tests for joining event tags to markets."""

    def test_join_events_tags_to_markets(
        self, sample_events_csv: Path, sample_markets_csv_with_event_id: Path
    ) -> None:
        """Test that event tags are properly joined to markets."""
        import pyarrow as pa

        from polymkt.pipeline.bootstrap import (
            MARKETS_COLUMN_MAPPING,
            _read_csv_with_schema,
        )

        events_table = _read_events_csv(sample_events_csv)
        markets_table = _read_csv_with_schema(
            sample_markets_csv_with_event_id, MARKETS_SCHEMA, MARKETS_COLUMN_MAPPING
        )

        # Join events to markets
        enriched_markets = _join_events_tags_to_markets(markets_table, events_table)

        assert enriched_markets.num_rows == 4
        assert "tags" in enriched_markets.column_names

        # Convert to Python for inspection
        rows = enriched_markets.to_pylist()

        # Market1 and Market2 should have evt_001 tags
        market1 = next(r for r in rows if r["id"] == "market1")
        assert market1["tags"] == ["politics", "election", "2024"]

        market2 = next(r for r in rows if r["id"] == "market2")
        assert market2["tags"] == ["politics", "election", "2024"]

        # Market3 should have evt_002 tags
        market3 = next(r for r in rows if r["id"] == "market3")
        assert market3["tags"] == ["sports", "basketball"]

        # Market4 should have evt_003 tags
        market4 = next(r for r in rows if r["id"] == "market4")
        assert market4["tags"] == ["weather"]

    def test_join_events_to_markets_no_match(self, temp_dir: Path) -> None:
        """Test that markets without matching events get empty tags."""
        # Create events without matching market event_id
        events_csv = temp_dir / "events.csv"
        events_csv.write_text(
            '''eventId,tags,title,description,createdAt
evt_other,"[""other""]",Other Event,Unrelated,2024-01-01 00:00:00
'''
        )

        markets_csv = temp_dir / "markets.csv"
        markets_csv.write_text(
            """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category,eventId
2024-01-01 00:00:00,market1,Test?,Yes,No,false,test,t1a,t1b,c1,100.0,TST,2024-12-31 23:59:59,Test,Test,evt_nonexistent
"""
        )

        from polymkt.pipeline.bootstrap import (
            MARKETS_COLUMN_MAPPING,
            _read_csv_with_schema,
        )

        events_table = _read_events_csv(events_csv)
        markets_table = _read_csv_with_schema(
            markets_csv, MARKETS_SCHEMA, MARKETS_COLUMN_MAPPING
        )

        enriched_markets = _join_events_tags_to_markets(markets_table, events_table)
        rows = enriched_markets.to_pylist()

        # Market should have empty tags since event_id doesn't match
        assert rows[0]["tags"] == []


class TestSchemaValidation:
    """Tests for schema validation with actionable errors."""

    def test_validate_schema_requirements_passes(self) -> None:
        """Test that validation passes with valid schema."""
        import pyarrow as pa

        markets_table = pa.table({
            "id": ["m1", "m2"],
            "question": ["Q1?", "Q2?"],
            "event_id": ["e1", "e2"],
        })

        events_table = pa.table({
            "event_id": ["e1", "e2"],
            "tags": [["tag1"], ["tag2"]],
        })

        # Should not raise
        validate_schema_requirements(
            markets_table=markets_table,
            events_table=events_table,
            require_events_for_tags=True,
        )

    def test_validate_schema_requires_market_id(self) -> None:
        """Test that validation fails if market.id is missing."""
        import pyarrow as pa

        # Missing 'id' column
        markets_table = pa.table({
            "question": ["Q1?"],
        })

        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema_requirements(markets_table=markets_table)

        # Verify error message mentions the missing field
        assert "id" in str(exc_info.value)
        assert "markets.id" in exc_info.value.missing_fields

    def test_validate_schema_requires_market_question(self) -> None:
        """Test that validation fails if market.question is missing."""
        import pyarrow as pa

        # Missing 'question' column
        markets_table = pa.table({
            "id": ["m1"],
        })

        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema_requirements(markets_table=markets_table)

        assert "question" in str(exc_info.value)

    def test_validate_schema_requires_event_id_for_tags_join(self) -> None:
        """Test that validation fails if event_id is missing when tags required."""
        import pyarrow as pa

        # Missing 'event_id' in markets
        markets_table = pa.table({
            "id": ["m1"],
            "question": ["Q1?"],
        })

        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema_requirements(
                markets_table=markets_table,
                require_events_for_tags=True,
            )

        assert "event_id" in str(exc_info.value)

    def test_validate_schema_requires_events_table_for_tags(self) -> None:
        """Test that validation fails if events table is required but missing."""
        import pyarrow as pa

        markets_table = pa.table({
            "id": ["m1"],
            "question": ["Q1?"],
            "event_id": ["e1"],
        })

        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema_requirements(
                markets_table=markets_table,
                events_table=None,
                require_events_for_tags=True,
            )

        assert "Events table is required" in str(exc_info.value)

    def test_validate_schema_requires_events_tags_column(self) -> None:
        """Test that validation fails if events.tags is missing."""
        import pyarrow as pa

        markets_table = pa.table({
            "id": ["m1"],
            "question": ["Q1?"],
            "event_id": ["e1"],
        })

        # Missing 'tags' column in events
        events_table = pa.table({
            "event_id": ["e1"],
        })

        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema_requirements(
                markets_table=markets_table,
                events_table=events_table,
                require_events_for_tags=True,
            )

        # Verify error mentions missing tags
        assert "tags" in str(exc_info.value)
        assert "events.tags" in exc_info.value.missing_fields


class TestBootstrapWithEvents:
    """Integration tests for bootstrap with events processing."""

    def test_bootstrap_with_events_creates_events_parquet(
        self,
        temp_dir: Path,
        sample_events_csv: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap creates events Parquet file."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        summary = run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=sample_events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        assert summary.status == "completed"
        assert summary.events_rows == 3
        assert (parquet_dir / "events.parquet").exists()

    def test_bootstrap_with_events_joins_tags_to_markets(
        self,
        temp_dir: Path,
        sample_events_csv: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap joins event tags to markets."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=sample_events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        # Read markets Parquet and verify tags
        markets_table = pq.read_table(parquet_dir / "markets.parquet")
        assert "tags" in markets_table.column_names

        rows = markets_table.to_pylist()
        market1 = next(r for r in rows if r["id"] == "market1")
        assert market1["tags"] == ["politics", "election", "2024"]

    def test_bootstrap_preserves_category_and_closed_time(
        self,
        temp_dir: Path,
        sample_events_csv: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap preserves category and closedTime on markets."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=sample_events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        markets_table = pq.read_table(parquet_dir / "markets.parquet")
        rows = markets_table.to_pylist()

        # Verify category is preserved
        market1 = next(r for r in rows if r["id"] == "market1")
        assert market1["category"] == "Politics"

        # Verify closed_time is preserved
        assert market1["closed_time"] is not None

    def test_bootstrap_sets_events_watermark(
        self,
        temp_dir: Path,
        sample_events_csv: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap sets watermark for events."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=sample_events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        metadata_store = MetadataStore(metadata_path)
        watermarks = metadata_store.get_all_watermarks()

        assert "events" in watermarks
        assert "bootstrap_completed" in watermarks["events"]

    def test_bootstrap_without_events_works(
        self,
        temp_dir: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap works without events CSV (backward compatible)."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        # Non-existent events CSV
        events_csv = temp_dir / "nonexistent_events.csv"

        summary = run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        assert summary.status == "completed"
        assert summary.events_rows == 0
        assert summary.markets_rows == 4

    def test_bootstrap_fails_when_events_required_but_missing(
        self,
        temp_dir: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that bootstrap fails when events are required but not provided."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        # Non-existent events CSV
        events_csv = temp_dir / "nonexistent_events.csv"

        with pytest.raises(SchemaValidationError) as exc_info:
            run_bootstrap(
                markets_csv=sample_markets_csv_with_event_id,
                trades_csv=sample_trades_csv,
                order_filled_csv=sample_order_filled_csv,
                events_csv=events_csv,
                parquet_dir=parquet_dir,
                duckdb_path=duckdb_path,
                metadata_db_path=metadata_path,
                require_events_for_tags=True,
            )

        assert "Events CSV not found" in str(exc_info.value)


class TestDuckDBEventsView:
    """Tests for DuckDB events view and markets with tags."""

    def test_duckdb_creates_events_view(
        self,
        temp_dir: Path,
        sample_events_csv: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that DuckDB creates v_events view."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=sample_events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            counts = duckdb_layer.verify_views()

            assert "v_events" in counts
            assert counts["v_events"] == 3
        finally:
            duckdb_layer.close()

    def test_duckdb_markets_view_includes_tags(
        self,
        temp_dir: Path,
        sample_events_csv: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that v_markets view includes tags column."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=sample_events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query markets with tags
            result = duckdb_layer.conn.execute(
                "SELECT id, tags FROM v_markets ORDER BY id"
            ).fetchall()

            assert len(result) == 4

            # Verify tags are present
            market1_tags = result[0][1]  # market1
            assert market1_tags == ["politics", "election", "2024"]
        finally:
            duckdb_layer.close()

    def test_duckdb_trades_with_markets_includes_tags(
        self,
        temp_dir: Path,
        sample_events_csv: Path,
        sample_markets_csv_with_event_id: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
    ) -> None:
        """Test that v_trades_with_markets view includes tags column."""
        parquet_dir = temp_dir / "parquet"
        duckdb_path = temp_dir / "test.duckdb"
        metadata_path = temp_dir / "metadata.db"

        run_bootstrap(
            markets_csv=sample_markets_csv_with_event_id,
            trades_csv=sample_trades_csv,
            order_filled_csv=sample_order_filled_csv,
            events_csv=sample_events_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()

            # Query trades with markets
            result = duckdb_layer.conn.execute(
                "SELECT market_id, tags FROM v_trades_with_markets WHERE market_id = 'market1'"
            ).fetchall()

            assert len(result) == 1
            assert result[0][1] == ["politics", "election", "2024"]
        finally:
            duckdb_layer.close()
