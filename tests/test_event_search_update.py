"""Tests for event change detection in incremental search index updates."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.search_index_updater import (
    SearchIndexUpdater,
    compute_event_content_hash,
)


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
        '''eventId,tags,title,description,createdAt
evt_001,"[""politics"",""election""]",US Election,Election markets,2024-01-01 00:00:00
evt_002,"[""sports"",""basketball""]",NBA Finals,Basketball markets,2024-01-02 00:00:00
evt_003,"[""weather""]",Hurricane Season,Weather markets,2024-01-03 00:00:00
'''
    )
    return csv_path


@pytest.fixture
def sample_markets_csv(temp_dir: Path) -> Path:
    """Create a sample markets CSV file."""
    csv_path = temp_dir / "markets.csv"
    csv_path.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category,eventId
2024-01-01 00:00:00,market1,Will Biden win?,Yes,No,false,biden-2024,t1a,t1b,c1,10000.0,BID,2024-11-05 23:59:59,Election prediction,Politics,evt_001
2024-01-01 00:00:01,market2,Will Trump win?,Yes,No,false,trump-2024,t2a,t2b,c2,15000.0,TRP,2024-11-05 23:59:59,Election prediction,Politics,evt_001
2024-01-02 00:00:00,market3,Will Lakers win?,Yes,No,false,lakers-finals,t3a,t3b,c3,3000.0,LAK,2024-06-30 23:59:59,NBA prediction,Sports,evt_002
2024-01-03 00:00:00,market4,Will there be Cat5?,Yes,No,false,cat5-hurricane,t4a,t4b,c4,1000.0,HRR,2024-09-30 23:59:59,Weather prediction,Weather,evt_003
"""
    )
    return csv_path


@pytest.fixture
def sample_trades_csv(temp_dir: Path) -> Path:
    """Create a minimal trades CSV file."""
    csv_path = temp_dir / "trades.csv"
    csv_path.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
"""
    )
    return csv_path


@pytest.fixture
def sample_order_filled_csv(temp_dir: Path) -> Path:
    """Create a minimal order filled CSV file."""
    csv_path = temp_dir / "orderFilled.csv"
    csv_path.write_text(
        """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash1
"""
    )
    return csv_path


@pytest.fixture
def bootstrapped_db(
    temp_dir: Path,
    sample_events_csv: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> tuple[Path, Path]:
    """Bootstrap the database with sample data and return paths."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        events_csv=sample_events_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    return duckdb_path, parquet_dir


class TestComputeEventContentHash:
    """Tests for event content hash computation."""

    def test_hash_with_tags(self) -> None:
        """Test hash computation with tags."""
        hash1 = compute_event_content_hash(["politics", "election"])
        assert hash1 is not None
        assert len(hash1) == 32  # MD5 hex length

    def test_hash_deterministic(self) -> None:
        """Test that hash is deterministic for same tags."""
        hash1 = compute_event_content_hash(["politics", "election"])
        hash2 = compute_event_content_hash(["politics", "election"])
        assert hash1 == hash2

    def test_hash_different_for_different_tags(self) -> None:
        """Test that different tags produce different hashes."""
        hash1 = compute_event_content_hash(["politics", "election"])
        hash2 = compute_event_content_hash(["sports", "basketball"])
        assert hash1 != hash2

    def test_hash_tags_order_independent(self) -> None:
        """Test that tag order doesn't affect the hash (sorted internally)."""
        hash1 = compute_event_content_hash(["election", "politics"])
        hash2 = compute_event_content_hash(["politics", "election"])
        assert hash1 == hash2

    def test_hash_with_none_tags(self) -> None:
        """Test hash computation with None tags."""
        hash1 = compute_event_content_hash(None)
        assert hash1 is not None
        assert len(hash1) == 32

    def test_hash_with_empty_tags(self) -> None:
        """Test hash computation with empty tags list."""
        hash1 = compute_event_content_hash([])
        hash2 = compute_event_content_hash(None)
        assert hash1 == hash2  # Both should produce same "empty" hash


class TestEventChangeDetection:
    """Tests for detecting event tag changes."""

    def test_detect_new_events(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that new events are detected on first run."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First detection should find all events as new
            new_events, changed_events, deleted_events = updater.detect_changed_events()

            assert len(new_events) == 3  # evt_001, evt_002, evt_003
            assert len(changed_events) == 0
            assert len(deleted_events) == 0
        finally:
            duckdb_layer.close()

    def test_no_changes_after_initial_update(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that no changes are detected after initial index build."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First update - builds index and stores hashes
            updater.update_indices(force_rebuild=True)

            # Second detection should find no changes
            new_events, changed_events, deleted_events = updater.detect_changed_events()

            assert len(new_events) == 0
            assert len(changed_events) == 0
            assert len(deleted_events) == 0
        finally:
            duckdb_layer.close()

    def test_check_events_available_with_events(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that events availability check works when events exist."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            assert updater._check_events_available() is True
        finally:
            duckdb_layer.close()


class TestGetMarketsForEvents:
    """Tests for finding markets affected by event changes."""

    def test_get_markets_for_single_event(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test finding markets for a single event."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # evt_001 has market1 and market2
            markets = updater._get_markets_for_events(["evt_001"])

            assert len(markets) == 2
            assert set(markets) == {"market1", "market2"}
        finally:
            duckdb_layer.close()

    def test_get_markets_for_multiple_events(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test finding markets for multiple events."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # evt_001 has 2 markets, evt_002 has 1 market
            markets = updater._get_markets_for_events(["evt_001", "evt_002"])

            assert len(markets) == 3
            assert set(markets) == {"market1", "market2", "market3"}
        finally:
            duckdb_layer.close()

    def test_get_markets_for_empty_events(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that empty event list returns empty market list."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            markets = updater._get_markets_for_events([])

            assert markets == []
        finally:
            duckdb_layer.close()

    def test_get_markets_for_nonexistent_event(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that nonexistent event returns empty market list."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            markets = updater._get_markets_for_events(["nonexistent_event"])

            assert markets == []
        finally:
            duckdb_layer.close()


class TestUpdateIndicesWithEvents:
    """Tests for update_indices including event change detection."""

    def test_force_rebuild_stores_event_hashes(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that force rebuild stores event content hashes."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            result = updater.update_indices(force_rebuild=True)

            assert result["mode"] == "full_rebuild"

            # Verify event hashes were stored
            stats = updater.get_stats()
            assert stats["total_event_hashes"] == 3  # 3 events
        finally:
            duckdb_layer.close()

    def test_incremental_update_detects_event_changes(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that incremental update detects new events."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First run - all events are new
            result = updater.update_indices()

            assert result["new_events"] == 3
            assert result["changed_events"] == 0
            assert result["deleted_events"] == 0
        finally:
            duckdb_layer.close()

    def test_incremental_update_includes_event_affected_markets(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that markets affected by event changes are included in update."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First run - all markets are new, so event_affected_markets won't count them
            # (they're already in new_markets)
            result = updater.update_indices()

            # All 4 markets should be updated
            assert result["bm25_updated"] == 4
            # event_affected_markets counts only markets that changed due to events
            # but weren't already in new_markets or changed_markets
            # Since all markets are new, event_affected_markets should be 0
            assert result["event_affected_markets"] == 0
        finally:
            duckdb_layer.close()

    def test_no_changes_returns_correct_result(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that no changes returns appropriate result."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First run - builds index
            updater.update_indices(force_rebuild=True)

            # Second run - no changes
            result = updater.update_indices()

            assert result["mode"] == "incremental"
            assert result["new_markets"] == 0
            assert result["changed_markets"] == 0
            assert result["new_events"] == 0
            assert result["changed_events"] == 0
            assert result["event_affected_markets"] == 0
        finally:
            duckdb_layer.close()


class TestSearchIndexUpdaterStats:
    """Tests for get_stats including event hash stats."""

    def test_stats_includes_event_hashes(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that stats includes event hash information."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # Build index
            updater.update_indices(force_rebuild=True)

            stats = updater.get_stats()

            assert "total_event_hashes" in stats
            assert stats["total_event_hashes"] == 3
            assert "event_first_updated" in stats
            assert "event_last_updated" in stats
        finally:
            duckdb_layer.close()

    def test_stats_empty_when_no_index(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that stats shows zeros when no index has been built."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # Don't build index
            stats = updater.get_stats()

            assert stats["total_hashes"] == 0
            assert stats["total_event_hashes"] == 0
        finally:
            duckdb_layer.close()


class TestAPIEventSearchUpdate:
    """Tests for API endpoints with event change detection."""

    def test_api_update_returns_event_fields(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that API update endpoint returns event-related fields."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path

        try:
            client = TestClient(app)
            response = client.post("/api/search-index/update?force_rebuild=true")
            assert response.status_code == 200

            data = response.json()
            assert "new_events" in data
            assert "changed_events" in data
            assert "deleted_events" in data
            assert "event_affected_markets" in data
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_stats_returns_event_hash_fields(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that API stats endpoint returns event hash fields."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path

        try:
            client = TestClient(app)

            # Build index first
            client.post("/api/search-index/update?force_rebuild=true")

            # Get stats
            response = client.get("/api/search-index/stats")
            assert response.status_code == 200

            data = response.json()
            assert "total_event_hashes" in data
            assert data["total_event_hashes"] == 3
            assert "event_first_updated" in data
            assert "event_last_updated" in data
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path
