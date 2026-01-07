"""Tests for incremental search index updates."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.parquet import EVENTS_SCHEMA, MARKETS_SCHEMA, ORDER_FILLED_SCHEMA, TRADES_SCHEMA
from polymkt.storage.search_index_updater import (
    SearchIndexUpdater,
    compute_market_content_hash,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_markets_data():
    """Create sample markets data."""
    return pa.table(
        {
            "id": ["market-1", "market-2", "market-3"],
            "question": [
                "Will Bitcoin reach $100k?",
                "Who will win the election?",
                "Will it rain tomorrow?",
            ],
            "created_at": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc),
            ],
            "answer1": ["Yes", "Candidate A", "Yes"],
            "answer2": ["No", "Candidate B", "No"],
            "neg_risk": [False, False, False],
            "market_slug": ["bitcoin-100k", "election-2024", "rain-tomorrow"],
            "token1": ["token1", "token1", "token1"],
            "token2": ["token2", "token2", "token2"],
            "condition_id": ["cond1", "cond2", "cond3"],
            "volume": [1000.0, 2000.0, 500.0],
            "ticker": ["BTC", "ELEC", "RAIN"],
            "closed_time": [
                datetime(2024, 12, 31, tzinfo=timezone.utc),
                datetime(2024, 11, 5, tzinfo=timezone.utc),
                datetime(2024, 1, 4, tzinfo=timezone.utc),
            ],
            "description": [
                "Bitcoin price prediction",
                "US Presidential election",
                "Weather forecast",
            ],
            "category": ["crypto", "politics", "weather"],
            "event_id": ["event-1", "event-2", "event-3"],
            "tags": [["crypto", "bitcoin"], ["politics", "election"], ["weather"]],
        },
        schema=MARKETS_SCHEMA,
    )


@pytest.fixture
def sample_events_data():
    """Create sample events data."""
    return pa.table(
        {
            "event_id": ["event-1", "event-2", "event-3"],
            "tags": [["crypto", "bitcoin"], ["politics", "election"], ["weather"]],
            "title": ["Bitcoin Events", "Election 2024", "Weather Events"],
            "description": ["Crypto events", "Political events", "Weather events"],
            "created_at": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, tzinfo=timezone.utc),
            ],
        },
        schema=EVENTS_SCHEMA,
    )


@pytest.fixture
def sample_trades_data():
    """Create sample trades data."""
    return pa.table(
        {
            "timestamp": [
                datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 11, 12, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 12, 12, 0, tzinfo=timezone.utc),
            ],
            "market_id": ["market-1", "market-2", "market-3"],
            "maker": ["0x1111111111111111111111111111111111111111"] * 3,
            "taker": ["0x2222222222222222222222222222222222222222"] * 3,
            "nonusdc_side": ["yes", "no", "yes"],
            "maker_direction": ["buy", "sell", "buy"],
            "taker_direction": ["sell", "buy", "sell"],
            "price": [0.55, 0.45, 0.6],
            "usd_amount": [100.0, 200.0, 150.0],
            "token_amount": [181.82, 444.44, 250.0],
            "transaction_hash": ["tx1", "tx2", "tx3"],
        },
        schema=TRADES_SCHEMA,
    )


@pytest.fixture
def sample_order_filled_data():
    """Create sample order filled data."""
    return pa.table(
        {
            "timestamp": [
                datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 11, 12, 0, tzinfo=timezone.utc),
            ],
            "maker": ["0x1111111111111111111111111111111111111111"] * 2,
            "maker_asset_id": ["asset1", "asset2"],
            "maker_amount_filled": [100.0, 200.0],
            "taker": ["0x2222222222222222222222222222222222222222"] * 2,
            "taker_asset_id": ["asset3", "asset4"],
            "taker_amount_filled": [100.0, 200.0],
            "transaction_hash": ["tx1", "tx2"],
        },
        schema=ORDER_FILLED_SCHEMA,
    )


@pytest.fixture
def setup_test_data(temp_dir, sample_markets_data, sample_events_data, sample_trades_data, sample_order_filled_data):
    """Set up test data in parquet files."""
    parquet_dir = temp_dir / "parquet"
    parquet_dir.mkdir(parents=True)

    # Write markets
    pq.write_table(sample_markets_data, parquet_dir / "markets.parquet", compression="zstd")

    # Write events
    pq.write_table(sample_events_data, parquet_dir / "events.parquet", compression="zstd")

    # Write trades
    pq.write_table(sample_trades_data, parquet_dir / "trades.parquet", compression="zstd")

    # Write order_filled
    pq.write_table(sample_order_filled_data, parquet_dir / "order_filled.parquet", compression="zstd")

    return parquet_dir


class TestComputeMarketContentHash:
    """Tests for compute_market_content_hash function."""

    def test_hash_with_all_fields(self):
        """Test hash computation with all fields present."""
        hash1 = compute_market_content_hash(
            question="Will Bitcoin hit $100k?",
            tags=["crypto", "bitcoin"],
            description="Bitcoin price prediction",
        )
        assert hash1
        assert len(hash1) == 32  # MD5 hex digest length

    def test_hash_deterministic(self):
        """Test that hash is deterministic for same input."""
        hash1 = compute_market_content_hash(
            question="Test question", tags=["tag1", "tag2"], description="Test description"
        )
        hash2 = compute_market_content_hash(
            question="Test question", tags=["tag1", "tag2"], description="Test description"
        )
        assert hash1 == hash2

    def test_hash_different_for_different_question(self):
        """Test that hash changes when question changes."""
        hash1 = compute_market_content_hash(
            question="Question A", tags=["tag"], description="Desc"
        )
        hash2 = compute_market_content_hash(
            question="Question B", tags=["tag"], description="Desc"
        )
        assert hash1 != hash2

    def test_hash_different_for_different_tags(self):
        """Test that hash changes when tags change."""
        hash1 = compute_market_content_hash(
            question="Question", tags=["tag1"], description="Desc"
        )
        hash2 = compute_market_content_hash(
            question="Question", tags=["tag2"], description="Desc"
        )
        assert hash1 != hash2

    def test_hash_different_for_different_description(self):
        """Test that hash changes when description changes."""
        hash1 = compute_market_content_hash(
            question="Question", tags=["tag"], description="Desc A"
        )
        hash2 = compute_market_content_hash(
            question="Question", tags=["tag"], description="Desc B"
        )
        assert hash1 != hash2

    def test_hash_with_none_fields(self):
        """Test hash with None fields."""
        hash1 = compute_market_content_hash(question="Question", tags=None, description=None)
        assert hash1
        assert len(hash1) == 32

    def test_hash_tags_order_independent(self):
        """Test that tag order is normalized (sorted)."""
        hash1 = compute_market_content_hash(
            question="Q", tags=["b", "a", "c"], description=None
        )
        hash2 = compute_market_content_hash(
            question="Q", tags=["a", "b", "c"], description=None
        )
        assert hash1 == hash2


class TestSearchIndexUpdater:
    """Tests for SearchIndexUpdater class."""

    def test_init_creates_hash_table(self, setup_test_data):
        """Test that initialization creates the hash table."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # Check hash table exists
            result = duckdb_layer.conn.execute(
                "SELECT COUNT(*) FROM market_content_hashes"
            ).fetchone()
            assert result is not None
            assert result[0] == 0  # Initially empty
        finally:
            duckdb_layer.close()

    def test_detect_new_markets(self, setup_test_data):
        """Test detection of new markets when hash table is empty."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)
            new_ids, changed_ids, deleted_ids = updater.detect_changed_markets()

            # All markets should be new since hash table is empty
            assert len(new_ids) == 3
            assert len(changed_ids) == 0
            assert len(deleted_ids) == 0
            assert "market-1" in new_ids
            assert "market-2" in new_ids
            assert "market-3" in new_ids
        finally:
            duckdb_layer.close()

    def test_detect_changed_markets(self, setup_test_data, sample_markets_data):
        """Test detection of changed markets after content update."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First, build the index to populate hash table
            updater.update_indices(force_rebuild=True)

            # Now modify a market's content in the parquet file
            modified_data = sample_markets_data.to_pandas()
            modified_data.loc[0, "question"] = "Will Bitcoin reach $200k?"
            modified_table = pa.Table.from_pandas(modified_data, schema=MARKETS_SCHEMA)
            pq.write_table(
                modified_table, setup_test_data / "markets.parquet", compression="zstd"
            )

            # Refresh DuckDB views
            duckdb_layer.create_views()

            # Detect changes
            new_ids, changed_ids, deleted_ids = updater.detect_changed_markets()

            assert len(new_ids) == 0
            assert len(changed_ids) == 1
            assert "market-1" in changed_ids
            assert len(deleted_ids) == 0
        finally:
            duckdb_layer.close()

    def test_update_indices_full_rebuild(self, setup_test_data):
        """Test full index rebuild."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)
            result = updater.update_indices(force_rebuild=True)

            assert result["mode"] == "full_rebuild"
            assert result["bm25_indexed"] == 3
            # semantic_indexed is 0 without OpenAI key
            assert result["semantic_indexed"] == 0

            # Verify hash table is populated
            hash_count = duckdb_layer.conn.execute(
                "SELECT COUNT(*) FROM market_content_hashes"
            ).fetchone()[0]
            assert hash_count == 3
        finally:
            duckdb_layer.close()

    def test_update_indices_incremental_no_changes(self, setup_test_data):
        """Test incremental update when no markets have changed."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First build
            updater.update_indices(force_rebuild=True)

            # Second update with no changes
            result = updater.update_indices(force_rebuild=False)

            assert result["mode"] == "incremental"
            assert result["new_markets"] == 0
            assert result["changed_markets"] == 0
            assert result["deleted_markets"] == 0
        finally:
            duckdb_layer.close()

    def test_update_indices_incremental_with_changes(self, setup_test_data, sample_markets_data):
        """Test incremental update when markets have changed."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First build
            updater.update_indices(force_rebuild=True)

            # Modify a market
            modified_data = sample_markets_data.to_pandas()
            modified_data.loc[1, "description"] = "Updated election description"
            modified_table = pa.Table.from_pandas(modified_data, schema=MARKETS_SCHEMA)
            pq.write_table(
                modified_table, setup_test_data / "markets.parquet", compression="zstd"
            )

            # Refresh views
            duckdb_layer.create_views()

            # Incremental update
            result = updater.update_indices(force_rebuild=False)

            assert result["mode"] == "incremental"
            assert result["changed_markets"] == 1
            assert result["bm25_updated"] == 1
        finally:
            duckdb_layer.close()

    def test_update_specific_markets(self, setup_test_data):
        """Test updating specific market IDs."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First build indices
            updater.update_indices(force_rebuild=True)

            # Update specific markets
            result = updater.update_specific_markets(["market-1", "market-2"])

            assert result["mode"] == "specific"
            assert result["markets_requested"] == 2
            assert result["bm25_updated"] == 2
        finally:
            duckdb_layer.close()

    def test_update_specific_markets_empty_list(self, setup_test_data):
        """Test updating with empty market list."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)
            result = updater.update_specific_markets([])

            assert result["mode"] == "specific"
            assert result["markets_requested"] == 0
            assert result["bm25_updated"] == 0
        finally:
            duckdb_layer.close()

    def test_get_stats(self, setup_test_data):
        """Test getting stats from updater."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # Initial stats (no data)
            stats = updater.get_stats()
            assert stats["total_hashes"] == 0
            assert stats["bm25_available"] is False

            # Build index
            updater.update_indices(force_rebuild=True)

            # Stats after build
            stats = updater.get_stats()
            assert stats["total_hashes"] == 3
            assert stats["bm25_available"] is True
            assert stats["bm25_markets_indexed"] == 3
        finally:
            duckdb_layer.close()


class TestSearchIndexUpdaterWithNewMarkets:
    """Tests for detecting new markets added during updates."""

    def test_detect_new_markets_after_add(self, setup_test_data, sample_markets_data):
        """Test detection of new markets added after initial build."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First build
            updater.update_indices(force_rebuild=True)

            # Add a new market
            new_market = pa.table(
                {
                    "id": ["market-4"],
                    "question": ["Will gold hit $3000?"],
                    "created_at": [datetime(2024, 2, 1, tzinfo=timezone.utc)],
                    "answer1": ["Yes"],
                    "answer2": ["No"],
                    "neg_risk": [False],
                    "market_slug": ["gold-3000"],
                    "token1": ["token1"],
                    "token2": ["token2"],
                    "condition_id": ["cond4"],
                    "volume": [3000.0],
                    "ticker": ["GOLD"],
                    "closed_time": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
                    "description": ["Gold price prediction"],
                    "category": ["commodities"],
                    "event_id": ["event-4"],
                    "tags": [["commodities", "gold"]],
                },
                schema=MARKETS_SCHEMA,
            )

            # Combine with existing markets
            combined = pa.concat_tables([sample_markets_data, new_market])
            pq.write_table(combined, setup_test_data / "markets.parquet", compression="zstd")

            # Refresh views
            duckdb_layer.create_views()

            # Detect changes
            new_ids, changed_ids, deleted_ids = updater.detect_changed_markets()

            assert len(new_ids) == 1
            assert "market-4" in new_ids
            assert len(changed_ids) == 0
        finally:
            duckdb_layer.close()

    def test_incremental_update_adds_new_market(self, setup_test_data, sample_markets_data):
        """Test that incremental update properly indexes new markets."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First build
            updater.update_indices(force_rebuild=True)

            # Verify search works for original markets
            results, _ = updater.hybrid_index.search("bitcoin")
            assert len(results) > 0

            # Add a new market
            new_market = pa.table(
                {
                    "id": ["market-5"],
                    "question": ["Will Ethereum merge succeed?"],
                    "created_at": [datetime(2024, 2, 1, tzinfo=timezone.utc)],
                    "answer1": ["Yes"],
                    "answer2": ["No"],
                    "neg_risk": [False],
                    "market_slug": ["eth-merge"],
                    "token1": ["token1"],
                    "token2": ["token2"],
                    "condition_id": ["cond5"],
                    "volume": [5000.0],
                    "ticker": ["ETH"],
                    "closed_time": [datetime(2024, 9, 15, tzinfo=timezone.utc)],
                    "description": ["Ethereum merge prediction"],
                    "category": ["crypto"],
                    "event_id": ["event-5"],
                    "tags": [["crypto", "ethereum"]],
                },
                schema=MARKETS_SCHEMA,
            )

            combined = pa.concat_tables([sample_markets_data, new_market])
            pq.write_table(combined, setup_test_data / "markets.parquet", compression="zstd")

            # Refresh views
            duckdb_layer.create_views()

            # Incremental update
            result = updater.update_indices(force_rebuild=False)
            assert result["new_markets"] == 1

            # Verify new market is searchable
            results, _ = updater.hybrid_index.search("ethereum")
            assert len(results) > 0
            assert any(r["id"] == "market-5" for r in results)
        finally:
            duckdb_layer.close()


class TestSearchIndexUpdaterAPI:
    """Tests for search index update API endpoints."""

    @pytest.fixture
    def client_with_data(self, setup_test_data, monkeypatch):
        """Create test client with data."""
        from polymkt import config

        # Override settings
        monkeypatch.setattr(config.settings, "parquet_dir", setup_test_data)
        monkeypatch.setattr(config.settings, "duckdb_path", setup_test_data / "test.duckdb")

        return TestClient(app)

    def test_api_update_search_indices_force_rebuild(self, client_with_data):
        """Test API endpoint for force rebuild."""
        response = client_with_data.post("/api/search-index/update?force_rebuild=true")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["mode"] == "full_rebuild"
        assert data["bm25_updated"] > 0 or data.get("bm25_indexed", 0) > 0

    def test_api_update_search_indices_incremental(self, client_with_data):
        """Test API endpoint for incremental update."""
        # First build
        client_with_data.post("/api/search-index/update?force_rebuild=true")

        # Incremental (no changes)
        response = client_with_data.post("/api/search-index/update?force_rebuild=false")
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "incremental"
        assert data["status"] == "no_changes"

    def test_api_get_search_index_stats(self, client_with_data):
        """Test API endpoint for getting stats."""
        # First build
        client_with_data.post("/api/search-index/update?force_rebuild=true")

        response = client_with_data.get("/api/search-index/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_hashes"] == 3
        assert data["bm25_available"] is True
        assert data["bm25_markets_indexed"] == 3

    def test_api_update_requires_bootstrap(self, monkeypatch):
        """Test that update endpoint requires bootstrap first."""
        import tempfile

        from polymkt import config

        with tempfile.TemporaryDirectory() as td:
            empty_dir = Path(td)
            monkeypatch.setattr(config.settings, "parquet_dir", empty_dir)

            client = TestClient(app)
            response = client.post("/api/search-index/update")

            assert response.status_code == 400
            assert "Run bootstrap first" in response.json()["detail"]


class TestIncrementalSearchUpdateWithEvents:
    """Tests for incremental search updates when events change."""

    def test_detect_change_from_tags_update(self, setup_test_data, sample_markets_data, sample_events_data):
        """Test that tag changes trigger market index update."""
        duckdb_layer = DuckDBLayer(setup_test_data / "test.duckdb", setup_test_data)
        try:
            duckdb_layer.create_views()

            updater = SearchIndexUpdater(conn=duckdb_layer.conn)

            # First build
            updater.update_indices(force_rebuild=True)

            # Update market tags (simulating event update that affects tags)
            modified_data = sample_markets_data.to_pandas()
            modified_data.at[0, "tags"] = ["crypto", "bitcoin", "new-tag"]
            modified_table = pa.Table.from_pandas(modified_data, schema=MARKETS_SCHEMA)
            pq.write_table(
                modified_table, setup_test_data / "markets.parquet", compression="zstd"
            )

            # Refresh views
            duckdb_layer.create_views()

            # Detect changes
            new_ids, changed_ids, deleted_ids = updater.detect_changed_markets()

            assert len(changed_ids) == 1
            assert "market-1" in changed_ids
        finally:
            duckdb_layer.close()
