"""Tests for hybrid search combining BM25 and semantic vector search."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.hybrid_search import HybridSearchIndex, create_hybrid_search_index


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_events_csv(temp_dir: Path) -> Path:
    """Create a sample events CSV file with tags for search testing."""
    csv_path = temp_dir / "events.csv"
    csv_path.write_text(
        '''eventId,tags,title,description,createdAt
evt_001,"[""politics"",""election"",""2024"",""president""]",US Election 2024,Presidential election markets,2024-01-01 00:00:00
evt_002,"[""sports"",""basketball"",""nba""]",NBA Finals 2024,Basketball championship markets,2024-01-02 00:00:00
evt_003,"[""weather"",""hurricane""]",Hurricane Season,Weather prediction markets,2024-01-03 00:00:00
evt_004,"[""politics"",""senate""]",Senate Elections,Senate race predictions,2024-01-04 00:00:00
evt_005,"[""crypto"",""bitcoin""]",Crypto Markets,Cryptocurrency predictions,2024-01-05 00:00:00
'''
    )
    return csv_path


@pytest.fixture
def sample_markets_csv(temp_dir: Path) -> Path:
    """Create a sample markets CSV file with questions for search testing."""
    csv_path = temp_dir / "markets.csv"
    csv_path.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category,eventId
2024-01-01 00:00:00,market1,Will Biden win the 2024 presidential election?,Yes,No,false,biden-2024,token1a,token1b,cond1,10000.0,PRES,2024-11-05 23:59:59,Betting on Biden victory in presidential race,Politics,evt_001
2024-01-01 00:00:01,market2,Will Trump win the 2024 presidential election?,Yes,No,false,trump-2024,token2a,token2b,cond2,15000.0,PRES2,2024-11-05 23:59:59,Betting on Trump victory in presidential race,Politics,evt_001
2024-01-02 00:00:00,market3,Will the Lakers win the NBA Finals?,Yes,No,false,lakers-finals,token3a,token3b,cond3,3000.0,LAK,2024-06-30 23:59:59,NBA championship prediction for Lakers,Sports,evt_002
2024-01-02 00:00:01,market4,Will the Celtics win the NBA Finals?,Yes,No,false,celtics-finals,token4a,token4b,cond4,2500.0,CEL,2024-06-30 23:59:59,NBA championship prediction for Celtics,Sports,evt_002
2024-01-03 00:00:00,market5,Will there be a Category 5 hurricane this season?,Yes,No,false,cat5-hurricane,token5a,token5b,cond5,1000.0,HURR,2024-09-30 23:59:59,Weather prediction for major hurricane,Weather,evt_003
2024-01-04 00:00:00,market6,Will Democrats control the Senate after 2024 election?,Yes,No,false,senate-dem,token6a,token6b,cond6,5000.0,SEN,2024-11-05 23:59:59,Senate control prediction,Politics,evt_004
2024-01-05 00:00:00,market7,Will Bitcoin reach $100k in 2024?,Yes,No,false,btc-100k,token7a,token7b,cond7,8000.0,BTC,2024-12-31 23:59:59,Bitcoin price prediction,Crypto,evt_005
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


class TestHybridSearchIndexCreation:
    """Tests for creating the hybrid search index."""

    def test_build_bm25_index(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that BM25 index can be built independently."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            count = hybrid_index.build_bm25_index()

            assert count == 7  # 7 markets
            assert hybrid_index._bm25_ready is True
        finally:
            duckdb_layer.close()

    def test_build_semantic_index_requires_api_key(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that semantic index requires OpenAI API key."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)

            with pytest.raises(ValueError) as exc_info:
                hybrid_index.build_semantic_index()

            assert "OpenAI API key is required" in str(exc_info.value)
        finally:
            duckdb_layer.close()

    def test_create_hybrid_search_index_bm25_only(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test creating hybrid index with only BM25 (no API key)."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = create_hybrid_search_index(duckdb_layer.conn)

            assert hybrid_index._bm25_ready is True
            assert hybrid_index._semantic_ready is False
        finally:
            duckdb_layer.close()


class TestHybridSearchBM25Only:
    """Tests for hybrid search when only BM25 is available."""

    def test_search_with_bm25_only(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test hybrid search falls back to BM25 when semantic not available."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search("election")

            assert total_count >= 1
            assert all("election" in r["question"].lower() for r in results)
            # Results should have bm25_score but no semantic_score
            assert all(r.get("bm25_score") is not None for r in results)
        finally:
            duckdb_layer.close()

    def test_search_returns_relevance_score(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that search returns combined relevance scores."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, _ = hybrid_index.search("election")

            assert len(results) > 0
            assert all("score" in r for r in results)
            assert all(isinstance(r["score"], (int, float)) for r in results)
        finally:
            duckdb_layer.close()

    def test_search_results_sorted_by_score(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that search results are sorted by score (descending)."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, _ = hybrid_index.search("election")

            assert len(results) > 1
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            duckdb_layer.close()


class TestHybridSearchFiltering:
    """Tests for hybrid search with category and time filters."""

    def test_filter_by_category(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test filtering search results by category."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search("win", category="Sports")

            assert total_count >= 1
            assert all(r["category"] == "Sports" for r in results)
        finally:
            duckdb_layer.close()

    def test_filter_by_closed_time_range(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test filtering search results by closed_time range."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search(
                "election",
                closed_time_min="2024-10-01",
                closed_time_max="2024-12-31",
            )

            # Should include only election markets (Nov 5) and Bitcoin (Dec 31)
            assert total_count >= 1
        finally:
            duckdb_layer.close()

    def test_combined_filters(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test filtering with both category and time."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search(
                "election",
                category="Politics",
                closed_time_min="2024-10-01",
            )

            assert total_count >= 1
            assert all(r["category"] == "Politics" for r in results)
        finally:
            duckdb_layer.close()


class TestHybridSearchPagination:
    """Tests for hybrid search pagination."""

    def test_pagination_limit(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test limiting search results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search("win", limit=2)

            assert len(results) == 2
            assert total_count > 2  # There are more matches
        finally:
            duckdb_layer.close()

    def test_pagination_offset(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test offsetting search results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            # Get first page
            results_page1, _ = hybrid_index.search("win", limit=2, offset=0)

            # Get second page
            results_page2, _ = hybrid_index.search("win", limit=2, offset=2)

            # Pages should be different
            page1_ids = {r["id"] for r in results_page1}
            page2_ids = {r["id"] for r in results_page2}
            assert page1_ids.isdisjoint(page2_ids)
        finally:
            duckdb_layer.close()

    def test_total_count_for_pagination_ui(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that total_count is returned for pagination UI."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search("win", limit=2)

            # Should have the total count regardless of limit
            assert total_count >= len(results)
        finally:
            duckdb_layer.close()


class TestRRFScoring:
    """Tests for Reciprocal Rank Fusion scoring."""

    def test_compute_rrf_score(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test RRF score computation."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn, rrf_k=60)

            # RRF(rank=1) = 1/(60+1) ≈ 0.0164
            score_rank_1 = hybrid_index._compute_rrf_score(1)
            assert abs(score_rank_1 - (1.0 / 61)) < 0.0001

            # RRF(rank=10) = 1/(60+10) ≈ 0.0143
            score_rank_10 = hybrid_index._compute_rrf_score(10)
            assert abs(score_rank_10 - (1.0 / 70)) < 0.0001

            # Higher rank should have lower score
            assert score_rank_1 > score_rank_10
        finally:
            duckdb_layer.close()

    def test_merge_results_combines_scores(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that merge_results combines BM25 and semantic scores."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn, rrf_k=60)

            # Create mock results
            bm25_results = [
                {"id": "market1", "question": "Q1", "score": 5.0},
                {"id": "market2", "question": "Q2", "score": 4.0},
            ]
            semantic_results = [
                {"id": "market2", "question": "Q2", "score": 0.95},
                {"id": "market3", "question": "Q3", "score": 0.90},
            ]

            merged = hybrid_index._merge_results(bm25_results, semantic_results)

            # market2 should be ranked higher (appears in both)
            market2 = next(r for r in merged if r["id"] == "market2")
            market1 = next(r for r in merged if r["id"] == "market1")

            # market2 has RRF from both sources, market1 only from BM25
            assert market2["score"] > market1["score"]

            # Check that original scores are preserved
            assert market2.get("bm25_score") == 4.0
            assert market2.get("semantic_score") == 0.95
        finally:
            duckdb_layer.close()

    def test_merge_results_deduplicates(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that merge_results deduplicates markets from both sources."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)

            bm25_results = [
                {"id": "market1", "question": "Q1", "score": 5.0},
                {"id": "market2", "question": "Q2", "score": 4.0},
            ]
            semantic_results = [
                {"id": "market1", "question": "Q1", "score": 0.95},
                {"id": "market2", "question": "Q2", "score": 0.90},
            ]

            merged = hybrid_index._merge_results(bm25_results, semantic_results)

            # Should have exactly 2 unique markets
            assert len(merged) == 2
            ids = {r["id"] for r in merged}
            assert ids == {"market1", "market2"}
        finally:
            duckdb_layer.close()


class TestHybridSearchEdgeCases:
    """Tests for hybrid search edge cases and error handling."""

    def test_search_without_building_index_raises_error(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that searching without building index raises RuntimeError."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)

            with pytest.raises(RuntimeError) as exc_info:
                hybrid_index.search("test")

            assert "No search indices available" in str(exc_info.value)
        finally:
            duckdb_layer.close()

    def test_search_with_no_matches(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test searching for a term with no matches."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search("xyznonexistent")

            assert len(results) == 0
            assert total_count == 0
        finally:
            duckdb_layer.close()

    def test_search_with_empty_query(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test searching with empty query string."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            results, total_count = hybrid_index.search("")

            assert len(results) == 0
            assert total_count == 0
        finally:
            duckdb_layer.close()


class TestHybridSearchIndexMaintenance:
    """Tests for hybrid search index refresh and incremental updates."""

    def test_refresh_index_rebuilds(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that refresh_index rebuilds the BM25 index."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            counts = hybrid_index.refresh_index()
            assert counts["bm25"] == 7
        finally:
            duckdb_layer.close()

    def test_update_markets_updates_bm25(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that update_markets updates specific markets in BM25."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            counts = hybrid_index.update_markets(["market1", "market2"])
            assert counts["bm25"] == 2
        finally:
            duckdb_layer.close()

    def test_get_index_stats(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test getting index statistics."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            hybrid_index = HybridSearchIndex(duckdb_layer.conn)
            hybrid_index.build_bm25_index()

            stats = hybrid_index.get_index_stats()

            assert stats["bm25_available"] is True
            assert stats["semantic_available"] is False
            assert stats["bm25_markets_indexed"] == 7
        finally:
            duckdb_layer.close()


class TestHybridSearchWithMockedSemantic:
    """Tests for hybrid search with mocked semantic search."""

    def test_hybrid_search_with_both_sources(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test hybrid search combining both BM25 and semantic results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()

            # Create hybrid index with mocked semantic search
            hybrid_index = HybridSearchIndex(
                duckdb_layer.conn,
                openai_api_key="fake-key",  # Will be mocked
            )
            hybrid_index.build_bm25_index()

            # Mock the semantic search method
            mock_semantic_results = [
                {
                    "id": "market1",
                    "question": "Will Biden win the 2024 presidential election?",
                    "tags": ["politics", "election"],
                    "category": "Politics",
                    "closed_time": None,
                    "event_id": "evt_001",
                    "score": 0.95,
                },
                {
                    "id": "market6",
                    "question": "Will Democrats control the Senate after 2024 election?",
                    "tags": ["politics", "senate"],
                    "category": "Politics",
                    "closed_time": None,
                    "event_id": "evt_004",
                    "score": 0.85,
                },
            ]

            # Patch the semantic index search
            with patch.object(
                hybrid_index.semantic_index,
                "search",
                return_value=(mock_semantic_results, 2),
            ):
                # Mark semantic as ready
                hybrid_index._semantic_ready = True

                results, total_count = hybrid_index.search("election")

                # Should have combined results
                assert total_count >= 2

                # Results should have both bm25 and semantic scores where applicable
                market1 = next((r for r in results if r["id"] == "market1"), None)
                if market1:
                    assert market1.get("bm25_score") is not None
                    assert market1.get("semantic_score") is not None
        finally:
            duckdb_layer.close()


class TestHybridSearchAPIIntegration:
    """Tests for hybrid search API endpoint integration."""

    def test_api_build_hybrid_index(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test the /api/hybrid-search/build endpoint."""
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

            response = client.post("/api/hybrid-search/build")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "success"
            assert data["bm25_markets_indexed"] == 7
            # Semantic not built without API key
            assert data["semantic_markets_indexed"] == 0
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_hybrid_search_endpoint(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test the /api/markets/hybrid-search endpoint."""
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

            # First build the index
            client.post("/api/hybrid-search/build")

            # Then search
            response = client.get("/api/markets/hybrid-search?q=election")
            assert response.status_code == 200

            data = response.json()
            assert "results" in data
            assert "count" in data
            assert "total_count" in data
            assert "has_more" in data

            assert data["count"] >= 1
            assert data["total_count"] >= 1
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_hybrid_search_with_filters(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test the hybrid search API with category filter."""
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
            client.post("/api/hybrid-search/build")

            response = client.get(
                "/api/markets/hybrid-search?q=win&category=Sports&limit=10"
            )
            assert response.status_code == 200

            data = response.json()
            assert data["count"] >= 1
            # All results should be Sports category
            assert all(r["category"] == "Sports" for r in data["results"])
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_hybrid_search_pagination(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test the hybrid search API pagination."""
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
            client.post("/api/hybrid-search/build")

            # Get first page
            response1 = client.get("/api/markets/hybrid-search?q=win&limit=2&offset=0")
            assert response1.status_code == 200
            data1 = response1.json()

            # Get second page
            response2 = client.get("/api/markets/hybrid-search?q=win&limit=2&offset=2")
            assert response2.status_code == 200
            data2 = response2.json()

            # Verify has_more is set correctly
            if data1["total_count"] > 2:
                assert data1["has_more"] is True

            # Pages should have different results
            page1_ids = {r["id"] for r in data1["results"]}
            page2_ids = {r["id"] for r in data2["results"]}
            assert page1_ids.isdisjoint(page2_ids)
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_get_hybrid_search_stats(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test the /api/hybrid-search/stats endpoint."""
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
            client.post("/api/hybrid-search/build")

            response = client.get("/api/hybrid-search/stats")
            assert response.status_code == 200

            data = response.json()
            assert data["bm25_available"] is True
            assert data["bm25_markets_indexed"] == 7
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path
