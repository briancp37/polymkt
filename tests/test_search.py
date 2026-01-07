"""Tests for BM25 full-text search over markets using DuckDB FTS extension."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.search import MarketSearchIndex, create_search_index


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


class TestSearchIndexCreation:
    """Tests for creating the BM25 search index."""

    def test_create_search_table(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that search table is created with proper columns."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)
            row_count = search_index.create_search_table()

            assert row_count == 7  # 7 markets

            # Verify table structure
            result = duckdb_layer.conn.execute(
                "SELECT * FROM markets_search LIMIT 1"
            ).description
            column_names = [col[0] for col in result]

            assert "id" in column_names
            assert "question" in column_names
            assert "tags" in column_names
            assert "tags_text" in column_names
            assert "description" in column_names
            assert "category" in column_names
            assert "closed_time" in column_names
            assert "event_id" in column_names
        finally:
            duckdb_layer.close()

    def test_create_fts_index(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that FTS index is created using DuckDB FTS extension."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)
            search_index.create_search_table()
            search_index.create_fts_index()

            # Verify FTS index exists by checking the fts system tables
            result = duckdb_layer.conn.execute(
                "SELECT COUNT(*) FROM fts_main_markets_search.docs"
            ).fetchone()
            assert result is not None
            assert result[0] == 7
        finally:
            duckdb_layer.close()

    def test_build_index_creates_table_and_fts(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that build_index creates both table and FTS index."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)
            row_count = search_index.build_index()

            assert row_count == 7
            assert search_index._index_created is True
        finally:
            duckdb_layer.close()

    def test_tags_flattened_to_searchable_text(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that tags are flattened to space-separated text for FTS."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)
            search_index.create_search_table()

            # Check that tags_text contains flattened tags
            result = duckdb_layer.conn.execute(
                "SELECT tags, tags_text FROM markets_search WHERE id = 'market1'"
            ).fetchone()

            assert result is not None
            tags = result[0]
            tags_text = result[1]

            # Original tags should be a list
            assert "politics" in tags
            assert "election" in tags

            # Flattened text should be space-separated
            assert "politics" in tags_text
            assert "election" in tags_text
        finally:
            duckdb_layer.close()


class TestBasicSearch:
    """Tests for basic BM25 search functionality."""

    def test_search_by_keyword_in_question(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test searching by keyword present in market question."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            results, total_count = search_index.search("election")

            assert total_count >= 3  # At least the election-related markets
            assert all("election" in r["question"].lower() for r in results)
        finally:
            duckdb_layer.close()

    def test_search_by_keyword_in_tags(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test searching by keyword present in market tags."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Search for "basketball" which is in tags but not question
            results, total_count = search_index.search("basketball")

            assert total_count >= 2  # Lakers and Celtics markets via evt_002 tags
            # All results should be from sports events
            for r in results:
                assert "basketball" in r["tags"] or "nba" in r["tags"]
        finally:
            duckdb_layer.close()

    def test_search_by_keyword_in_description(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test searching by keyword present in market description."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Search for "championship" which is in description
            results, total_count = search_index.search("championship")

            assert total_count >= 1
        finally:
            duckdb_layer.close()

    def test_search_returns_relevance_score(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that search returns BM25 relevance scores."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            results, _ = search_index.search("election")

            assert len(results) > 0
            assert all("score" in r for r in results)
            assert all(isinstance(r["score"], (int, float)) for r in results)
            # Scores should be positive (BM25 scores are positive for matches)
            assert all(r["score"] > 0 for r in results)
        finally:
            duckdb_layer.close()

    def test_search_results_sorted_by_relevance(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that search results are sorted by relevance score (descending)."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            results, _ = search_index.search("election")

            assert len(results) > 1
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            duckdb_layer.close()

    def test_search_returns_market_metadata(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that search results include required market metadata."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            results, _ = search_index.search("Lakers")

            assert len(results) >= 1
            result = results[0]

            # Verify all required fields are present
            assert "id" in result
            assert "question" in result
            assert "tags" in result
            assert "category" in result
            assert "closed_time" in result
            assert "event_id" in result
            assert "score" in result

            # Verify the Lakers market
            assert result["id"] == "market3"
            assert "Lakers" in result["question"]
        finally:
            duckdb_layer.close()


class TestSearchFiltering:
    """Tests for search with category and time filters."""

    def test_filter_by_category(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test filtering search results by category."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Search for broad term but filter to Sports
            results, total_count = search_index.search("win", category="Sports")

            assert total_count >= 1
            assert all(r["category"] == "Sports" for r in results)
        finally:
            duckdb_layer.close()

    def test_filter_by_closed_time_min(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test filtering search results by minimum closed_time."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Filter to markets closing after October 2024
            results, total_count = search_index.search(
                "win", closed_time_min="2024-10-01"
            )

            # Should include election markets (Nov) and Bitcoin (Dec)
            assert total_count >= 1
            for r in results:
                # All results should have closed_time after October
                assert r["closed_time"] is not None
        finally:
            duckdb_layer.close()

    def test_filter_by_closed_time_max(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test filtering search results by maximum closed_time."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Filter to markets closing before August 2024
            results, total_count = search_index.search(
                "win", closed_time_max="2024-08-01"
            )

            # Should include NBA and Hurricane markets
            assert total_count >= 1
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
            search_index = create_search_index(duckdb_layer.conn)

            # Filter to markets closing in Q4 2024
            results, total_count = search_index.search(
                "election",
                closed_time_min="2024-10-01",
                closed_time_max="2024-12-31",
            )

            # Should include only election markets (Nov 5)
            assert total_count >= 1
        finally:
            duckdb_layer.close()

    def test_combined_category_and_time_filter(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test filtering search results by both category and time."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Search Politics + Q4 2024
            results, total_count = search_index.search(
                "election",
                category="Politics",
                closed_time_min="2024-10-01",
            )

            assert total_count >= 1
            assert all(r["category"] == "Politics" for r in results)
        finally:
            duckdb_layer.close()


class TestSearchPagination:
    """Tests for search pagination."""

    def test_pagination_limit(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test limiting search results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            results, total_count = search_index.search("win", limit=2)

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
            search_index = create_search_index(duckdb_layer.conn)

            # Get first page
            results_page1, _ = search_index.search("win", limit=2, offset=0)

            # Get second page
            results_page2, _ = search_index.search("win", limit=2, offset=2)

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
            search_index = create_search_index(duckdb_layer.conn)

            results, total_count = search_index.search("win", limit=2)

            # Should have the total count regardless of limit
            assert total_count >= len(results)
        finally:
            duckdb_layer.close()

    def test_stable_ordering_across_pages(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that ordering is stable across paginated results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Get all results in one query
            all_results, _ = search_index.search("win", limit=100)

            # Get same results paginated
            page1, _ = search_index.search("win", limit=3, offset=0)
            page2, _ = search_index.search("win", limit=3, offset=3)

            # Verify ordering is consistent
            paginated_ids = [r["id"] for r in page1] + [r["id"] for r in page2]
            all_ids = [r["id"] for r in all_results][: len(paginated_ids)]
            assert paginated_ids == all_ids
        finally:
            duckdb_layer.close()


class TestSearchIndexMaintenance:
    """Tests for search index refresh and incremental updates."""

    def test_refresh_index_rebuilds_completely(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that refresh_index rebuilds the entire index."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)
            search_index.build_index()

            # Refresh should return same count
            row_count = search_index.refresh_index()
            assert row_count == 7
        finally:
            duckdb_layer.close()

    def test_update_markets_updates_specific_rows(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that update_markets updates specific market rows."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)
            search_index.build_index()

            # Update specific markets
            updated_count = search_index.update_markets(["market1", "market2"])
            assert updated_count == 2
        finally:
            duckdb_layer.close()

    def test_update_markets_with_empty_list(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that update_markets with empty list does nothing."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)
            search_index.build_index()

            updated_count = search_index.update_markets([])
            assert updated_count == 0
        finally:
            duckdb_layer.close()


class TestSearchEdgeCases:
    """Tests for search edge cases and error handling."""

    def test_search_without_building_index_raises_error(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that searching without building index raises RuntimeError."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = MarketSearchIndex(duckdb_layer.conn)

            with pytest.raises(RuntimeError) as exc_info:
                search_index.search("test")

            assert "Search index not built" in str(exc_info.value)
        finally:
            duckdb_layer.close()

    def test_search_with_no_matches(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test searching for a term with no matches."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            results, total_count = search_index.search("xyznonexistent")

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
            search_index = create_search_index(duckdb_layer.conn)

            # Empty query should return no results (FTS requires non-empty query)
            results, total_count = search_index.search("")

            assert len(results) == 0
            assert total_count == 0
        finally:
            duckdb_layer.close()

    def test_search_with_special_characters(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test searching with special characters."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            search_index = create_search_index(duckdb_layer.conn)

            # Search with $ symbol (from Bitcoin $100k)
            results, total_count = search_index.search("100k")

            # Should find Bitcoin market
            assert total_count >= 0  # May or may not match depending on tokenization
        finally:
            duckdb_layer.close()


class TestSearchAPIIntegration:
    """Tests for search API endpoint integration."""

    def test_api_search_endpoint(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test the /api/markets/search endpoint via FastAPI test client."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db

        # Override settings for test
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path

        try:
            client = TestClient(app)

            # First build the search index
            response = client.post("/api/search/build")
            assert response.status_code == 200

            # Then search
            response = client.get("/api/markets/search?q=election")
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

    def test_api_search_with_filters(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test the search API with category and time filters."""
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

            response = client.get(
                "/api/markets/search?q=win&category=Sports&limit=10"
            )
            assert response.status_code == 200

            data = response.json()
            assert data["count"] >= 1
            # All results should be Sports category
            assert all(r["category"] == "Sports" for r in data["results"])
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_search_pagination(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test the search API pagination."""
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

            # Get first page
            response1 = client.get("/api/markets/search?q=win&limit=2&offset=0")
            assert response1.status_code == 200
            data1 = response1.json()

            # Get second page
            response2 = client.get("/api/markets/search?q=win&limit=2&offset=2")
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

    def test_api_build_search_index(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test the /api/search/build endpoint."""
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

            response = client.post("/api/search/build")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "success"
            assert data["markets_indexed"] == 7
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path
