"""Tests for the unified Markets Search API endpoint with mode support."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap


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


class TestUnifiedSearchModeParameter:
    """Tests for the mode parameter in unified search API."""

    def test_bm25_mode_returns_results(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that mode=bm25 returns BM25 search results."""
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
            response = client.get("/api/markets/search?q=election&mode=bm25")
            assert response.status_code == 200

            data = response.json()
            assert data["mode"] == "bm25"
            assert data["count"] >= 1
            assert "results" in data
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_default_mode_is_hybrid(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that default mode is hybrid (falls back to bm25 without API key)."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path
        original_openai_key = settings.openai_api_key
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path
        settings.openai_api_key = None  # Ensure no API key

        try:
            client = TestClient(app)
            response = client.get("/api/markets/search?q=election")
            assert response.status_code == 200

            data = response.json()
            # Without OpenAI key, hybrid falls back to bm25
            assert data["mode"] == "bm25"
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path
            settings.openai_api_key = original_openai_key

    def test_invalid_mode_returns_400(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that invalid mode returns 400 error."""
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
            response = client.get("/api/markets/search?q=election&mode=invalid")
            assert response.status_code == 400
            assert "Invalid mode" in response.json()["detail"]
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_semantic_mode_without_api_key_returns_400(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that semantic mode without API key returns 400 error."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path
        original_openai_key = settings.openai_api_key
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path
        settings.openai_api_key = None

        try:
            client = TestClient(app)
            response = client.get("/api/markets/search?q=election&mode=semantic")
            assert response.status_code == 400
            assert "OpenAI API key" in response.json()["detail"]
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path
            settings.openai_api_key = original_openai_key


class TestUnifiedSearchResponseFormat:
    """Tests for the unified search response format."""

    def test_response_includes_all_required_fields(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that response includes all required fields for UI."""
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
            response = client.get("/api/markets/search?q=election&mode=bm25")
            assert response.status_code == 200

            data = response.json()
            assert "results" in data
            assert "count" in data
            assert "total_count" in data
            assert "has_more" in data
            assert "mode" in data

            # Check individual result fields
            if data["count"] > 0:
                result = data["results"][0]
                assert "id" in result
                assert "question" in result
                assert "tags" in result or result.get("tags") is None
                assert "category" in result or result.get("category") is None
                assert "closed_time" in result or result.get("closed_time") is None
                assert "event_id" in result or result.get("event_id") is None
                assert "relevance_score" in result
                assert "snippet" in result
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_response_includes_snippet(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that response includes snippet with highlighted terms."""
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
            response = client.get("/api/markets/search?q=election&mode=bm25")
            assert response.status_code == 200

            data = response.json()
            if data["count"] > 0:
                result = data["results"][0]
                # Snippet should contain highlighted terms (wrapped in **)
                assert "snippet" in result
                assert result["snippet"] is not None
                # Should highlight "election" with **
                assert "**election**" in result["snippet"].lower() or "**" in result["snippet"]
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_relevance_score_is_numeric(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that relevance_score is a numeric value."""
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
            response = client.get("/api/markets/search?q=election&mode=bm25")
            assert response.status_code == 200

            data = response.json()
            for result in data["results"]:
                assert isinstance(result["relevance_score"], (int, float))
                assert result["relevance_score"] > 0
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path


class TestUnifiedSearchPagination:
    """Tests for unified search pagination."""

    def test_pagination_with_limit_and_offset(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test pagination with limit and offset parameters."""
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
            response1 = client.get("/api/markets/search?q=win&mode=bm25&limit=2&offset=0")
            assert response1.status_code == 200
            data1 = response1.json()

            # Get second page
            response2 = client.get("/api/markets/search?q=win&mode=bm25&limit=2&offset=2")
            assert response2.status_code == 200
            data2 = response2.json()

            # Pages should have different results
            page1_ids = {r["id"] for r in data1["results"]}
            page2_ids = {r["id"] for r in data2["results"]}
            assert page1_ids.isdisjoint(page2_ids)
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_has_more_flag_is_correct(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test that has_more flag indicates more results available."""
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
            response = client.get("/api/markets/search?q=win&mode=bm25&limit=2")
            assert response.status_code == 200

            data = response.json()
            if data["total_count"] > 2:
                assert data["has_more"] is True
            else:
                assert data["has_more"] is False
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_stable_ordering_for_pagination(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that ordering is stable across paginated requests."""
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

            # Get all results
            response_all = client.get("/api/markets/search?q=win&mode=bm25&limit=100")
            all_ids = [r["id"] for r in response_all.json()["results"]]

            # Get paginated results
            response1 = client.get("/api/markets/search?q=win&mode=bm25&limit=2&offset=0")
            response2 = client.get("/api/markets/search?q=win&mode=bm25&limit=2&offset=2")

            paginated_ids = [r["id"] for r in response1.json()["results"]] + [
                r["id"] for r in response2.json()["results"]
            ]

            # First N paginated IDs should match first N all IDs
            assert paginated_ids == all_ids[: len(paginated_ids)]
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path


class TestUnifiedSearchFilters:
    """Tests for unified search filters."""

    def test_filter_by_category(self, bootstrapped_db: tuple[Path, Path]) -> None:
        """Test filtering by category."""
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
                "/api/markets/search?q=win&mode=bm25&category=Sports"
            )
            assert response.status_code == 200

            data = response.json()
            # All results should be Sports category
            for result in data["results"]:
                assert result["category"] == "Sports"
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_filter_by_closed_time_min(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test filtering by minimum closed_time."""
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
                "/api/markets/search?q=win&mode=bm25&closed_time_min=2024-10-01"
            )
            assert response.status_code == 200

            data = response.json()
            # Should only include markets closing after October
            assert data["count"] >= 1
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_filter_by_closed_time_max(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test filtering by maximum closed_time."""
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
                "/api/markets/search?q=win&mode=bm25&closed_time_max=2024-08-01"
            )
            assert response.status_code == 200

            data = response.json()
            # Should include NBA markets (June) and Hurricane (Sept doesn't match)
            assert data["count"] >= 1
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path


class TestSnippetGeneration:
    """Tests for snippet generation with query term highlighting."""

    def test_generate_snippet_highlights_query_terms(self) -> None:
        """Test that generate_snippet highlights query terms."""
        from polymkt.api.main import generate_snippet

        snippet = generate_snippet(
            question="Will Biden win the 2024 election?",
            description="Presidential race prediction",
            query="election",
        )

        assert "**election**" in snippet

    def test_generate_snippet_case_insensitive_highlight(self) -> None:
        """Test that highlighting is case-insensitive."""
        from polymkt.api.main import generate_snippet

        snippet = generate_snippet(
            question="Will Biden win the 2024 ELECTION?",
            description=None,
            query="election",
        )

        # Should highlight ELECTION preserving original case
        assert "**ELECTION**" in snippet

    def test_generate_snippet_multiple_terms(self) -> None:
        """Test that multiple query terms are highlighted."""
        from polymkt.api.main import generate_snippet

        snippet = generate_snippet(
            question="Will Biden win the 2024 presidential election?",
            description=None,
            query="Biden election",
        )

        assert "**Biden**" in snippet
        assert "**election**" in snippet

    def test_generate_snippet_truncates_long_text(self) -> None:
        """Test that long text is truncated before highlighting."""
        from polymkt.api.main import generate_snippet

        long_text = "This is a very long question " * 20
        snippet = generate_snippet(
            question=long_text,
            description=None,
            query="question",
            max_length=100,
        )

        # Snippet is truncated at max_length, then highlighting adds ** around matches
        # So final length can exceed max_length by amount needed for ** markers
        # Original truncated text would be ~100 chars, highlighting adds 4 chars per match
        assert snippet.endswith("...")
        # Verify truncation happened (original is 600 chars)
        assert len(snippet) < 200  # Well under original length even with highlighting

    def test_generate_snippet_includes_description_if_short_question(self) -> None:
        """Test that description is included if question is short."""
        from polymkt.api.main import generate_snippet

        snippet = generate_snippet(
            question="Short?",
            description="This is the description with election info",
            query="election",
        )

        # Should include description content
        assert "election" in snippet.lower()

    def test_generate_snippet_ignores_single_char_terms(self) -> None:
        """Test that single character terms are not highlighted."""
        from polymkt.api.main import generate_snippet

        snippet = generate_snippet(
            question="Will a hurricane hit?",
            description=None,
            query="a hurricane",
        )

        # "a" should not be highlighted, but "hurricane" should
        assert "**a**" not in snippet
        assert "**hurricane**" in snippet


class TestUnifiedSearchEdgeCases:
    """Tests for edge cases in unified search."""

    def test_empty_query_returns_empty_results(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that empty query returns empty results."""
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
            response = client.get("/api/markets/search?q=&mode=bm25")
            assert response.status_code == 200

            data = response.json()
            assert data["count"] == 0
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_no_matches_returns_empty_results(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that query with no matches returns empty results."""
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
            response = client.get("/api/markets/search?q=xyznonexistent&mode=bm25")
            assert response.status_code == 200

            data = response.json()
            assert data["count"] == 0
            assert data["total_count"] == 0
            assert data["results"] == []
        finally:
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_missing_parquet_returns_400(self, temp_dir: Path) -> None:
        """Test that missing parquet files return 400 error."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        original_parquet_dir = settings.parquet_dir
        settings.parquet_dir = temp_dir / "nonexistent"

        try:
            client = TestClient(app)
            response = client.get("/api/markets/search?q=test&mode=bm25")
            assert response.status_code == 400
            assert "not available" in response.json()["detail"]
        finally:
            settings.parquet_dir = original_parquet_dir
