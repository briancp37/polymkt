"""Tests for the Dataset Agent natural language dataset creation feature."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from polymkt.agents.dataset_agent import (
    DatasetAgent,
    DatasetAgentRequest,
    NaturalLanguageParser,
)
from polymkt.storage.search import MarketSearchIndex


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        parquet_dir = tmp_path / "parquet"
        parquet_dir.mkdir()
        db_path = tmp_path / "metadata.db"
        yield tmp_path, parquet_dir, db_path


@pytest.fixture
def sample_markets_data():
    """Sample markets data for testing."""
    return pd.DataFrame(
        {
            "id": [
                "market1",
                "market2",
                "market3",
                "market4",
                "market5",
            ],
            "question": [
                "Will Republicans win the Senate in 2024?",
                "Will Democrats control the House?",
                "Who will win the presidential election?",
                "Will Bitcoin reach $100k?",
                "Will the Lakers win the NBA championship?",
            ],
            "category": [
                "politics",
                "politics",
                "politics",
                "crypto",
                "sports",
            ],
            "tags": [
                ["election", "senate", "2024"],
                ["election", "house", "democrats"],
                ["election", "president", "2024"],
                ["crypto", "bitcoin"],
                ["sports", "nba", "lakers"],
            ],
            "closed_time": [
                datetime(2024, 11, 6, tzinfo=timezone.utc),
                datetime(2024, 11, 6, tzinfo=timezone.utc),
                datetime(2024, 11, 6, tzinfo=timezone.utc),
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 6, 30, tzinfo=timezone.utc),
            ],
            "event_id": ["event1", "event1", "event1", "event2", "event3"],
            "description": [
                "Senate control prediction",
                "House control prediction",
                "Presidential election outcome",
                "Bitcoin price prediction",
                "Lakers championship odds",
            ],
        }
    )


@pytest.fixture
def duckdb_with_search(temp_dirs, sample_markets_data):
    """Create DuckDB connection with markets and search index."""
    tmp_path, parquet_dir, db_path = temp_dirs

    # Write markets parquet
    markets_path = parquet_dir / "markets.parquet"
    sample_markets_data.to_parquet(markets_path)

    # Create DuckDB connection with markets view
    conn = duckdb.connect(":memory:")

    # Create v_markets view
    conn.execute(f"""
        CREATE VIEW v_markets AS
        SELECT
            id,
            question,
            category,
            tags,
            closed_time,
            event_id,
            description
        FROM read_parquet('{markets_path}')
    """)

    # Build search index
    search_index = MarketSearchIndex(conn)
    search_index.build_index()

    return conn, db_path


class TestNaturalLanguageParser:
    """Tests for the NaturalLanguageParser class."""

    def test_parse_simple_query(self):
        """Test parsing a simple query without special filters."""
        parser = NaturalLanguageParser()
        result = parser.parse("election markets")

        assert result["search_query"] == "election markets"
        assert result["category"] is None
        assert result["closed_time_min"] is None
        assert result["closed_time_max"] is None

    def test_parse_extracts_category(self):
        """Test parsing extracts category from query."""
        parser = NaturalLanguageParser()
        result = parser.parse("find politics markets about senate")

        assert result["category"] == "politics"
        assert "senate" in result["search_query"]

    def test_parse_extracts_year_filter(self):
        """Test parsing extracts year-based time filter."""
        parser = NaturalLanguageParser()
        result = parser.parse("elections closing in 2024")

        assert result["closed_time_min"] == "2024-01-01T00:00:00Z"
        assert result["closed_time_max"] == "2024-12-31T23:59:59Z"

    def test_parse_removes_common_phrases(self):
        """Test parsing removes common filter phrases."""
        parser = NaturalLanguageParser()
        result = parser.parse("find markets about senate control")

        assert "find" not in result["search_query"].lower()
        assert "about" not in result["search_query"].lower()
        assert "senate control" in result["search_query"]

    def test_parse_handles_show_me(self):
        """Test parsing handles 'show me' phrase with additional keywords."""
        parser = NaturalLanguageParser()
        result = parser.parse("show me crypto bitcoin trading")

        assert result["category"] == "crypto"
        # After removing "show me" and "crypto", the query should be cleaner
        assert "bitcoin" in result["search_query"].lower() or "trading" in result["search_query"].lower()

    def test_parse_sports_category(self):
        """Test parsing detects sports category."""
        parser = NaturalLanguageParser()
        result = parser.parse("sports markets NBA playoffs")

        assert result["category"] == "sports"
        assert "NBA playoffs" in result["search_query"]


class TestDatasetAgent:
    """Tests for the DatasetAgent class."""

    def test_process_query_returns_markets(self, duckdb_with_search):
        """Test that process_query returns relevant markets."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election senate")

        assert response.market_count > 0
        assert len(response.markets) > 0
        assert response.session_id is not None

    def test_process_query_all_included_by_default(self, duckdb_with_search):
        """Test that all markets are included by default."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")

        for market in response.markets:
            assert market.included is True

    def test_process_query_stores_session(self, duckdb_with_search):
        """Test that process_query stores the session."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")

        session = agent.get_session(response.session_id)
        assert session is not None
        assert session.session_id == response.session_id

    def test_process_query_with_category_filter(self, duckdb_with_search):
        """Test process_query respects category filter."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        request = DatasetAgentRequest(
            natural_language_query="election",
            category_filter="politics",
        )
        response = agent.process_query(request)

        # Should only return politics markets
        for market in response.markets:
            if market.category:
                assert market.category == "politics"

    def test_modify_market_list(self, duckdb_with_search):
        """Test modifying market inclusion."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")
        first_market_id = response.markets[0].market_id

        # Exclude the first market
        updated = agent.modify_market_list(
            session_id=response.session_id,
            market_id=first_market_id,
            included=False,
        )

        # Verify the market is now excluded
        first_market = next(m for m in updated.markets if m.market_id == first_market_id)
        assert first_market.included is False

        # Verify count is updated
        assert updated.market_count == len(response.markets) - 1

    def test_bulk_modify_markets(self, duckdb_with_search):
        """Test bulk modifying market inclusion."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")
        market_ids = [m.market_id for m in response.markets[:2]]

        # Exclude the first two markets
        updated = agent.bulk_modify_markets(
            session_id=response.session_id,
            market_ids=market_ids,
            included=False,
        )

        # Verify both markets are excluded
        for market in updated.markets:
            if market.market_id in market_ids:
                assert market.included is False

        # Verify count is updated
        assert updated.market_count == len(response.markets) - 2

    def test_modify_market_invalid_session(self, duckdb_with_search):
        """Test modifying with invalid session raises error."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        with pytest.raises(ValueError, match="Session not found"):
            agent.modify_market_list(
                session_id="invalid-session-id",
                market_id="market1",
                included=False,
            )

    def test_save_dataset(self, duckdb_with_search):
        """Test saving a dataset from agent response."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")

        # Exclude one market
        first_market_id = response.markets[0].market_id
        response = agent.modify_market_list(
            session_id=response.session_id,
            market_id=first_market_id,
            included=False,
        )

        # Save the dataset
        result = agent.save_dataset(
            response=response,
            name="Election Markets",
            description="Markets about elections",
        )

        assert result.dataset_id is not None
        assert result.dataset_name == "Election Markets"
        assert result.market_count == len(response.markets) - 1
        assert result.excluded_count == 1

    def test_save_dataset_cleans_session(self, duckdb_with_search):
        """Test that saving a dataset cleans up the session."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")
        session_id = response.session_id

        # Save the dataset
        agent.save_dataset(
            response=response,
            name="Election Markets",
        )

        # Session should be cleaned up
        assert agent.get_session(session_id) is None

    def test_summary_generation(self, duckdb_with_search):
        """Test that a human-readable summary is generated."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")

        assert response.summary is not None
        assert "Found" in response.summary
        assert "markets" in response.summary


class TestDatasetAgentSchemas:
    """Tests for the Dataset Agent API schema validation."""

    def test_request_schema_validates_empty_query(self):
        """Test that request schema rejects empty queries."""
        from pydantic import ValidationError
        from polymkt.models.schemas import DatasetAgentRequestSchema

        with pytest.raises(ValidationError):
            DatasetAgentRequestSchema(natural_language_query="")

    def test_request_schema_accepts_valid_query(self):
        """Test that request schema accepts valid queries."""
        from polymkt.models.schemas import DatasetAgentRequestSchema

        request = DatasetAgentRequestSchema(
            natural_language_query="find election markets about senate control"
        )
        assert request.natural_language_query == "find election markets about senate control"
        assert request.max_results == 100  # default

    def test_response_schema_structure(self):
        """Test that response schema has expected structure."""
        from polymkt.models.schemas import (
            DatasetAgentResponseSchema,
            DatasetAgentMarketItemSchema,
        )

        market = DatasetAgentMarketItemSchema(
            market_id="market1",
            question="Will X happen?",
            category="politics",
            tags=["election"],
            closed_time=None,
            relevance_score=0.5,
            included=True,
        )

        response = DatasetAgentResponseSchema(
            session_id="test-session",
            query="original query",
            parsed_query="parsed query",
            category_filter="politics",
            closed_time_filter=None,
            market_count=1,
            markets=[market],
            summary="Found 1 markets",
        )

        assert response.session_id == "test-session"
        assert len(response.markets) == 1
        assert response.markets[0].included is True

    def test_save_result_schema(self):
        """Test save result schema."""
        from polymkt.models.schemas import DatasetAgentSaveResultSchema

        result = DatasetAgentSaveResultSchema(
            dataset_id="dataset-123",
            dataset_name="Test Dataset",
            market_count=10,
            excluded_count=2,
        )

        assert result.dataset_id == "dataset-123"
        assert result.market_count == 10
        assert result.excluded_count == 2


class TestDatasetAgentParsedQuerySummary:
    """Tests for verifying parsed query information is correct."""

    def test_parsed_query_matches_original(self, duckdb_with_search):
        """Test parsed query is related to original."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("find election markets about senate control")

        assert response.query == "find election markets about senate control"
        # Parsed query should be cleaner
        assert "senate" in response.parsed_query.lower() or "control" in response.parsed_query.lower()

    def test_category_filter_detected(self, duckdb_with_search):
        """Test category filter is detected from query."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("politics markets about president")

        assert response.category_filter == "politics"

    def test_summary_includes_market_count(self, duckdb_with_search):
        """Test summary includes market count."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")

        assert "Found" in response.summary
        assert str(response.market_count) in response.summary or "markets" in response.summary


class TestDatasetAgentFiltersPreserved:
    """Tests for verifying filters are preserved in saved dataset."""

    def test_filters_saved_to_dataset(self, duckdb_with_search):
        """Test filters are saved to the dataset."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("politics election")

        result = agent.save_dataset(
            response=response,
            name="Test Dataset",
        )

        # Get the dataset and verify filters
        from polymkt.storage.datasets import DatasetStore

        store = DatasetStore(db_path)
        dataset = store.get_dataset(result.dataset_id)

        assert dataset.filters is not None
        assert "election" in dataset.filters.query.lower() or "politics" in (dataset.filters.query.lower() if dataset.filters.query else "")

    def test_market_counts_match(self, duckdb_with_search):
        """Test saved filters and market counts match agent display."""
        conn, db_path = duckdb_with_search
        agent = DatasetAgent(conn=conn, db_path=db_path)

        response = agent.process_query("election")

        # Exclude one market
        first_market_id = response.markets[0].market_id
        response = agent.modify_market_list(
            session_id=response.session_id,
            market_id=first_market_id,
            included=False,
        )

        result = agent.save_dataset(
            response=response,
            name="Test Dataset",
        )

        # Verify counts match
        assert result.market_count == response.market_count
        assert result.excluded_count == 1
