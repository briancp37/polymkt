"""Tests for semantic search using OpenAI embeddings and DuckDB vss extension."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.semantic_search import SemanticSearchIndex


# Mock embedding dimension for tests
MOCK_EMBEDDING_DIM = 8


def create_mock_embedding(text: str) -> list[float]:
    """
    Create a deterministic mock embedding based on text content.

    This creates embeddings that are somewhat meaningful for testing:
    - Similar texts will have similar embeddings
    - Different texts will have different embeddings
    """
    import hashlib

    # Create a hash-based seed for reproducibility
    hash_bytes = hashlib.md5(text.encode()).digest()

    # Generate embedding values from hash
    embedding = []
    for i in range(MOCK_EMBEDDING_DIM):
        # Use different byte positions for different dimensions
        byte_val = hash_bytes[i % len(hash_bytes)]
        # Normalize to [-1, 1] range
        normalized = (byte_val / 127.5) - 1.0
        embedding.append(normalized)

    return embedding


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_events_csv(temp_dir: Path) -> Path:
    """Create a sample events CSV file with tags for semantic search testing."""
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
    """Create a sample markets CSV file for semantic search testing."""
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


@pytest.fixture
def mock_openai_client() -> Generator[MagicMock, None, None]:
    """Create a mock OpenAI client for testing."""
    with patch("openai.OpenAI") as mock_class:
        mock_client = MagicMock()

        def mock_create_embeddings(**kwargs: Any) -> MagicMock:
            """Mock the embeddings.create method."""
            input_texts = kwargs.get("input", [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]

            mock_response = MagicMock()
            mock_response.data = []

            for i, text in enumerate(input_texts):
                mock_item = MagicMock()
                mock_item.index = i
                mock_item.embedding = create_mock_embedding(text)
                mock_response.data.append(mock_item)

            return mock_response

        mock_client.embeddings.create = mock_create_embeddings
        mock_class.return_value = mock_client

        yield mock_client


class TestSemanticSearchIndexCreation:
    """Tests for creating the semantic search embeddings table and vss index."""

    def test_create_embeddings_table(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that embeddings table is created with proper structure."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.create_embeddings_table()

            # Verify table structure
            result = duckdb_layer.conn.execute(
                "SELECT * FROM market_embeddings LIMIT 0"
            ).description
            column_names = [col[0] for col in result]

            assert "market_id" in column_names
            assert "embedding" in column_names
            assert "embedding_model" in column_names
            assert "embedding_dim" in column_names
            assert "created_at" in column_names
        finally:
            duckdb_layer.close()

    def test_create_vss_index(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that vss index is created using DuckDB vss extension."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.create_embeddings_table()
            semantic_index.create_vss_index()

            assert semantic_index._index_created is True
        finally:
            duckdb_layer.close()

    def test_build_index_creates_embeddings_for_all_markets(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that build_index creates embeddings for all markets."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            markets_indexed = semantic_index.build_index()

            assert markets_indexed == 7  # 7 markets in test data

            # Verify embeddings are stored
            result = duckdb_layer.conn.execute(
                "SELECT COUNT(*) FROM market_embeddings"
            ).fetchone()
            assert result is not None
            assert result[0] == 7
        finally:
            duckdb_layer.close()

    def test_embeddings_store_model_and_dimension(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that embeddings store the model and dimension metadata."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            result = duckdb_layer.conn.execute(
                "SELECT embedding_model, embedding_dim FROM market_embeddings LIMIT 1"
            ).fetchone()

            assert result is not None
            assert result[0] == "text-embedding-3-small"
            assert result[1] == MOCK_EMBEDDING_DIM
        finally:
            duckdb_layer.close()


class TestSemanticSearch:
    """Tests for semantic search functionality."""

    def test_search_returns_results(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that semantic search returns results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, total_count = semantic_index.search("election")

            assert total_count == 7  # All markets are returned (semantic search matches all)
            assert len(results) > 0
        finally:
            duckdb_layer.close()

    def test_search_returns_similarity_scores(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that search returns cosine similarity scores."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, _ = semantic_index.search("presidential election")

            assert len(results) > 0
            assert all("score" in r for r in results)
            # Cosine similarity ranges from -1 to 1
            assert all(-1 <= r["score"] <= 1 for r in results)
        finally:
            duckdb_layer.close()

    def test_search_results_sorted_by_similarity(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that search results are sorted by similarity score (descending)."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, _ = semantic_index.search("election voting")

            assert len(results) > 1
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            duckdb_layer.close()

    def test_search_returns_market_metadata(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that search results include required market metadata."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, _ = semantic_index.search("basketball")

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
        finally:
            duckdb_layer.close()


class TestSemanticSearchFiltering:
    """Tests for search with category and time filters."""

    def test_filter_by_category(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test filtering search results by category."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, total_count = semantic_index.search("win", category="Sports")

            assert total_count == 2  # Lakers and Celtics
            assert all(r["category"] == "Sports" for r in results)
        finally:
            duckdb_layer.close()

    def test_filter_by_closed_time_min(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test filtering search results by minimum closed_time."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, total_count = semantic_index.search(
                "prediction", closed_time_min="2024-10-01"
            )

            # Should include election markets (Nov) and Bitcoin (Dec)
            assert total_count >= 1
        finally:
            duckdb_layer.close()

    def test_filter_by_closed_time_range(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test filtering search results by closed_time range."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, total_count = semantic_index.search(
                "market",
                closed_time_min="2024-06-01",
                closed_time_max="2024-07-31",
            )

            # Should include only NBA markets (June 30)
            assert total_count == 2
            assert all(r["category"] == "Sports" for r in results)
        finally:
            duckdb_layer.close()

    def test_combined_category_and_time_filter(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test filtering search results by both category and time."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, total_count = semantic_index.search(
                "election",
                category="Politics",
                closed_time_min="2024-10-01",
            )

            assert total_count == 3  # Biden, Trump, Senate
            assert all(r["category"] == "Politics" for r in results)
        finally:
            duckdb_layer.close()


class TestSemanticSearchPagination:
    """Tests for search pagination."""

    def test_pagination_limit(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test limiting search results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, total_count = semantic_index.search("market", limit=2)

            assert len(results) == 2
            assert total_count == 7  # Total matches
        finally:
            duckdb_layer.close()

    def test_pagination_offset(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test offsetting search results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            # Get first page
            results_page1, _ = semantic_index.search("market", limit=2, offset=0)

            # Get second page
            results_page2, _ = semantic_index.search("market", limit=2, offset=2)

            # Pages should be different
            page1_ids = {r["id"] for r in results_page1}
            page2_ids = {r["id"] for r in results_page2}
            assert page1_ids.isdisjoint(page2_ids)
        finally:
            duckdb_layer.close()

    def test_total_count_for_pagination_ui(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that total_count is returned for pagination UI."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            results, total_count = semantic_index.search("market", limit=2)

            assert total_count >= len(results)
        finally:
            duckdb_layer.close()


class TestSemanticSearchIndexMaintenance:
    """Tests for semantic search index refresh and incremental updates."""

    def test_refresh_index_rebuilds_completely(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that refresh_index rebuilds the entire index."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            # Refresh should rebuild
            markets_indexed = semantic_index.refresh_index()
            assert markets_indexed == 7
        finally:
            duckdb_layer.close()

    def test_update_markets_updates_specific_rows(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that update_markets updates specific market embeddings."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            # Update specific markets
            updated_count = semantic_index.update_markets(["market1", "market2"])
            assert updated_count == 2

            # Verify total still 7
            result = duckdb_layer.conn.execute(
                "SELECT COUNT(*) FROM market_embeddings"
            ).fetchone()
            assert result is not None
            assert result[0] == 7
        finally:
            duckdb_layer.close()

    def test_update_markets_with_empty_list(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that update_markets with empty list does nothing."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            updated_count = semantic_index.update_markets([])
            assert updated_count == 0
        finally:
            duckdb_layer.close()


class TestSemanticSearchEdgeCases:
    """Tests for semantic search edge cases and error handling."""

    def test_search_without_building_index_raises_error(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test that searching without building index raises RuntimeError."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )

            with pytest.raises(RuntimeError) as exc_info:
                semantic_index.search("test")

            assert "Semantic search index not built" in str(exc_info.value)
        finally:
            duckdb_layer.close()

    def test_search_without_api_key_raises_error(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that searching without API key raises ValueError."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key=None,  # No API key
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )

            with pytest.raises(ValueError) as exc_info:
                semantic_index.build_index()

            assert "OpenAI API key is required" in str(exc_info.value)
        finally:
            duckdb_layer.close()

    def test_get_embedding_stats_with_no_embeddings(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test getting stats when no embeddings exist."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )

            stats = semantic_index.get_embedding_stats()

            assert stats["total_embeddings"] == 0
            assert stats["embedding_model"] is None
        finally:
            duckdb_layer.close()

    def test_get_embedding_stats_with_embeddings(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test getting stats after building index."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)

        try:
            duckdb_layer.create_views()
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key="test-key",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=MOCK_EMBEDDING_DIM,
            )
            semantic_index.build_index()

            stats = semantic_index.get_embedding_stats()

            assert stats["total_embeddings"] == 7
            assert stats["embedding_model"] == "text-embedding-3-small"
            assert stats["embedding_dim"] == MOCK_EMBEDDING_DIM
        finally:
            duckdb_layer.close()


class TestSemanticSearchAPIIntegration:
    """Tests for semantic search API endpoint integration."""

    def test_api_build_semantic_index_requires_api_key(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that /api/semantic-search/build requires API key."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db

        # Save and clear API key
        original_api_key = settings.openai_api_key
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path

        settings.openai_api_key = None
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path

        try:
            client = TestClient(app)
            response = client.post("/api/semantic-search/build")

            assert response.status_code == 400
            assert "OpenAI API key not configured" in response.json()["detail"]
        finally:
            settings.openai_api_key = original_api_key
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_semantic_search_requires_api_key(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that /api/markets/semantic-search requires API key."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db

        original_api_key = settings.openai_api_key
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path

        settings.openai_api_key = None
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path

        try:
            client = TestClient(app)
            response = client.get("/api/markets/semantic-search?q=election")

            assert response.status_code == 400
            assert "OpenAI API key not configured" in response.json()["detail"]
        finally:
            settings.openai_api_key = original_api_key
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path

    def test_api_get_embedding_stats(
        self, bootstrapped_db: tuple[Path, Path], mock_openai_client: MagicMock
    ) -> None:
        """Test the /api/semantic-search/stats endpoint."""
        from fastapi.testclient import TestClient

        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db

        original_api_key = settings.openai_api_key
        original_parquet_dir = settings.parquet_dir
        original_duckdb_path = settings.duckdb_path
        original_embedding_dim = settings.openai_embedding_dimensions

        settings.openai_api_key = "test-key"
        settings.parquet_dir = parquet_dir
        settings.duckdb_path = duckdb_path
        settings.openai_embedding_dimensions = MOCK_EMBEDDING_DIM

        try:
            client = TestClient(app)
            response = client.get("/api/semantic-search/stats")

            assert response.status_code == 200
            data = response.json()

            assert "total_embeddings" in data
            assert "embedding_model" in data
            assert "embedding_dim" in data
        finally:
            settings.openai_api_key = original_api_key
            settings.parquet_dir = original_parquet_dir
            settings.duckdb_path = original_duckdb_path
            settings.openai_embedding_dimensions = original_embedding_dim
