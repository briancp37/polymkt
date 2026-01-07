"""Hybrid search combining BM25 full-text and semantic vector search."""

from typing import Any

import duckdb
import structlog

from polymkt.storage.search import MarketSearchIndex
from polymkt.storage.semantic_search import SemanticSearchIndex

logger = structlog.get_logger()


class HybridSearchIndex:
    """Hybrid search combining BM25 and semantic vector search for markets.

    This class merges results from BM25 full-text search and semantic vector
    search to provide better results that work for both keyword-heavy and
    semantic-heavy queries.

    The hybrid scoring uses Reciprocal Rank Fusion (RRF) to combine rankings
    from both search methods. This is a proven technique that works well
    without needing to tune weights between different score scales.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        rrf_k: int = 60,
    ) -> None:
        """
        Initialize the hybrid search index.

        Args:
            conn: DuckDB connection with v_markets view already created
            openai_api_key: OpenAI API key for semantic search
            embedding_model: OpenAI embedding model to use
            embedding_dimensions: Dimension of the embedding vectors
            rrf_k: RRF constant (default 60, typical value from literature)
        """
        self.conn = conn
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.rrf_k = rrf_k

        # Initialize component indices
        self.bm25_index = MarketSearchIndex(conn)
        self.semantic_index = SemanticSearchIndex(
            conn=conn,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )

        self._bm25_ready = False
        self._semantic_ready = False

    def build_bm25_index(self) -> int:
        """
        Build or rebuild the BM25 full-text search index.

        Returns:
            Number of markets indexed
        """
        count = self.bm25_index.build_index()
        self._bm25_ready = True
        logger.info("hybrid_bm25_index_built", markets_indexed=count)
        return count

    def build_semantic_index(self, batch_size: int = 100) -> int:
        """
        Build or rebuild the semantic search index.

        Requires OpenAI API key to be configured.

        Args:
            batch_size: Number of markets to embed per API call

        Returns:
            Number of markets indexed
        """
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required for semantic search. "
                "Set POLYMKT_OPENAI_API_KEY environment variable."
            )
        count = self.semantic_index.build_index(batch_size=batch_size)
        self._semantic_ready = True
        logger.info("hybrid_semantic_index_built", markets_indexed=count)
        return count

    def build_index(self, batch_size: int = 100) -> dict[str, int]:
        """
        Build both BM25 and semantic search indices.

        Args:
            batch_size: Number of markets to embed per API call for semantic

        Returns:
            Dictionary with 'bm25' and 'semantic' counts
        """
        bm25_count = self.build_bm25_index()
        semantic_count = self.build_semantic_index(batch_size=batch_size)
        return {
            "bm25": bm25_count,
            "semantic": semantic_count,
        }

    def _compute_rrf_score(self, rank: int) -> float:
        """
        Compute Reciprocal Rank Fusion score for a given rank.

        RRF(d) = 1 / (k + rank(d))

        where k is a constant (default 60) that mitigates the impact of
        high rankings from outlier results.

        Args:
            rank: 1-based rank of the document

        Returns:
            RRF score (higher is better)
        """
        return 1.0 / (self.rrf_k + rank)

    def _merge_results(
        self,
        bm25_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge BM25 and semantic results using Reciprocal Rank Fusion.

        Args:
            bm25_results: Results from BM25 search (ranked by BM25 score)
            semantic_results: Results from semantic search (ranked by cosine similarity)

        Returns:
            Merged and re-ranked results with hybrid scores
        """
        # Collect all unique market IDs and their data
        markets: dict[str, dict[str, Any]] = {}

        # Track RRF scores for each source
        rrf_scores: dict[str, dict[str, float]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            market_id = result["id"]
            if market_id not in markets:
                markets[market_id] = {
                    "id": result["id"],
                    "question": result["question"],
                    "tags": result.get("tags"),
                    "category": result.get("category"),
                    "closed_time": result.get("closed_time"),
                    "event_id": result.get("event_id"),
                }
                rrf_scores[market_id] = {"bm25": 0.0, "semantic": 0.0}

            rrf_scores[market_id]["bm25"] = self._compute_rrf_score(rank)
            markets[market_id]["bm25_score"] = result["score"]
            markets[market_id]["bm25_rank"] = rank

        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            market_id = result["id"]
            if market_id not in markets:
                markets[market_id] = {
                    "id": result["id"],
                    "question": result["question"],
                    "tags": result.get("tags"),
                    "category": result.get("category"),
                    "closed_time": result.get("closed_time"),
                    "event_id": result.get("event_id"),
                }
                rrf_scores[market_id] = {"bm25": 0.0, "semantic": 0.0}

            rrf_scores[market_id]["semantic"] = self._compute_rrf_score(rank)
            markets[market_id]["semantic_score"] = result["score"]
            markets[market_id]["semantic_rank"] = rank

        # Compute combined RRF scores and add to results
        merged_results = []
        for market_id, market_data in markets.items():
            bm25_rrf = rrf_scores[market_id]["bm25"]
            semantic_rrf = rrf_scores[market_id]["semantic"]
            combined_rrf = bm25_rrf + semantic_rrf

            market_data["score"] = combined_rrf
            market_data["bm25_rrf"] = bm25_rrf
            market_data["semantic_rrf"] = semantic_rrf
            merged_results.append(market_data)

        # Sort by combined RRF score (descending), then by market ID for stability
        merged_results.sort(key=lambda x: (-x["score"], x["id"]))

        return merged_results

    def search(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        closed_time_min: str | None = None,
        closed_time_max: str | None = None,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        top_k_per_source: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Search markets using hybrid BM25 + semantic search.

        Retrieves top-K results from each source, merges them using
        Reciprocal Rank Fusion, and returns the final ranked list.

        Args:
            query: Search query string
            limit: Maximum results to return (default 50)
            offset: Number of results to skip (default 0)
            category: Filter by category (optional)
            closed_time_min: Filter by min closed_time (optional)
            closed_time_max: Filter by max closed_time (optional)
            bm25_weight: Weight for BM25 scores (not used with RRF, kept for API compatibility)
            semantic_weight: Weight for semantic scores (not used with RRF, kept for API compatibility)
            top_k_per_source: Number of results to retrieve from each source
                             before merging. Default is limit * 2 for good coverage.

        Returns:
            Tuple of (results list, total_count) where results include:
            - id: Market ID
            - question: Market question
            - tags: Tags list
            - category: Market category
            - closed_time: Market close time
            - event_id: Parent event ID
            - score: Combined RRF score
            - bm25_score: Original BM25 score (if in BM25 results)
            - semantic_score: Original cosine similarity (if in semantic results)
            - bm25_rank: Rank in BM25 results (if present)
            - semantic_rank: Rank in semantic results (if present)

        Raises:
            RuntimeError: If neither index has been built
            ValueError: If semantic search needed but no API key
        """
        # Determine how many results to fetch from each source
        if top_k_per_source is None:
            # Fetch more than needed to ensure good coverage after merging
            top_k_per_source = max(limit + offset, 100)

        # Check what indices are available
        bm25_available = self._check_bm25_available()
        semantic_available = self._check_semantic_available()

        if not bm25_available and not semantic_available:
            raise RuntimeError(
                "No search indices available. Build at least one index first "
                "(build_bm25_index, build_semantic_index, or build_index)."
            )

        # Collect results from available sources
        bm25_results: list[dict[str, Any]] = []
        semantic_results: list[dict[str, Any]] = []

        if bm25_available:
            try:
                bm25_results, _ = self.bm25_index.search(
                    query=query,
                    limit=top_k_per_source,
                    offset=0,  # Always start from 0 for merging
                    category=category,
                    closed_time_min=closed_time_min,
                    closed_time_max=closed_time_max,
                )
            except Exception as e:
                logger.warning("bm25_search_failed", error=str(e))

        if semantic_available and self.openai_api_key:
            try:
                semantic_results, _ = self.semantic_index.search(
                    query=query,
                    limit=top_k_per_source,
                    offset=0,  # Always start from 0 for merging
                    category=category,
                    closed_time_min=closed_time_min,
                    closed_time_max=closed_time_max,
                )
            except Exception as e:
                logger.warning("semantic_search_failed", error=str(e))

        # If only one source returned results, use those directly
        if not bm25_results and semantic_results:
            merged = semantic_results
            for r in merged:
                r["score"] = r.get("score", 0)
                r["semantic_score"] = r["score"]
                r["bm25_score"] = None
        elif bm25_results and not semantic_results:
            merged = bm25_results
            for r in merged:
                r["bm25_score"] = r.get("score", 0)
                r["semantic_score"] = None
        elif not bm25_results and not semantic_results:
            return [], 0
        else:
            # Merge results using RRF
            merged = self._merge_results(bm25_results, semantic_results)

        # Calculate total unique results before pagination
        total_count = len(merged)

        # Apply pagination
        paginated = merged[offset : offset + limit]

        logger.info(
            "hybrid_search_completed",
            query=query[:50] + "..." if len(query) > 50 else query,
            bm25_results=len(bm25_results),
            semantic_results=len(semantic_results),
            total_unique=total_count,
            returned=len(paginated),
        )

        return paginated, total_count

    def _check_bm25_available(self) -> bool:
        """Check if BM25 index is available."""
        if self._bm25_ready:
            return True
        try:
            self.conn.execute(
                "SELECT 1 FROM fts_main_markets_search.docs LIMIT 1"
            ).fetchone()
            self._bm25_ready = True
            return True
        except Exception:
            return False

    def _check_semantic_available(self) -> bool:
        """Check if semantic index is available."""
        if self._semantic_ready:
            return True
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM market_embeddings"
            ).fetchone()
            if result and result[0] > 0:
                # Load vss extension
                self.conn.execute("INSTALL vss")
                self.conn.execute("LOAD vss")
                self._semantic_ready = True
                return True
        except Exception:
            pass
        return False

    def refresh_index(self, batch_size: int = 100) -> dict[str, int]:
        """
        Refresh both search indices with current market data.

        Args:
            batch_size: Number of markets to embed per API call for semantic

        Returns:
            Dictionary with 'bm25' and 'semantic' refresh counts
        """
        bm25_count = self.bm25_index.refresh_index()
        self._bm25_ready = True

        semantic_count = 0
        if self.openai_api_key:
            semantic_count = self.semantic_index.refresh_index(batch_size=batch_size)
            self._semantic_ready = True

        logger.info(
            "hybrid_index_refreshed",
            bm25_count=bm25_count,
            semantic_count=semantic_count,
        )
        return {
            "bm25": bm25_count,
            "semantic": semantic_count,
        }

    def update_markets(self, market_ids: list[str], batch_size: int = 100) -> dict[str, int]:
        """
        Update specific markets in both indices.

        Args:
            market_ids: List of market IDs that changed
            batch_size: Number of markets to embed per API call for semantic

        Returns:
            Dictionary with 'bm25' and 'semantic' update counts
        """
        if not market_ids:
            return {"bm25": 0, "semantic": 0}

        bm25_count = self.bm25_index.update_markets(market_ids)

        semantic_count = 0
        if self.openai_api_key:
            semantic_count = self.semantic_index.update_markets(
                market_ids, batch_size=batch_size
            )

        logger.info(
            "hybrid_index_updated",
            market_ids=len(market_ids),
            bm25_count=bm25_count,
            semantic_count=semantic_count,
        )
        return {
            "bm25": bm25_count,
            "semantic": semantic_count,
        }

    def get_index_stats(self) -> dict[str, Any]:
        """
        Get statistics about both search indices.

        Returns:
            Dictionary with stats for both indices
        """
        stats: dict[str, Any] = {
            "bm25_available": self._check_bm25_available(),
            "semantic_available": self._check_semantic_available(),
        }

        if stats["bm25_available"]:
            try:
                result = self.conn.execute(
                    "SELECT COUNT(*) FROM markets_search"
                ).fetchone()
                stats["bm25_markets_indexed"] = result[0] if result else 0
            except Exception:
                stats["bm25_markets_indexed"] = 0

        if stats["semantic_available"]:
            semantic_stats = self.semantic_index.get_embedding_stats()
            stats["semantic_markets_indexed"] = semantic_stats["total_embeddings"]
            stats["semantic_embedding_model"] = semantic_stats["embedding_model"]
            stats["semantic_embedding_dim"] = semantic_stats["embedding_dim"]

        return stats


def create_hybrid_search_index(
    conn: duckdb.DuckDBPyConnection,
    openai_api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_dimensions: int = 1536,
    batch_size: int = 100,
) -> HybridSearchIndex:
    """
    Create and build a hybrid search index.

    If OpenAI API key is provided, builds both BM25 and semantic indices.
    Otherwise, only builds BM25 index.

    Args:
        conn: DuckDB connection with v_markets view already created
        openai_api_key: OpenAI API key for semantic search (optional)
        embedding_model: OpenAI embedding model to use
        embedding_dimensions: Dimension of the embedding vectors
        batch_size: Number of markets to embed per API call

    Returns:
        Initialized and built HybridSearchIndex
    """
    index = HybridSearchIndex(
        conn=conn,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )

    # Always build BM25 index
    index.build_bm25_index()

    # Build semantic index if API key provided
    if openai_api_key:
        index.build_semantic_index(batch_size=batch_size)

    return index
