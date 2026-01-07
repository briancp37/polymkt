"""Incremental search index updater that detects changed markets and updates indices."""

import hashlib
from typing import Any

import duckdb
import structlog

from polymkt.storage.hybrid_search import HybridSearchIndex

logger = structlog.get_logger()


def compute_market_content_hash(
    question: str | None,
    tags: list[str] | None,
    description: str | None,
) -> str:
    """
    Compute a hash of market content fields used for search.

    This hash is used to detect changes that require search index updates.

    Args:
        question: Market question text
        tags: List of tags
        description: Market description

    Returns:
        MD5 hash of combined content
    """
    parts = []
    if question:
        parts.append(question)
    if tags:
        parts.append("|".join(sorted(tags)))
    if description:
        parts.append(description)

    content = "||".join(parts)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class SearchIndexUpdater:
    """
    Manages incremental updates to search indices.

    This class:
    1. Tracks content hashes for markets to detect changes
    2. Identifies new/changed markets during updates
    3. Updates only affected markets in BM25 and semantic indices
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
    ) -> None:
        """
        Initialize the search index updater.

        Args:
            conn: DuckDB connection with v_markets view already created
            openai_api_key: OpenAI API key for semantic search (optional)
            embedding_model: OpenAI embedding model to use
            embedding_dimensions: Dimension of the embedding vectors
        """
        self.conn = conn
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

        # Initialize hybrid search index
        self.hybrid_index = HybridSearchIndex(
            conn=conn,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )

        # Ensure content hash table exists
        self._ensure_hash_table()

    def _ensure_hash_table(self) -> None:
        """Create the market content hash table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_content_hashes (
                market_id VARCHAR PRIMARY KEY,
                content_hash VARCHAR NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _get_stored_hashes(self) -> dict[str, str]:
        """Get all stored content hashes."""
        try:
            result = self.conn.execute("""
                SELECT market_id, content_hash
                FROM market_content_hashes
            """)
            return {row[0]: row[1] for row in result.fetchall()}
        except Exception:
            return {}

    def _compute_current_hashes(self) -> dict[str, str]:
        """Compute content hashes for all current markets."""
        result = self.conn.execute("""
            SELECT id, question, tags, description
            FROM v_markets
        """)

        hashes = {}
        for row in result.fetchall():
            market_id = row[0]
            question = row[1]
            tags = row[2] if row[2] else None
            description = row[3] if row[3] else None
            hashes[market_id] = compute_market_content_hash(question, tags, description)

        return hashes

    def detect_changed_markets(self) -> tuple[list[str], list[str], list[str]]:
        """
        Detect which markets have changed by comparing content hashes.

        Returns:
            Tuple of (new_market_ids, changed_market_ids, deleted_market_ids)
        """
        stored_hashes = self._get_stored_hashes()
        current_hashes = self._compute_current_hashes()

        new_markets: list[str] = []
        changed_markets: list[str] = []
        deleted_markets: list[str] = []

        # Find new and changed markets
        for market_id, current_hash in current_hashes.items():
            if market_id not in stored_hashes:
                new_markets.append(market_id)
            elif stored_hashes[market_id] != current_hash:
                changed_markets.append(market_id)

        # Find deleted markets
        for market_id in stored_hashes:
            if market_id not in current_hashes:
                deleted_markets.append(market_id)

        logger.info(
            "markets_change_detection",
            new_count=len(new_markets),
            changed_count=len(changed_markets),
            deleted_count=len(deleted_markets),
        )

        return new_markets, changed_markets, deleted_markets

    def _update_stored_hashes(
        self,
        new_market_ids: list[str],
        changed_market_ids: list[str],
        deleted_market_ids: list[str],
        current_hashes: dict[str, str],
    ) -> None:
        """Update the stored content hashes after processing."""
        # Delete removed markets
        if deleted_market_ids:
            placeholders = ", ".join(["?" for _ in deleted_market_ids])
            self.conn.execute(
                f"DELETE FROM market_content_hashes WHERE market_id IN ({placeholders})",
                deleted_market_ids,
            )

        # Insert/update new and changed markets
        all_affected = new_market_ids + changed_market_ids
        for market_id in all_affected:
            content_hash = current_hashes.get(market_id)
            if content_hash:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO market_content_hashes
                    (market_id, content_hash, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    [market_id, content_hash],
                )

    def update_indices(
        self,
        force_rebuild: bool = False,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """
        Update search indices for changed markets only.

        This method:
        1. Detects which markets have new/changed content
        2. Updates only those markets in BM25 and semantic indices
        3. Updates stored content hashes

        Args:
            force_rebuild: If True, rebuild all indices from scratch
            batch_size: Batch size for embedding generation

        Returns:
            Dictionary with update statistics
        """
        if force_rebuild:
            logger.info("force_rebuild_requested", rebuilding_all=True)

            # Build BM25 index (always)
            bm25_count = self.hybrid_index.build_bm25_index()

            # Build semantic index only if API key is available
            semantic_count = 0
            if self.openai_api_key:
                semantic_count = self.hybrid_index.build_semantic_index(batch_size=batch_size)

            # Rebuild hash table
            self.conn.execute("DELETE FROM market_content_hashes")
            current_hashes = self._compute_current_hashes()
            for market_id, content_hash in current_hashes.items():
                self.conn.execute(
                    """
                    INSERT INTO market_content_hashes
                    (market_id, content_hash, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    [market_id, content_hash],
                )

            return {
                "mode": "full_rebuild",
                "bm25_indexed": bm25_count,
                "semantic_indexed": semantic_count,
                "new_markets": 0,
                "changed_markets": 0,
                "deleted_markets": 0,
            }

        # Detect changes
        new_market_ids, changed_market_ids, deleted_market_ids = self.detect_changed_markets()
        affected_market_ids = new_market_ids + changed_market_ids

        if not affected_market_ids and not deleted_market_ids:
            logger.info("no_markets_to_update")
            return {
                "mode": "incremental",
                "bm25_updated": 0,
                "semantic_updated": 0,
                "new_markets": 0,
                "changed_markets": 0,
                "deleted_markets": 0,
            }

        # Check if indices exist, if not do full build
        bm25_available = self.hybrid_index._check_bm25_available()
        semantic_available = self.hybrid_index._check_semantic_available()

        if not bm25_available:
            logger.info("bm25_index_not_found", building_from_scratch=True)
            self.hybrid_index.build_bm25_index()
            bm25_updated = len(affected_market_ids)
        else:
            # Update only affected markets in BM25
            bm25_updated = self.hybrid_index.bm25_index.update_markets(affected_market_ids)

        semantic_updated = 0
        if self.openai_api_key:
            if not semantic_available:
                logger.info("semantic_index_not_found", building_from_scratch=True)
                self.hybrid_index.build_semantic_index(batch_size=batch_size)
                semantic_updated = len(affected_market_ids)
            else:
                # Update only affected markets in semantic index
                semantic_updated = self.hybrid_index.semantic_index.update_markets(
                    affected_market_ids, batch_size=batch_size
                )

        # Update stored hashes
        current_hashes = self._compute_current_hashes()
        self._update_stored_hashes(
            new_market_ids, changed_market_ids, deleted_market_ids, current_hashes
        )

        logger.info(
            "search_indices_updated",
            new_markets=len(new_market_ids),
            changed_markets=len(changed_market_ids),
            deleted_markets=len(deleted_market_ids),
            bm25_updated=bm25_updated,
            semantic_updated=semantic_updated,
        )

        return {
            "mode": "incremental",
            "bm25_updated": bm25_updated,
            "semantic_updated": semantic_updated,
            "new_markets": len(new_market_ids),
            "changed_markets": len(changed_market_ids),
            "deleted_markets": len(deleted_market_ids),
        }

    def update_specific_markets(
        self,
        market_ids: list[str],
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """
        Update search indices for specific market IDs.

        Use this when you already know which markets changed (e.g., from update pipeline).

        Args:
            market_ids: List of market IDs to update
            batch_size: Batch size for embedding generation

        Returns:
            Dictionary with update statistics
        """
        if not market_ids:
            return {
                "mode": "specific",
                "bm25_updated": 0,
                "semantic_updated": 0,
                "markets_requested": 0,
            }

        # Update BM25
        bm25_updated = self.hybrid_index.bm25_index.update_markets(market_ids)

        # Update semantic if API key available
        semantic_updated = 0
        if self.openai_api_key:
            semantic_updated = self.hybrid_index.semantic_index.update_markets(
                market_ids, batch_size=batch_size
            )

        # Update stored hashes for these markets
        current_hashes = self._compute_current_hashes()
        for market_id in market_ids:
            content_hash = current_hashes.get(market_id)
            if content_hash:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO market_content_hashes
                    (market_id, content_hash, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    [market_id, content_hash],
                )

        logger.info(
            "specific_markets_updated",
            markets_requested=len(market_ids),
            bm25_updated=bm25_updated,
            semantic_updated=semantic_updated,
        )

        return {
            "mode": "specific",
            "bm25_updated": bm25_updated,
            "semantic_updated": semantic_updated,
            "markets_requested": len(market_ids),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the search index updater."""
        # Get hash table stats
        try:
            result = self.conn.execute("""
                SELECT
                    COUNT(*) as total_hashes,
                    MIN(updated_at) as first_updated,
                    MAX(updated_at) as last_updated
                FROM market_content_hashes
            """).fetchone()
            hash_stats = {
                "total_hashes": result[0] if result else 0,
                "first_updated": result[1] if result else None,
                "last_updated": result[2] if result else None,
            }
        except Exception:
            hash_stats = {
                "total_hashes": 0,
                "first_updated": None,
                "last_updated": None,
            }

        # Get hybrid index stats
        index_stats = self.hybrid_index.get_index_stats()

        return {
            **hash_stats,
            **index_stats,
        }
