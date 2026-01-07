"""Full-text search functionality using DuckDB FTS extension."""

from typing import Any

import duckdb
import structlog

logger = structlog.get_logger()


class MarketSearchIndex:
    """BM25-based full-text search index for markets using DuckDB FTS extension."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Initialize the search index.

        Args:
            conn: DuckDB connection with v_markets view already created
        """
        self.conn = conn
        self._index_created = False

    def create_search_table(self) -> int:
        """
        Create the markets_search table with flattened tags for FTS indexing.

        This table contains:
        - id: Market ID
        - question: Market question text
        - tags: Original tags list (preserved for UI)
        - tags_text: Flattened tags as space-separated string (for FTS)
        - description: Market description (optional, for FTS)
        - category: Market category
        - closed_time: Market close time

        Returns:
            Number of rows in the search table
        """
        # Create the search table by materializing from v_markets with tags flattened
        create_sql = """
            CREATE OR REPLACE TABLE markets_search AS
            SELECT
                id,
                question,
                tags,
                -- Flatten tags list into searchable text field
                CASE
                    WHEN tags IS NOT NULL AND len(tags) > 0
                    THEN array_to_string(tags, ' ')
                    ELSE ''
                END AS tags_text,
                COALESCE(description, '') AS description,
                category,
                closed_time,
                event_id
            FROM v_markets
        """
        self.conn.execute(create_sql)

        # Get row count
        result = self.conn.execute("SELECT COUNT(*) FROM markets_search").fetchone()
        row_count = result[0] if result else 0

        logger.info("markets_search_table_created", rows=row_count)
        return row_count

    def create_fts_index(self) -> None:
        """
        Create the FTS index over question and tags_text using DuckDB's FTS extension.

        The index uses BM25 scoring for relevance ranking.
        """
        # Install and load FTS extension
        self.conn.execute("INSTALL fts")
        self.conn.execute("LOAD fts")

        # Create the FTS index over question, tags_text, and description
        # Using stemmer='porter' for English text
        create_index_sql = """
            PRAGMA create_fts_index(
                'markets_search',
                'id',
                'question',
                'tags_text',
                'description',
                stemmer='porter',
                stopwords='english',
                ignore='(\\.|[^a-z])+',
                strip_accents=1,
                lower=1,
                overwrite=1
            )
        """
        self.conn.execute(create_index_sql)
        self._index_created = True

        logger.info("fts_index_created", table="markets_search")

    def build_index(self) -> int:
        """
        Build the complete search index (table + FTS).

        Returns:
            Number of markets indexed
        """
        row_count = self.create_search_table()
        self.create_fts_index()
        return row_count

    def search(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        closed_time_min: str | None = None,
        closed_time_max: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Search markets using BM25 full-text search.

        Args:
            query: Search query string
            limit: Maximum results to return (default 50)
            offset: Number of results to skip (default 0)
            category: Filter by category (optional)
            closed_time_min: Filter by min closed_time (optional)
            closed_time_max: Filter by max closed_time (optional)

        Returns:
            Tuple of (results list, total_count) where results include:
            - id: Market ID
            - question: Market question
            - tags: Tags list
            - category: Market category
            - closed_time: Market close time
            - score: BM25 relevance score

        Raises:
            RuntimeError: If index has not been built
        """
        if not self._index_created:
            # Check if index exists by trying a simple query
            try:
                self.conn.execute(
                    "SELECT 1 FROM fts_main_markets_search.docs LIMIT 1"
                ).fetchone()
                self._index_created = True
            except Exception:
                raise RuntimeError(
                    "Search index not built. Call build_index() first."
                )

        # Build the search query with BM25 scoring
        # DuckDB FTS returns score via fts_main_<table>.match_bm25()
        conditions: list[str] = []
        params: list[Any] = []

        if category:
            conditions.append("ms.category = ?")
            params.append(category)

        if closed_time_min:
            conditions.append("ms.closed_time >= ?")
            params.append(closed_time_min)

        if closed_time_max:
            conditions.append("ms.closed_time <= ?")
            params.append(closed_time_max)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # First get total count
        # NOTE: DuckDB FTS returns one row per indexed field per document,
        # so we aggregate by id using MAX(score) to get one result per market
        count_sql = f"""
            SELECT COUNT(*)
            FROM markets_search ms
            INNER JOIN (
                SELECT id, MAX(fts_main_markets_search.match_bm25(id, ?, fields := 'question,tags_text,description')) AS score
                FROM fts_main_markets_search.docs
                GROUP BY id
            ) fts ON ms.id = fts.id
            WHERE fts.score IS NOT NULL AND {where_clause}
        """
        count_params = [query] + params
        count_result = self.conn.execute(count_sql, count_params).fetchone()
        total_count = count_result[0] if count_result else 0

        # Now get paginated results sorted by relevance
        # Use MAX(score) and GROUP BY id to deduplicate across indexed fields
        search_sql = f"""
            SELECT
                ms.id,
                ms.question,
                ms.tags,
                ms.category,
                ms.closed_time,
                ms.event_id,
                fts.score AS score
            FROM markets_search ms
            INNER JOIN (
                SELECT id, MAX(fts_main_markets_search.match_bm25(id, ?, fields := 'question,tags_text,description')) AS score
                FROM fts_main_markets_search.docs
                GROUP BY id
            ) fts ON ms.id = fts.id
            WHERE fts.score IS NOT NULL AND {where_clause}
            ORDER BY fts.score DESC, ms.id ASC
            LIMIT ? OFFSET ?
        """
        search_params = [query] + params + [limit, offset]

        result = self.conn.execute(search_sql, search_params)
        columns = [desc[0] for desc in result.description]
        results = [dict(zip(columns, row)) for row in result.fetchall()]

        logger.info(
            "market_search_completed",
            query=query,
            total_count=total_count,
            returned=len(results),
        )

        return results, total_count

    def refresh_index(self) -> int:
        """
        Refresh the search index with current market data.

        This rebuilds the entire index. For incremental updates, use
        update_markets() instead.

        Returns:
            Number of markets indexed
        """
        return self.build_index()

    def update_markets(self, market_ids: list[str]) -> int:
        """
        Update specific markets in the search index.

        This is more efficient than rebuild for incremental updates.

        Args:
            market_ids: List of market IDs that changed

        Returns:
            Number of markets updated
        """
        if not market_ids:
            return 0

        # Delete existing rows for these markets
        placeholders = ", ".join(["?" for _ in market_ids])
        delete_sql = f"DELETE FROM markets_search WHERE id IN ({placeholders})"
        self.conn.execute(delete_sql, market_ids)

        # Insert updated rows from v_markets
        insert_sql = f"""
            INSERT INTO markets_search
            SELECT
                id,
                question,
                tags,
                CASE
                    WHEN tags IS NOT NULL AND len(tags) > 0
                    THEN array_to_string(tags, ' ')
                    ELSE ''
                END AS tags_text,
                COALESCE(description, '') AS description,
                category,
                closed_time,
                event_id
            FROM v_markets
            WHERE id IN ({placeholders})
        """
        self.conn.execute(insert_sql, market_ids)

        # Rebuild FTS index (DuckDB FTS requires full rebuild for updates)
        self.create_fts_index()

        logger.info("markets_search_updated", market_ids=len(market_ids))
        return len(market_ids)


def create_search_index(conn: duckdb.DuckDBPyConnection) -> MarketSearchIndex:
    """
    Create and build a market search index.

    Args:
        conn: DuckDB connection with v_markets view already created

    Returns:
        Initialized and built MarketSearchIndex
    """
    index = MarketSearchIndex(conn)
    index.build_index()
    return index
