"""Semantic search functionality using OpenAI embeddings and DuckDB vss extension."""

from typing import Any

import duckdb
import structlog

logger = structlog.get_logger()


class SemanticSearchIndex:
    """Semantic search index for markets using OpenAI embeddings and DuckDB vss extension."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
    ) -> None:
        """
        Initialize the semantic search index.

        Args:
            conn: DuckDB connection with v_markets view already created
            openai_api_key: OpenAI API key for generating embeddings
            embedding_model: OpenAI embedding model to use
            embedding_dimensions: Dimension of the embedding vectors
        """
        self.conn = conn
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self._index_created = False
        self._openai_client: Any = None

    def _get_openai_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._openai_client is None:
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key is required for semantic search. "
                    "Set POLYMKT_OPENAI_API_KEY environment variable."
                )
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.openai_api_key)
        return self._openai_client

    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text using OpenAI API.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        client = self._get_openai_client()
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=self.embedding_dimensions,
        )
        embedding: list[float] = response.data[0].embedding
        return embedding

    def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = self._get_openai_client()
        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            dimensions=self.embedding_dimensions,
        )
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def _prepare_text_for_embedding(
        self, question: str, tags: list[str] | None, description: str | None
    ) -> str:
        """
        Prepare market text for embedding generation.

        Combines question, tags, and description into a single text.

        Args:
            question: Market question
            tags: List of tags (can be None)
            description: Market description (can be None)

        Returns:
            Combined text for embedding
        """
        parts = [question]
        if tags:
            parts.append(" ".join(tags))
        if description:
            parts.append(description)
        return " ".join(parts)

    def create_embeddings_table(self) -> None:
        """
        Create the market_embeddings table to store embedding vectors.

        The table has:
        - market_id: Primary key linking to markets
        - embedding: FLOAT array of embedding vectors
        - embedding_model: Model used to generate the embedding
        - embedding_dim: Dimension of the embedding
        - created_at: Timestamp of embedding creation
        """
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS market_embeddings (
                market_id VARCHAR PRIMARY KEY,
                embedding FLOAT[{self.embedding_dimensions}],
                embedding_model VARCHAR,
                embedding_dim INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        self.conn.execute(create_sql)
        logger.info(
            "market_embeddings_table_created",
            embedding_dim=self.embedding_dimensions,
            model=self.embedding_model,
        )

    def create_vss_index(self) -> None:
        """
        Create the vector similarity search (vss) index using DuckDB vss extension.

        This enables efficient approximate nearest neighbor (ANN) queries.
        Note: HNSW index creation requires either in-memory databases or
        enabling experimental persistence with SET hnsw_enable_experimental_persistence = true.
        """
        # Install and load vss extension
        self.conn.execute("INSTALL vss")
        self.conn.execute("LOAD vss")

        # Enable experimental persistence for on-disk databases
        try:
            self.conn.execute("SET hnsw_enable_experimental_persistence = true")
        except Exception:
            pass  # May not be required for in-memory databases

        # Create HNSW index for approximate nearest neighbor search
        # Drop existing index if it exists
        try:
            self.conn.execute("DROP INDEX IF EXISTS market_embeddings_vss_idx")
        except Exception:
            pass  # Index may not exist

        create_index_sql = """
            CREATE INDEX market_embeddings_vss_idx ON market_embeddings
            USING HNSW (embedding)
            WITH (metric = 'cosine')
        """
        self.conn.execute(create_index_sql)
        self._index_created = True

        logger.info("vss_index_created", table="market_embeddings", metric="cosine")

    def build_index(self, batch_size: int = 100) -> int:
        """
        Build the complete semantic search index.

        This:
        1. Creates the embeddings table
        2. Generates embeddings for all markets
        3. Creates the vss index

        Args:
            batch_size: Number of markets to embed per API call

        Returns:
            Number of markets indexed
        """
        # Create table if it doesn't exist
        self.create_embeddings_table()

        # Get all markets that need embeddings
        markets_sql = """
            SELECT
                id,
                question,
                tags,
                description
            FROM v_markets
            WHERE id NOT IN (SELECT market_id FROM market_embeddings)
        """
        result = self.conn.execute(markets_sql)
        markets = result.fetchall()

        if not markets:
            logger.info("no_new_markets_to_embed")
            self.create_vss_index()
            return 0

        # Generate embeddings in batches
        total_embedded = 0
        for i in range(0, len(markets), batch_size):
            batch = markets[i : i + batch_size]

            # Prepare texts for embedding
            texts = [
                self._prepare_text_for_embedding(
                    question=row[1],
                    tags=row[2] if row[2] else None,
                    description=row[3] if row[3] else None,
                )
                for row in batch
            ]

            # Generate embeddings
            embeddings = self._generate_embeddings_batch(texts)

            # Insert into table
            for j, (market_id, _, _, _) in enumerate(batch):
                embedding = embeddings[j]
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO market_embeddings
                    (market_id, embedding, embedding_model, embedding_dim)
                    VALUES (?, ?, ?, ?)
                """,
                    [market_id, embedding, self.embedding_model, self.embedding_dimensions],
                )

            total_embedded += len(batch)
            logger.info(
                "embeddings_batch_completed",
                batch_num=i // batch_size + 1,
                markets_in_batch=len(batch),
                total_embedded=total_embedded,
            )

        # Create the vss index
        self.create_vss_index()

        logger.info(
            "semantic_index_built",
            total_markets=total_embedded,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
        )

        return total_embedded

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
        Search markets using semantic similarity.

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
            - event_id: Parent event ID
            - score: Cosine similarity score (0-1, higher is better)

        Raises:
            RuntimeError: If index has not been built
            ValueError: If OpenAI API key is not configured
        """
        if not self._index_created:
            # Check if index exists
            try:
                self.conn.execute(
                    "SELECT 1 FROM market_embeddings LIMIT 1"
                ).fetchone()
                # Load vss extension and mark as ready
                self.conn.execute("INSTALL vss")
                self.conn.execute("LOAD vss")
                self._index_created = True
            except Exception:
                raise RuntimeError(
                    "Semantic search index not built. Call build_index() first."
                )

        # Generate embedding for query
        query_embedding = self._generate_embedding(query)

        # Build filter conditions
        conditions: list[str] = []
        params: list[Any] = []

        if category:
            conditions.append("m.category = ?")
            params.append(category)

        if closed_time_min:
            conditions.append("m.closed_time >= ?")
            params.append(closed_time_min)

        if closed_time_max:
            conditions.append("m.closed_time <= ?")
            params.append(closed_time_max)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # First get total count
        count_sql = f"""
            SELECT COUNT(*)
            FROM market_embeddings e
            JOIN v_markets m ON e.market_id = m.id
            WHERE {where_clause}
        """
        count_result = self.conn.execute(count_sql, params).fetchone()
        total_count = count_result[0] if count_result else 0

        # Search using cosine similarity
        # DuckDB vss uses array_cosine_similarity for similarity calculation
        # We convert to a similarity score where higher is better
        search_sql = f"""
            SELECT
                m.id,
                m.question,
                m.tags,
                m.category,
                m.closed_time,
                m.event_id,
                array_cosine_similarity(e.embedding, ?::FLOAT[{self.embedding_dimensions}]) AS score
            FROM market_embeddings e
            JOIN v_markets m ON e.market_id = m.id
            WHERE {where_clause}
            ORDER BY score DESC, m.id ASC
            LIMIT ? OFFSET ?
        """
        search_params = [query_embedding] + params + [limit, offset]

        result = self.conn.execute(search_sql, search_params)
        columns = [desc[0] for desc in result.description]
        results = [dict(zip(columns, row)) for row in result.fetchall()]

        logger.info(
            "semantic_search_completed",
            query=query[:50] + "..." if len(query) > 50 else query,
            total_count=total_count,
            returned=len(results),
        )

        return results, total_count

    def refresh_index(self, batch_size: int = 100) -> int:
        """
        Refresh the semantic search index with current market data.

        This rebuilds embeddings for all markets.

        Args:
            batch_size: Number of markets to embed per API call

        Returns:
            Number of markets indexed
        """
        # Drop existing embeddings
        self.conn.execute("DELETE FROM market_embeddings")

        # Rebuild
        return self.build_index(batch_size=batch_size)

    def update_markets(self, market_ids: list[str], batch_size: int = 100) -> int:
        """
        Update embeddings for specific markets.

        This is more efficient than rebuild for incremental updates.

        Args:
            market_ids: List of market IDs that changed
            batch_size: Number of markets to embed per API call

        Returns:
            Number of markets updated
        """
        if not market_ids:
            return 0

        # Delete existing embeddings for these markets
        placeholders = ", ".join(["?" for _ in market_ids])
        self.conn.execute(
            f"DELETE FROM market_embeddings WHERE market_id IN ({placeholders})",
            market_ids,
        )

        # Get market data for these IDs
        markets_sql = f"""
            SELECT id, question, tags, description
            FROM v_markets
            WHERE id IN ({placeholders})
        """
        result = self.conn.execute(markets_sql, market_ids)
        markets = result.fetchall()

        if not markets:
            return 0

        # Generate embeddings in batches
        for i in range(0, len(markets), batch_size):
            batch = markets[i : i + batch_size]

            texts = [
                self._prepare_text_for_embedding(
                    question=row[1],
                    tags=row[2] if row[2] else None,
                    description=row[3] if row[3] else None,
                )
                for row in batch
            ]

            embeddings = self._generate_embeddings_batch(texts)

            for j, (market_id, _, _, _) in enumerate(batch):
                self.conn.execute(
                    """
                    INSERT INTO market_embeddings
                    (market_id, embedding, embedding_model, embedding_dim)
                    VALUES (?, ?, ?, ?)
                """,
                    [market_id, embeddings[j], self.embedding_model, self.embedding_dimensions],
                )

        # Note: vss index is automatically updated on insert

        logger.info("market_embeddings_updated", market_ids=len(market_ids))
        return len(market_ids)

    def get_embedding_stats(self) -> dict[str, Any]:
        """
        Get statistics about the embeddings index.

        Returns:
            Dictionary with embedding statistics
        """
        try:
            result = self.conn.execute("""
                SELECT
                    COUNT(*) as total_embeddings,
                    MAX(embedding_model) as embedding_model,
                    MAX(embedding_dim) as embedding_dim,
                    MIN(created_at) as first_created,
                    MAX(created_at) as last_created
                FROM market_embeddings
            """).fetchone()

            if result:
                return {
                    "total_embeddings": result[0],
                    "embedding_model": result[1],
                    "embedding_dim": result[2],
                    "first_created": result[3],
                    "last_created": result[4],
                }
        except Exception:
            pass

        return {
            "total_embeddings": 0,
            "embedding_model": None,
            "embedding_dim": None,
            "first_created": None,
            "last_created": None,
        }


def create_semantic_search_index(
    conn: duckdb.DuckDBPyConnection,
    openai_api_key: str,
    embedding_model: str = "text-embedding-3-small",
    embedding_dimensions: int = 1536,
    batch_size: int = 100,
) -> SemanticSearchIndex:
    """
    Create and build a semantic search index.

    Args:
        conn: DuckDB connection with v_markets view already created
        openai_api_key: OpenAI API key for generating embeddings
        embedding_model: OpenAI embedding model to use
        embedding_dimensions: Dimension of the embedding vectors
        batch_size: Number of markets to embed per API call

    Returns:
        Initialized and built SemanticSearchIndex
    """
    index = SemanticSearchIndex(
        conn=conn,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    index.build_index(batch_size=batch_size)
    return index
