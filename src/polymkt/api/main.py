"""FastAPI application for the Polymkt analytics platform."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from polymkt.config import settings
from polymkt.models.schemas import (
    BacktestCreateRequest,
    BacktestListResponse,
    BacktestSchema,
    BacktestSummary,
    BacktestUpdateRequest,
    BootstrapSummary,
    CurateSummary,
    DatasetCreateRequest,
    DatasetListResponse,
    DatasetSchema,
    DatasetSummary,
    DatasetUpdateRequest,
    EmbeddingStats,
    HybridIndexStats,
    HybridSearchResult,
    MarketSearchResult,
    RunRecord,
    SearchIndexUpdateResult,
    SearchIndexUpdaterStats,
    SemanticSearchResult,
    UnifiedMarketSearchResult,
    UpdateSummary,
)
from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.pipeline.curate import run_curate
from polymkt.pipeline.update import run_update
from polymkt.storage.backtests import BacktestNotFoundError, BacktestStore
from polymkt.storage.datasets import DatasetNotFoundError, DatasetStore
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.search import MarketSearchIndex
from polymkt.storage.hybrid_search import HybridSearchIndex
from polymkt.storage.semantic_search import SemanticSearchIndex
from polymkt.storage.search_index_updater import SearchIndexUpdater

app = FastAPI(
    title="Polymkt Analytics API",
    description="Prediction market analytics platform with DuckDB-over-Parquet storage",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class RunListResponse(BaseModel):
    """List of pipeline runs."""

    runs: list[RunRecord]


class TradesQueryRequest(BaseModel):
    """Request for querying trades."""

    market_id: str | None = None
    market_ids: list[str] | None = None
    start_time: str | None = None
    end_time: str | None = None
    limit: int = 1000
    offset: int = 0
    order_by: str = "timestamp"
    order_dir: str = "ASC"


class TradesWithMarketsQueryRequest(BaseModel):
    """Request for querying trades with market data and days_to_exp."""

    market_id: str | None = None
    market_ids: list[str] | None = None
    start_time: str | None = None
    end_time: str | None = None
    days_to_exp_min: float | None = None
    days_to_exp_max: float | None = None
    limit: int = 1000
    offset: int = 0
    order_by: str = "timestamp"
    order_dir: str = "ASC"


class TradesQueryResponse(BaseModel):
    """Response for trades query."""

    trades: list[dict[str, Any]]
    count: int
    total_count: int
    has_more: bool


class MarketSearchRequest(BaseModel):
    """Request for searching markets."""

    q: str
    limit: int = 50
    offset: int = 0
    category: str | None = None
    closed_time_min: str | None = None
    closed_time_max: str | None = None


class MarketSearchResponse(BaseModel):
    """Response for market search."""

    results: list[MarketSearchResult]
    count: int
    total_count: int
    has_more: bool


class BuildSearchIndexResponse(BaseModel):
    """Response for building search index."""

    status: str
    markets_indexed: int


class SemanticSearchResponse(BaseModel):
    """Response for semantic search."""

    results: list[SemanticSearchResult]
    count: int
    total_count: int
    has_more: bool


class BuildSemanticIndexResponse(BaseModel):
    """Response for building semantic search index."""

    status: str
    markets_indexed: int
    embedding_model: str
    embedding_dimensions: int


class HybridSearchResponse(BaseModel):
    """Response for hybrid search."""

    results: list[HybridSearchResult]
    count: int
    total_count: int
    has_more: bool


class BuildHybridIndexResponse(BaseModel):
    """Response for building hybrid search index."""

    status: str
    bm25_markets_indexed: int
    semantic_markets_indexed: int
    embedding_model: str | None
    embedding_dimensions: int | None


class UnifiedSearchResponse(BaseModel):
    """Response for unified market search supporting all modes."""

    results: list[UnifiedMarketSearchResult]
    count: int
    total_count: int
    has_more: bool
    mode: str  # bm25, semantic, or hybrid


def generate_snippet(question: str, description: str | None, query: str, max_length: int = 150) -> str:
    """
    Generate a snippet with query terms highlighted.

    The snippet is extracted from the question or description, with query terms
    wrapped in ** for emphasis.

    Args:
        question: Market question text
        description: Market description (optional)
        query: Search query string
        max_length: Maximum snippet length

    Returns:
        Snippet with highlighted terms
    """
    # Combine text sources, preferring question
    text = question
    if description and len(question) < max_length // 2:
        text = f"{question} - {description}"

    # Truncate if too long
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    # Highlight query terms (case-insensitive)
    query_terms = query.lower().split()
    for term in query_terms:
        if len(term) >= 2:  # Only highlight terms with 2+ characters
            import re

            # Case-insensitive replace with highlighted version
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"**{m.group(0)}**", text)

    return text


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/api/bootstrap", response_model=BootstrapSummary)
def bootstrap_import() -> BootstrapSummary:
    """
    Run the bootstrap import process.

    Imports existing CSV files (markets.csv, trades.csv, orderFilled.csv)
    into Parquet format and creates DuckDB views.
    """
    try:
        summary = run_bootstrap()
        return summary
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"CSV file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bootstrap failed: {e}")


@app.post("/api/curate", response_model=CurateSummary)
def curate_analytics() -> CurateSummary:
    """
    Run the curate step to build the analytics layer.

    Reads raw Parquet files and computes derived fields (e.g., days_to_exp)
    to create the analytics layer. The raw layer is NOT modified.
    """
    try:
        summary = run_curate()
        return summary
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Raw data not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Curate failed: {e}")


@app.post("/api/update", response_model=UpdateSummary)
def incremental_update() -> UpdateSummary:
    """
    Run an incremental update from the last watermark.

    This endpoint performs an incremental update that:
    1. Reads current watermarks for each entity (trades, markets, order_filled)
    2. Reads new data from CSV files (simulating upstream fetch)
    3. Filters data to only rows after the watermark timestamp
    4. Deduplicates using transaction_hash (trades/order_filled) or id (markets)
    5. Appends new rows to Parquet files (or upserts for markets)
    6. Updates watermarks to the max timestamp of new data
    7. Refreshes DuckDB views

    The runtime is proportional to new data, not total history.

    Response includes:
    - rows_read: Total rows read from source per entity
    - rows_written: New rows written per entity
    - rows_skipped: Rows skipped due to deduplication per entity
    - rows_updated: Rows updated (for markets upsert)
    - watermark_before/after: Watermark state before and after the update
    """
    try:
        summary = run_update()
        return summary
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")


@app.get("/api/runs", response_model=RunListResponse)
def list_runs(limit: int = 100) -> RunListResponse:
    """List pipeline run history."""
    metadata_store = MetadataStore(settings.metadata_db_path)
    runs = metadata_store.list_runs(limit=limit)
    return RunListResponse(runs=runs)


@app.get("/api/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    """Get details of a specific run."""
    metadata_store = MetadataStore(settings.metadata_db_path)
    run = metadata_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return run


@app.post("/api/query/trades", response_model=TradesQueryResponse)
def query_trades(request: TradesQueryRequest) -> TradesQueryResponse:
    """
    Query trades with optional filters.

    Supports filtering by single market_id, multiple market_ids,
    and time range (start_time, end_time).

    Results can be sorted using order_by (comma-separated columns) and order_dir.
    Use order_by="timestamp,transaction_hash" for stable deterministic ordering.

    Response includes total_count and has_more for proper pagination.
    """
    # Check if Parquet files exist
    if not (settings.parquet_dir / "trades.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Trades data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        # Ensure views exist
        duckdb_layer.create_views()

        trades, total_count = duckdb_layer.query_trades(
            market_id=request.market_id,
            market_ids=request.market_ids,
            start_time=request.start_time,
            end_time=request.end_time,
            limit=request.limit,
            offset=request.offset,
            order_by=request.order_by,
            order_dir=request.order_dir,
        )
        has_more = (request.offset + len(trades)) < total_count
        return TradesQueryResponse(
            trades=trades,
            count=len(trades),
            total_count=total_count,
            has_more=has_more,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        duckdb_layer.close()


@app.get("/api/watermarks")
def get_watermarks() -> dict[str, Any]:
    """Get all current watermarks."""
    metadata_store = MetadataStore(settings.metadata_db_path)
    return metadata_store.get_all_watermarks()


@app.post("/api/search/build", response_model=BuildSearchIndexResponse)
def build_search_index() -> BuildSearchIndexResponse:
    """
    Build or rebuild the BM25 full-text search index for markets.

    This creates a searchable index over market.question, market.tags
    (derived from events), and market.description.

    The index uses DuckDB's FTS extension with BM25 scoring for relevance ranking.
    """
    # Check if Parquet files exist
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        # Ensure views exist
        duckdb_layer.create_views()

        # Build the search index
        search_index = MarketSearchIndex(duckdb_layer.conn)
        markets_indexed = search_index.build_index()

        return BuildSearchIndexResponse(
            status="success",
            markets_indexed=markets_indexed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build search index: {e}")
    finally:
        duckdb_layer.close()


@app.get("/api/markets/search", response_model=UnifiedSearchResponse)
def search_markets(
    q: str,
    mode: str = "hybrid",
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    closed_time_min: str | None = None,
    closed_time_max: str | None = None,
) -> UnifiedSearchResponse:
    """
    Search markets with support for multiple search modes.

    This unified search endpoint supports BM25 full-text search, semantic
    vector search, and hybrid search (combining both with RRF scoring).

    Results are sorted by relevance score and then by market_id for stable
    pagination ordering.

    Args:
        q: Search query string (required)
        mode: Search mode - "bm25", "semantic", or "hybrid" (default: "hybrid")
        limit: Maximum results to return (default 50)
        offset: Number of results to skip for pagination (default 0)
        category: Filter by market category (optional)
        closed_time_min: Filter by minimum closed_time (optional)
        closed_time_max: Filter by maximum closed_time (optional)

    Returns:
        Paginated search results with relevance scores and snippets
    """
    # Validate mode parameter
    valid_modes = ["bm25", "semantic", "hybrid"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}",
        )

    # Check if Parquet files exist
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    # Semantic and hybrid modes require OpenAI API key
    if mode in ["semantic", "hybrid"] and not settings.openai_api_key:
        if mode == "semantic":
            raise HTTPException(
                status_code=400,
                detail="Semantic search requires OpenAI API key. Set POLYMKT_OPENAI_API_KEY environment variable.",
            )
        # For hybrid, fall back to BM25-only silently
        mode = "bm25"

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        # Ensure views exist
        duckdb_layer.create_views()

        results_raw: list[dict[str, Any]] = []
        total_count = 0

        if mode == "bm25":
            # BM25 full-text search
            search_index = MarketSearchIndex(duckdb_layer.conn)
            search_index.build_index()
            results_raw, total_count = search_index.search(
                query=q,
                limit=limit,
                offset=offset,
                category=category,
                closed_time_min=closed_time_min,
                closed_time_max=closed_time_max,
            )
        elif mode == "semantic":
            # Semantic vector search
            semantic_index = SemanticSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key=settings.openai_api_key,
                embedding_model=settings.openai_embedding_model,
                embedding_dimensions=settings.openai_embedding_dimensions,
            )
            results_raw, total_count = semantic_index.search(
                query=q,
                limit=limit,
                offset=offset,
                category=category,
                closed_time_min=closed_time_min,
                closed_time_max=closed_time_max,
            )
        else:  # hybrid
            # Hybrid BM25 + semantic search with RRF
            hybrid_index = HybridSearchIndex(
                conn=duckdb_layer.conn,
                openai_api_key=settings.openai_api_key,
                embedding_model=settings.openai_embedding_model,
                embedding_dimensions=settings.openai_embedding_dimensions,
            )
            results_raw, total_count = hybrid_index.search(
                query=q,
                limit=limit,
                offset=offset,
                category=category,
                closed_time_min=closed_time_min,
                closed_time_max=closed_time_max,
            )

        # Fetch descriptions for snippet generation
        market_ids = [r["id"] for r in results_raw]
        descriptions: dict[str, str | None] = {}
        if market_ids:
            placeholders = ", ".join([f"'{mid}'" for mid in market_ids])
            desc_result = duckdb_layer.conn.execute(f"""
                SELECT id, description FROM v_markets WHERE id IN ({placeholders})
            """).fetchall()
            descriptions = {row[0]: row[1] for row in desc_result}

        # Convert to unified response model with snippets
        results = []
        for r in results_raw:
            description = descriptions.get(r["id"])
            snippet = generate_snippet(r["question"], description, q)

            results.append(
                UnifiedMarketSearchResult(
                    id=r["id"],
                    question=r["question"],
                    tags=r.get("tags") if r.get("tags") else None,
                    category=r.get("category"),
                    closed_time=r.get("closed_time"),
                    event_id=r.get("event_id"),
                    relevance_score=r["score"],
                    snippet=snippet,
                    bm25_score=r.get("bm25_score") if mode == "hybrid" else None,
                    semantic_score=r.get("semantic_score") if mode == "hybrid" else None,
                )
            )

        has_more = (offset + len(results)) < total_count
        return UnifiedSearchResponse(
            results=results,
            count=len(results),
            total_count=total_count,
            has_more=has_more,
            mode=mode,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    finally:
        duckdb_layer.close()


@app.post("/api/query/trades_with_markets", response_model=TradesQueryResponse)
def query_trades_with_markets(
    request: TradesWithMarketsQueryRequest,
) -> TradesQueryResponse:
    """
    Query trades joined with market data, including the derived days_to_exp field.

    The days_to_exp field is computed as:
        (market.closed_time - trade.timestamp) in days

    This endpoint supports filtering by days_to_exp range, which is useful for
    backtesting strategies like "buy at 90 days to expiry".

    Example: Filter trades between 89-91 days to expiry:
        {"days_to_exp_min": 89, "days_to_exp_max": 91}
    """
    # Check if Parquet files exist
    if not (settings.parquet_dir / "trades.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Trades data not available. Run bootstrap first.",
        )
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        # Ensure views exist
        duckdb_layer.create_views()

        trades, total_count = duckdb_layer.query_trades_with_markets(
            market_id=request.market_id,
            market_ids=request.market_ids,
            start_time=request.start_time,
            end_time=request.end_time,
            days_to_exp_min=request.days_to_exp_min,
            days_to_exp_max=request.days_to_exp_max,
            limit=request.limit,
            offset=request.offset,
            order_by=request.order_by,
            order_dir=request.order_dir,
        )
        has_more = (request.offset + len(trades)) < total_count
        return TradesQueryResponse(
            trades=trades,
            count=len(trades),
            total_count=total_count,
            has_more=has_more,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        duckdb_layer.close()


@app.post("/api/semantic-search/build", response_model=BuildSemanticIndexResponse)
def build_semantic_index() -> BuildSemanticIndexResponse:
    """
    Build or rebuild the semantic search index for markets using OpenAI embeddings.

    This creates embeddings for all markets using the configured OpenAI embedding model
    (default: text-embedding-3-small) and stores them in DuckDB with a vss (vector
    similarity search) index for efficient approximate nearest neighbor queries.

    Requires POLYMKT_OPENAI_API_KEY environment variable to be set.
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key not configured. Set POLYMKT_OPENAI_API_KEY environment variable.",
        )

    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        semantic_index = SemanticSearchIndex(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )
        markets_indexed = semantic_index.build_index()

        return BuildSemanticIndexResponse(
            status="success",
            markets_indexed=markets_indexed,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build semantic index: {e}")
    finally:
        duckdb_layer.close()


@app.get("/api/markets/semantic-search", response_model=SemanticSearchResponse)
def semantic_search_markets(
    q: str,
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    closed_time_min: str | None = None,
    closed_time_max: str | None = None,
) -> SemanticSearchResponse:
    """
    Search markets using semantic similarity with OpenAI embeddings.

    This endpoint uses vector similarity search to find markets semantically
    similar to the query, even when exact keywords are not present.

    For example, searching "voter turnout prediction" will find relevant
    election markets even if they don't contain those exact words.

    Args:
        q: Search query string (required)
        limit: Maximum results to return (default 50)
        offset: Number of results to skip for pagination (default 0)
        category: Filter by market category (optional)
        closed_time_min: Filter by minimum closed_time (optional)
        closed_time_max: Filter by maximum closed_time (optional)

    Returns:
        Paginated search results with cosine similarity scores (0-1, higher is better)
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key not configured. Set POLYMKT_OPENAI_API_KEY environment variable.",
        )

    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        semantic_index = SemanticSearchIndex(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )

        results_raw, total_count = semantic_index.search(
            query=q,
            limit=limit,
            offset=offset,
            category=category,
            closed_time_min=closed_time_min,
            closed_time_max=closed_time_max,
        )

        results = [
            SemanticSearchResult(
                id=r["id"],
                question=r["question"],
                tags=r["tags"] if r["tags"] else None,
                category=r["category"],
                closed_time=r["closed_time"],
                event_id=r["event_id"],
                score=r["score"],
            )
            for r in results_raw
        ]

        has_more = (offset + len(results)) < total_count
        return SemanticSearchResponse(
            results=results,
            count=len(results),
            total_count=total_count,
            has_more=has_more,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {e}")
    finally:
        duckdb_layer.close()


@app.get("/api/semantic-search/stats", response_model=EmbeddingStats)
def get_embedding_stats() -> EmbeddingStats:
    """
    Get statistics about the semantic search embeddings index.

    Returns information about the current state of the embeddings table,
    including total count, model used, and creation timestamps.
    """
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        semantic_index = SemanticSearchIndex(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )

        stats = semantic_index.get_embedding_stats()

        return EmbeddingStats(
            total_embeddings=stats["total_embeddings"],
            embedding_model=stats["embedding_model"],
            embedding_dim=stats["embedding_dim"],
            first_created=stats["first_created"],
            last_created=stats["last_created"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding stats: {e}")
    finally:
        duckdb_layer.close()


@app.post("/api/hybrid-search/build", response_model=BuildHybridIndexResponse)
def build_hybrid_index() -> BuildHybridIndexResponse:
    """
    Build or rebuild the hybrid search index (both BM25 and semantic).

    This creates:
    1. BM25 full-text search index over question, tags, and description
    2. Semantic search index using OpenAI embeddings (if API key configured)

    If OpenAI API key is not configured, only BM25 index is built.
    """
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        hybrid_index = HybridSearchIndex(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )

        # Always build BM25
        bm25_count = hybrid_index.build_bm25_index()

        # Build semantic if API key available
        semantic_count = 0
        if settings.openai_api_key:
            semantic_count = hybrid_index.build_semantic_index()

        return BuildHybridIndexResponse(
            status="success",
            bm25_markets_indexed=bm25_count,
            semantic_markets_indexed=semantic_count,
            embedding_model=settings.openai_embedding_model if settings.openai_api_key else None,
            embedding_dimensions=settings.openai_embedding_dimensions if settings.openai_api_key else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build hybrid index: {e}")
    finally:
        duckdb_layer.close()


@app.get("/api/markets/hybrid-search", response_model=HybridSearchResponse)
def hybrid_search_markets(
    q: str,
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    closed_time_min: str | None = None,
    closed_time_max: str | None = None,
) -> HybridSearchResponse:
    """
    Search markets using hybrid BM25 + semantic search.

    This endpoint combines results from BM25 full-text search and semantic
    vector search using Reciprocal Rank Fusion (RRF). This provides better
    results for both keyword-heavy and semantic-heavy queries.

    The hybrid approach:
    1. Retrieves top-K results from both BM25 and semantic search
    2. Merges candidates using RRF scoring
    3. Returns a single ranked list sorted by combined relevance

    Note: If OpenAI API key is not configured, only BM25 results are returned.

    Args:
        q: Search query string (required)
        limit: Maximum results to return (default 50)
        offset: Number of results to skip for pagination (default 0)
        category: Filter by market category (optional)
        closed_time_min: Filter by minimum closed_time (optional)
        closed_time_max: Filter by maximum closed_time (optional)

    Returns:
        Paginated search results with hybrid RRF scores
    """
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        hybrid_index = HybridSearchIndex(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )

        results_raw, total_count = hybrid_index.search(
            query=q,
            limit=limit,
            offset=offset,
            category=category,
            closed_time_min=closed_time_min,
            closed_time_max=closed_time_max,
        )

        results = [
            HybridSearchResult(
                id=r["id"],
                question=r["question"],
                tags=r.get("tags"),
                category=r.get("category"),
                closed_time=r.get("closed_time"),
                event_id=r.get("event_id"),
                score=r["score"],
                bm25_score=r.get("bm25_score"),
                semantic_score=r.get("semantic_score"),
                bm25_rank=r.get("bm25_rank"),
                semantic_rank=r.get("semantic_rank"),
            )
            for r in results_raw
        ]

        has_more = (offset + len(results)) < total_count
        return HybridSearchResponse(
            results=results,
            count=len(results),
            total_count=total_count,
            has_more=has_more,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")
    finally:
        duckdb_layer.close()


@app.get("/api/hybrid-search/stats", response_model=HybridIndexStats)
def get_hybrid_search_stats() -> HybridIndexStats:
    """
    Get statistics about the hybrid search indices.

    Returns information about the current state of both BM25 and semantic
    search indices.
    """
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        hybrid_index = HybridSearchIndex(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )

        stats = hybrid_index.get_index_stats()

        return HybridIndexStats(
            bm25_available=stats["bm25_available"],
            semantic_available=stats["semantic_available"],
            bm25_markets_indexed=stats.get("bm25_markets_indexed"),
            semantic_markets_indexed=stats.get("semantic_markets_indexed"),
            semantic_embedding_model=stats.get("semantic_embedding_model"),
            semantic_embedding_dim=stats.get("semantic_embedding_dim"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hybrid search stats: {e}")
    finally:
        duckdb_layer.close()


@app.post("/api/search-index/update", response_model=SearchIndexUpdateResult)
def update_search_indices(
    force_rebuild: bool = False,
) -> SearchIndexUpdateResult:
    """
    Update search indices incrementally based on changed markets and events.

    This endpoint detects which markets have changed (new, modified, or deleted)
    by comparing content hashes of question + tags + description, and also detects
    event tag changes that affect market search indexing.

    Markets are updated in both BM25 and semantic indices when:
    - The market's own content (question, description) changes
    - The event's tags change (which affects the market's derived tags)

    This is more efficient than rebuilding the entire index when only a few
    markets or events have changed.

    Args:
        force_rebuild: If True, rebuild all indices from scratch instead of
                      doing an incremental update. Use this if the indices
                      are corrupted or out of sync.

    Returns:
        Update statistics including number of markets and events updated.
    """
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        updater = SearchIndexUpdater(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )

        result = updater.update_indices(force_rebuild=force_rebuild)

        # Normalize field names (force_rebuild returns bm25_indexed, incremental returns bm25_updated)
        bm25_count = result.get("bm25_updated", 0) or result.get("bm25_indexed", 0)
        semantic_count = result.get("semantic_updated", 0) or result.get("semantic_indexed", 0)

        # Determine status based on results
        total_updated = bm25_count + semantic_count
        has_event_changes = (
            result.get("new_events", 0) > 0
            or result.get("changed_events", 0) > 0
            or result.get("deleted_events", 0) > 0
        )
        if result["mode"] == "full_rebuild":
            status = "success"
        elif total_updated > 0 or has_event_changes:
            status = "success"
        else:
            status = "no_changes"

        return SearchIndexUpdateResult(
            status=status,
            mode=result["mode"],
            bm25_updated=bm25_count,
            semantic_updated=semantic_count,
            new_markets=result.get("new_markets", 0),
            changed_markets=result.get("changed_markets", 0),
            deleted_markets=result.get("deleted_markets", 0),
            new_events=result.get("new_events", 0),
            changed_events=result.get("changed_events", 0),
            deleted_events=result.get("deleted_events", 0),
            event_affected_markets=result.get("event_affected_markets", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update search indices: {e}")
    finally:
        duckdb_layer.close()


@app.get("/api/search-index/stats", response_model=SearchIndexUpdaterStats)
def get_search_index_updater_stats() -> SearchIndexUpdaterStats:
    """
    Get statistics about the search index updater.

    Returns information about content hash tracking for both markets and events,
    as well as BM25 and semantic search index statistics.
    """
    if not (settings.parquet_dir / "markets.parquet").exists():
        raise HTTPException(
            status_code=400,
            detail="Markets data not available. Run bootstrap first.",
        )

    duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
    try:
        duckdb_layer.create_views()

        updater = SearchIndexUpdater(
            conn=duckdb_layer.conn,
            openai_api_key=settings.openai_api_key,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.openai_embedding_dimensions,
        )

        stats = updater.get_stats()

        return SearchIndexUpdaterStats(
            total_hashes=stats.get("total_hashes", 0),
            first_updated=stats.get("first_updated"),
            last_updated=stats.get("last_updated"),
            total_event_hashes=stats.get("total_event_hashes", 0),
            event_first_updated=stats.get("event_first_updated"),
            event_last_updated=stats.get("event_last_updated"),
            bm25_available=stats.get("bm25_available", False),
            semantic_available=stats.get("semantic_available", False),
            bm25_markets_indexed=stats.get("bm25_markets_indexed"),
            semantic_markets_indexed=stats.get("semantic_markets_indexed"),
            semantic_embedding_model=stats.get("semantic_embedding_model"),
            semantic_embedding_dim=stats.get("semantic_embedding_dim"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search index stats: {e}")
    finally:
        duckdb_layer.close()


# =============================================================================
# Dataset CRUD endpoints for persisting market sets with filters and market lists
# =============================================================================


@app.post("/api/datasets", response_model=DatasetSchema, status_code=201)
def create_dataset(request: DatasetCreateRequest) -> DatasetSchema:
    """
    Create a new dataset with a set of markets.

    A dataset is a saved collection of market IDs that can be reused for
    backtesting and analysis. Datasets can be created by:
    1. Explicit market list (via search, filters, or manual selection)
    2. Filters that define the selection criteria

    The market_ids list contains the actual markets to include. The filters
    field records how the dataset was created (for reference/reproduction).

    Args:
        request: Dataset creation request with name, description, market_ids,
                and optional filters/excluded_market_ids

    Returns:
        The created dataset with generated ID and timestamps
    """
    try:
        dataset_store = DatasetStore(settings.metadata_db_path)
        return dataset_store.create_dataset(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e}")


@app.get("/api/datasets", response_model=DatasetListResponse)
def list_datasets(limit: int = 50, offset: int = 0) -> DatasetListResponse:
    """
    List all datasets with pagination.

    Returns dataset summaries (id, name, description, market_count, timestamps)
    ordered by most recently updated first.

    Args:
        limit: Maximum number of datasets to return (default 50)
        offset: Number of datasets to skip for pagination (default 0)

    Returns:
        Paginated list of dataset summaries with total_count and has_more
    """
    try:
        dataset_store = DatasetStore(settings.metadata_db_path)
        summaries, total_count = dataset_store.list_datasets(limit=limit, offset=offset)
        has_more = (offset + len(summaries)) < total_count
        return DatasetListResponse(
            datasets=summaries,
            count=len(summaries),
            total_count=total_count,
            has_more=has_more,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}")


@app.get("/api/datasets/{dataset_id}", response_model=DatasetSchema)
def get_dataset(dataset_id: str) -> DatasetSchema:
    """
    Get a dataset by ID.

    Returns the full dataset including all market_ids and filters.

    Args:
        dataset_id: The dataset UUID

    Returns:
        The dataset with all fields
    """
    try:
        dataset_store = DatasetStore(settings.metadata_db_path)
        return dataset_store.get_dataset(dataset_id)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e}")


@app.put("/api/datasets/{dataset_id}", response_model=DatasetSchema)
def update_dataset(dataset_id: str, request: DatasetUpdateRequest) -> DatasetSchema:
    """
    Update an existing dataset.

    Only provided fields are updated; omitted fields retain their existing values.
    Use this endpoint to:
    - Rename a dataset
    - Update the description
    - Modify the market list (add/remove markets)
    - Update excluded_market_ids

    Args:
        dataset_id: The dataset UUID
        request: Fields to update (only provided fields are modified)

    Returns:
        The updated dataset
    """
    try:
        dataset_store = DatasetStore(settings.metadata_db_path)
        return dataset_store.update_dataset(dataset_id, request)
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update dataset: {e}")


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict[str, str]:
    """
    Delete a dataset.

    This permanently removes the dataset. Any backtests that reference this
    dataset will retain their own copies of the market list.

    Args:
        dataset_id: The dataset UUID

    Returns:
        Confirmation message
    """
    try:
        dataset_store = DatasetStore(settings.metadata_db_path)
        dataset_store.delete_dataset(dataset_id)
        return {"status": "deleted", "dataset_id": dataset_id}
    except DatasetNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}")


# =============================================================================
# Backtest CRUD endpoints for persisting backtest runs and results
# =============================================================================


@app.post("/api/backtests", response_model=BacktestSchema, status_code=201)
def create_backtest(request: BacktestCreateRequest) -> BacktestSchema:
    """
    Create a new backtest for a dataset with a strategy configuration.

    A backtest represents a strategy execution on a dataset. The backtest
    starts in 'pending' status and can be updated with results once execution
    completes.

    Args:
        request: Backtest creation request with dataset_id and strategy_config

    Returns:
        The created backtest with generated ID and timestamps
    """
    try:
        backtest_store = BacktestStore(settings.metadata_db_path)
        return backtest_store.create_backtest(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create backtest: {e}")


@app.get("/api/backtests", response_model=BacktestListResponse)
def list_backtests(
    limit: int = 50,
    offset: int = 0,
    dataset_id: str | None = None,
) -> BacktestListResponse:
    """
    List backtests with pagination.

    Returns backtest summaries ordered by most recently created first.
    Optionally filter by dataset_id to see all backtests for a specific dataset.

    Args:
        limit: Maximum number of backtests to return (default 50)
        offset: Number of backtests to skip for pagination (default 0)
        dataset_id: Optional dataset ID to filter by

    Returns:
        Paginated list of backtest summaries with total_count and has_more
    """
    try:
        backtest_store = BacktestStore(settings.metadata_db_path)
        summaries, total_count = backtest_store.list_backtests(
            limit=limit, offset=offset, dataset_id=dataset_id
        )
        has_more = (offset + len(summaries)) < total_count
        return BacktestListResponse(
            backtests=summaries,
            count=len(summaries),
            total_count=total_count,
            has_more=has_more,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {e}")


@app.get("/api/backtests/{backtest_id}", response_model=BacktestSchema)
def get_backtest(backtest_id: str) -> BacktestSchema:
    """
    Get a backtest by ID.

    Returns the full backtest including strategy config, metrics, trades,
    and equity curve (if completed).

    Args:
        backtest_id: The backtest UUID

    Returns:
        The backtest with all fields
    """
    try:
        backtest_store = BacktestStore(settings.metadata_db_path)
        return backtest_store.get_backtest(backtest_id)
    except BacktestNotFoundError:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest: {e}")


@app.put("/api/backtests/{backtest_id}", response_model=BacktestSchema)
def update_backtest(backtest_id: str, request: BacktestUpdateRequest) -> BacktestSchema:
    """
    Update an existing backtest.

    Use this endpoint to:
    - Update status (pending -> running -> completed/failed)
    - Add metrics after execution
    - Add trade records
    - Add equity curve data
    - Add error message if failed

    Only provided fields are updated; omitted fields retain their existing values.

    Args:
        backtest_id: The backtest UUID
        request: Fields to update (only provided fields are modified)

    Returns:
        The updated backtest
    """
    try:
        backtest_store = BacktestStore(settings.metadata_db_path)
        return backtest_store.update_backtest(backtest_id, request)
    except BacktestNotFoundError:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update backtest: {e}")


@app.delete("/api/backtests/{backtest_id}")
def delete_backtest(backtest_id: str) -> dict[str, str]:
    """
    Delete a backtest.

    This permanently removes the backtest and all its results.

    Args:
        backtest_id: The backtest UUID

    Returns:
        Confirmation message
    """
    try:
        backtest_store = BacktestStore(settings.metadata_db_path)
        backtest_store.delete_backtest(backtest_id)
        return {"status": "deleted", "backtest_id": backtest_id}
    except BacktestNotFoundError:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete backtest: {e}")
