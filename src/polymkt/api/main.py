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
    DataQualityCheckRequest,
    DataQualityReportListResponse,
    DataQualityReportSchema,
    DatasetCreateRequest,
    DatasetListResponse,
    DatasetSchema,
    DatasetSummary,
    DatasetUpdateRequest,
    ElectionGroupCreateRequest,
    ElectionGroupImportResult,
    ElectionGroupListResponse,
    ElectionGroupSchema,
    ElectionGroupSummary,
    ElectionGroupUpdateRequest,
    ElectionGroupValidationResult,
    EmbeddingStats,
    FavoriteComputeRequest,
    FavoriteComputeResultSchema,
    FavoriteSignalListResponse,
    FavoriteSignalSchema,
    FavoriteSnapshotSummary,
    HybridIndexStats,
    HybridSearchResult,
    MarketSearchResult,
    CompressionRatioSchema,
    CostEfficiencyReportSchema,
    InfrastructureRequirementsSchema,
    PerformanceBenchmarkReportSchema,
    PerformanceBenchmarkRequest,
    QueryBenchmarkResultSchema,
    StorageFootprintSchema,
    RangeIssueSchema,
    ReferentialIntegrityIssueSchema,
    RunRecord,
    SearchIndexUpdateResult,
    SearchIndexUpdaterStats,
    SemanticSearchResult,
    UnifiedMarketSearchResult,
    UniquenessIssueSchema,
    UnmappedMarketsResult,
    UpdateSummary,
)
from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.pipeline.curate import run_curate
from polymkt.pipeline.update import run_update
from polymkt.storage.backtests import BacktestNotFoundError, BacktestStore
from polymkt.storage.datasets import DatasetNotFoundError, DatasetStore
from polymkt.storage.election_groups import ElectionGroupNotFoundError, ElectionGroupStore
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.search import MarketSearchIndex
from polymkt.storage.hybrid_search import HybridSearchIndex
from polymkt.storage.semantic_search import SemanticSearchIndex
from polymkt.storage.search_index_updater import SearchIndexUpdater
from polymkt.signals.favorites import (
    FavoriteSignalStore,
    compute_favorites_for_groups,
)
from polymkt.backtest.engine import BacktestEngine
from polymkt.storage.data_quality import DataQualityChecker
from polymkt.storage.performance import PerformanceBenchmarker
from polymkt.storage.cost_efficiency import CostEfficiencyAnalyzer

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


@app.post("/api/backtests/{backtest_id}/execute", response_model=BacktestSchema)
def execute_backtest(backtest_id: str) -> BacktestSchema:
    """
    Execute a pending backtest.

    Runs the "buy the favorite" strategy on the backtest's dataset:
    1. Loads the dataset markets
    2. Loads pre-computed favorite signals at the entry days-to-exp (default 90)
    3. For each election group with a favorite in the dataset:
       - Enters at the snapshot price
       - Exits at market expiry (resolved price 0 or 1)
       - Calculates PnL with fees and slippage
    4. Computes aggregate metrics
    5. Updates the backtest with results

    Prerequisites:
    - Backtest must be in 'pending' status
    - Dataset must exist with markets
    - Election groups must be defined for the markets
    - Favorite signals must be computed (POST /api/favorite-signals/compute)

    Args:
        backtest_id: The backtest UUID to execute

    Returns:
        The completed backtest with metrics, trades, and equity curve
    """
    duckdb_layer: DuckDBLayer | None = None
    try:
        # Create all required stores
        duckdb_layer = DuckDBLayer(
            settings.duckdb_path,
            settings.parquet_dir,
            partitioned=settings.parquet_partitioning_enabled,
        )
        duckdb_layer.create_views()

        dataset_store = DatasetStore(settings.metadata_db_path)
        backtest_store = BacktestStore(settings.metadata_db_path)
        election_group_store = ElectionGroupStore(settings.metadata_db_path)
        favorite_signal_store = FavoriteSignalStore(settings.metadata_db_path)

        # Create engine and execute
        engine = BacktestEngine(
            duckdb_layer=duckdb_layer,
            dataset_store=dataset_store,
            backtest_store=backtest_store,
            election_group_store=election_group_store,
            favorite_signal_store=favorite_signal_store,
        )

        result = engine.execute(backtest_id)

        if result.status == "failed":
            raise HTTPException(
                status_code=500, detail=f"Backtest execution failed: {result.error_message}"
            )

        # Return the updated backtest
        return backtest_store.get_backtest(backtest_id)

    except BacktestNotFoundError:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute backtest: {e}")
    finally:
        if duckdb_layer:
            duckdb_layer.close()


# =============================================================================
# Election Group CRUD endpoints for grouping related markets
# =============================================================================


@app.post("/api/election-groups", response_model=ElectionGroupSchema, status_code=201)
def create_election_group(request: ElectionGroupCreateRequest) -> ElectionGroupSchema:
    """
    Create a new election group with a set of related markets.

    An election group is a collection of related markets (e.g., candidates
    in an election race). Groups are used to compute "buy the favorite"
    strategies by comparing prices across markets within the same group.

    Groups should have at least 2 markets to make favorite determination
    meaningful.

    Args:
        request: Group creation request with name, description, and market_ids

    Returns:
        The created election group with generated ID and timestamps
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        result = group_store.create_group(
            name=request.name,
            description=request.description,
            market_ids=request.market_ids,
        )
        return ElectionGroupSchema(
            id=result["id"],
            name=result["name"],
            description=result["description"],
            market_ids=result["market_ids"],
            created_at=result["created_at"],
            updated_at=result["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create election group: {e}")


@app.get("/api/election-groups", response_model=ElectionGroupListResponse)
def list_election_groups(limit: int = 50, offset: int = 0) -> ElectionGroupListResponse:
    """
    List all election groups with pagination.

    Returns group summaries (id, name, description, market_count, timestamps)
    ordered by most recently updated first.

    Args:
        limit: Maximum number of groups to return (default 50)
        offset: Number of groups to skip for pagination (default 0)

    Returns:
        Paginated list of group summaries with total_count and has_more
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        groups, total_count = group_store.list_groups(limit=limit, offset=offset)
        summaries = [
            ElectionGroupSummary(
                id=g["id"],
                name=g["name"],
                description=g["description"],
                market_count=g["market_count"],
                created_at=g["created_at"],
                updated_at=g["updated_at"],
            )
            for g in groups
        ]
        has_more = (offset + len(summaries)) < total_count
        return ElectionGroupListResponse(
            groups=summaries,
            count=len(summaries),
            total_count=total_count,
            has_more=has_more,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list election groups: {e}")


@app.get("/api/election-groups/{group_id}", response_model=ElectionGroupSchema)
def get_election_group(group_id: str) -> ElectionGroupSchema:
    """
    Get an election group by ID.

    Returns the full election group including all market_ids.

    Args:
        group_id: The election group UUID

    Returns:
        The election group with all fields
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        result = group_store.get_group(group_id)
        return ElectionGroupSchema(
            id=result["id"],
            name=result["name"],
            description=result["description"],
            market_ids=result["market_ids"],
            created_at=result["created_at"],
            updated_at=result["updated_at"],
        )
    except ElectionGroupNotFoundError:
        raise HTTPException(status_code=404, detail=f"Election group not found: {group_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get election group: {e}")


@app.put("/api/election-groups/{group_id}", response_model=ElectionGroupSchema)
def update_election_group(
    group_id: str, request: ElectionGroupUpdateRequest
) -> ElectionGroupSchema:
    """
    Update an existing election group.

    Only provided fields are updated; omitted fields retain their existing values.
    Use this endpoint to:
    - Rename a group
    - Update the description
    - Modify the market list (add/remove markets)

    Args:
        group_id: The election group UUID
        request: Fields to update (only provided fields are modified)

    Returns:
        The updated election group
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        result = group_store.update_group(
            group_id,
            name=request.name,
            description=request.description,
            market_ids=request.market_ids,
        )
        return ElectionGroupSchema(
            id=result["id"],
            name=result["name"],
            description=result["description"],
            market_ids=result["market_ids"],
            created_at=result["created_at"],
            updated_at=result["updated_at"],
        )
    except ElectionGroupNotFoundError:
        raise HTTPException(status_code=404, detail=f"Election group not found: {group_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update election group: {e}")


@app.delete("/api/election-groups/{group_id}")
def delete_election_group(group_id: str) -> dict[str, str]:
    """
    Delete an election group.

    This permanently removes the group and all its market mappings.
    Backtests that used this group will retain their own copies of the data.

    Args:
        group_id: The election group UUID

    Returns:
        Confirmation message
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        group_store.delete_group(group_id)
        return {"status": "deleted", "group_id": group_id}
    except ElectionGroupNotFoundError:
        raise HTTPException(status_code=404, detail=f"Election group not found: {group_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete election group: {e}")


class MarketIdsRequest(BaseModel):
    """Request body containing a list of market IDs."""

    market_ids: list[str]


@app.post("/api/election-groups/{group_id}/markets")
def add_markets_to_election_group(
    group_id: str, request: MarketIdsRequest
) -> dict[str, Any]:
    """
    Add markets to an existing election group.

    This is useful for incrementally building groups without replacing
    all existing market mappings.

    Args:
        group_id: The election group UUID
        request: Request body with market_ids list

    Returns:
        Number of markets added (excludes duplicates)
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        added = group_store.add_markets_to_group(group_id, request.market_ids)
        return {"status": "success", "markets_added": added, "group_id": group_id}
    except ElectionGroupNotFoundError:
        raise HTTPException(status_code=404, detail=f"Election group not found: {group_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add markets to group: {e}")


@app.post("/api/election-groups/{group_id}/markets/remove")
def remove_markets_from_election_group(
    group_id: str, request: MarketIdsRequest
) -> dict[str, Any]:
    """
    Remove markets from an existing election group.

    Args:
        group_id: The election group UUID
        request: Request body with market_ids list

    Returns:
        Number of markets removed
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        removed = group_store.remove_markets_from_group(group_id, request.market_ids)
        return {"status": "success", "markets_removed": removed, "group_id": group_id}
    except ElectionGroupNotFoundError:
        raise HTTPException(status_code=404, detail=f"Election group not found: {group_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove markets from group: {e}")


@app.post("/api/election-groups/import/csv", response_model=ElectionGroupImportResult)
def import_election_groups_from_csv(csv_path: str) -> ElectionGroupImportResult:
    """
    Import election group mappings from a CSV file.

    The CSV should have columns:
    - election_group_id (required): The group ID
    - market_id (required): The market ID to map
    - election_group_name (optional): Name for new groups
    - election_group_description (optional): Description for new groups

    New groups are created automatically when a new election_group_id is
    encountered. Existing groups have markets added to them.

    Args:
        csv_path: Absolute path to the CSV file

    Returns:
        Import summary with counts of groups created and markets mapped
    """
    from pathlib import Path

    path = Path(csv_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        result = group_store.import_from_csv(path)
        return ElectionGroupImportResult(
            groups_created=result["groups_created"],
            markets_mapped=result["markets_mapped"],
            errors=result["errors"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import from CSV: {e}")


@app.post("/api/election-groups/import/json", response_model=ElectionGroupImportResult)
def import_election_groups_from_json(json_path: str) -> ElectionGroupImportResult:
    """
    Import election group mappings from a JSON file.

    The JSON should be a list of objects with structure:
    [
        {
            "id": "group-uuid" (optional, generated if missing),
            "name": "Group Name" (required),
            "description": "Optional description",
            "market_ids": ["market1", "market2", ...]
        }
    ]

    Args:
        json_path: Absolute path to the JSON file

    Returns:
        Import summary with counts of groups created and markets mapped
    """
    from pathlib import Path

    path = Path(json_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"JSON file not found: {json_path}")

    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        result = group_store.import_from_json(path)
        return ElectionGroupImportResult(
            groups_created=result["groups_created"],
            markets_mapped=result["markets_mapped"],
            errors=result["errors"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import from JSON: {e}")


@app.post("/api/election-groups/validate", response_model=ElectionGroupValidationResult)
def validate_election_groups(min_markets: int = 2) -> ElectionGroupValidationResult:
    """
    Validate all election groups and report issues.

    Checks that each group has at least `min_markets` markets, which is
    required for meaningful favorite determination.

    Args:
        min_markets: Minimum markets required per group (default 2)

    Returns:
        Validation report with any issues found
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        result = group_store.validate_groups(min_markets=min_markets)
        return ElectionGroupValidationResult(
            total_groups=result["total_groups"],
            valid_groups=result["valid_groups"],
            invalid_groups=result["invalid_groups"],
            issues=result["issues"],
            min_markets_required=result["min_markets_required"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate groups: {e}")


@app.post("/api/election-groups/unmapped", response_model=UnmappedMarketsResult)
def find_unmapped_markets(request: MarketIdsRequest) -> UnmappedMarketsResult:
    """
    Find markets from a list that are not in any election group.

    This is useful for identifying markets that need to be assigned to
    groups before running "buy the favorite" backtests.

    Args:
        request: Request body with market_ids list

    Returns:
        List of unmapped market IDs with counts
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        unmapped = group_store.find_unmapped_markets(request.market_ids)
        return UnmappedMarketsResult(
            unmapped_market_ids=unmapped,
            total_checked=len(request.market_ids),
            unmapped_count=len(unmapped),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find unmapped markets: {e}")


@app.get("/api/markets/{market_id}/election-group")
def get_election_group_for_market(market_id: str) -> ElectionGroupSchema | dict[str, str]:
    """
    Get the election group for a specific market.

    Args:
        market_id: The market ID to look up

    Returns:
        The election group containing the market, or a message if not in any group
    """
    try:
        group_store = ElectionGroupStore(settings.metadata_db_path)
        result = group_store.get_group_for_market(market_id)
        if result is None:
            return {"status": "not_found", "message": f"Market {market_id} is not in any election group"}
        return ElectionGroupSchema(
            id=result["id"],
            name=result["name"],
            description=result["description"],
            market_ids=result["market_ids"],
            created_at=result["created_at"],
            updated_at=result["updated_at"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get election group for market: {e}")


# =============================================================================
# Favorite Signal Endpoints
# =============================================================================


@app.post("/api/favorite-signals/compute", response_model=FavoriteComputeResultSchema)
def compute_favorite_signals(request: FavoriteComputeRequest) -> FavoriteComputeResultSchema:
    """
    Compute favorite signals for election groups at a specified snapshot.

    For each election group:
    1. Get all market IDs in the group
    2. Query trades at the snapshot time (target_days_to_exp +/- tolerance)
    3. For each market, get the last trade price before snapshot
    4. Select the favorite as the market with highest YES price
    5. Handle ties deterministically (by market_id alphabetically)

    The computed signals are persisted to a SQLite table for use in backtests.

    Args:
        request: Compute request with target_days_to_exp, tolerance, optional group_ids

    Returns:
        Compute result with statistics about signals computed
    """
    try:
        # Check that parquet files exist
        if not settings.parquet_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="Parquet data not found. Run bootstrap first.",
            )

        # Initialize stores
        duckdb_layer = DuckDBLayer(settings.duckdb_path, settings.parquet_dir)
        duckdb_layer.create_views()
        election_group_store = ElectionGroupStore(settings.metadata_db_path)
        favorite_store = FavoriteSignalStore(settings.metadata_db_path)

        # Clear existing signals if requested
        if request.clear_existing:
            favorite_store.clear_signals_for_snapshot(
                request.target_days_to_exp,
                request.tolerance,
            )

        # Compute favorites
        result = compute_favorites_for_groups(
            duckdb_layer=duckdb_layer,
            election_group_store=election_group_store,
            target_days_to_exp=request.target_days_to_exp,
            tolerance=request.tolerance,
            group_ids=request.group_ids,
        )

        # Save signals
        signals_saved = favorite_store.save_signals(result.signals)

        duckdb_layer.close()

        return FavoriteComputeResultSchema(
            signals_computed=len(result.signals),
            signals_saved=signals_saved,
            groups_processed=result.groups_processed,
            groups_with_data=result.groups_with_data,
            groups_without_data=result.groups_without_data,
            total_markets=result.total_markets,
            markets_with_trades=result.markets_with_trades,
            markets_without_trades=result.markets_without_trades,
            snapshot_days_to_exp=result.snapshot_days_to_exp,
            tolerance=result.tolerance,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute favorites: {e}")


@app.get("/api/favorite-signals", response_model=FavoriteSignalListResponse)
def get_favorite_signals(
    snapshot_days_to_exp: float = 90.0,
    tolerance: float = 0.5,
) -> FavoriteSignalListResponse:
    """
    Get computed favorite signals for a specific snapshot.

    Args:
        snapshot_days_to_exp: Target days to expiry for the snapshot
        tolerance: +/- tolerance for matching snapshot

    Returns:
        List of favorite signals for the snapshot
    """
    try:
        favorite_store = FavoriteSignalStore(settings.metadata_db_path)
        signals = favorite_store.get_signals_for_snapshot(snapshot_days_to_exp, tolerance)

        return FavoriteSignalListResponse(
            signals=[
                FavoriteSignalSchema(
                    id=s["id"],
                    election_group_id=s["election_group_id"],
                    election_group_name=s["election_group_name"],
                    favorite_market_id=s["favorite_market_id"],
                    favorite_price=s["favorite_price"],
                    favorite_question=s["favorite_question"],
                    snapshot_days_to_exp=s["snapshot_days_to_exp"],
                    all_market_prices=s["all_market_prices"],
                    computed_at=s["computed_at"],
                    created_at=s["created_at"],
                )
                for s in signals
            ],
            count=len(signals),
            snapshot_days_to_exp=snapshot_days_to_exp,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get favorite signals: {e}")


@app.get(
    "/api/favorite-signals/group/{election_group_id}",
    response_model=FavoriteSignalSchema | dict[str, str],
)
def get_favorite_signal_for_group(
    election_group_id: str,
    snapshot_days_to_exp: float = 90.0,
    tolerance: float = 0.5,
) -> FavoriteSignalSchema | dict[str, str]:
    """
    Get the favorite signal for a specific election group at a snapshot.

    Args:
        election_group_id: The election group ID
        snapshot_days_to_exp: Target days to expiry for the snapshot
        tolerance: +/- tolerance for matching snapshot

    Returns:
        The favorite signal or a message if not found
    """
    try:
        favorite_store = FavoriteSignalStore(settings.metadata_db_path)
        signal = favorite_store.get_signal_for_group(
            election_group_id, snapshot_days_to_exp, tolerance
        )

        if signal is None:
            return {
                "status": "not_found",
                "message": f"No favorite signal found for group {election_group_id} at snapshot {snapshot_days_to_exp}",
            }

        return FavoriteSignalSchema(
            id=signal["id"],
            election_group_id=signal["election_group_id"],
            election_group_name=signal["election_group_name"],
            favorite_market_id=signal["favorite_market_id"],
            favorite_price=signal["favorite_price"],
            favorite_question=signal["favorite_question"],
            snapshot_days_to_exp=signal["snapshot_days_to_exp"],
            all_market_prices=signal["all_market_prices"],
            computed_at=signal["computed_at"],
            created_at=signal["created_at"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get favorite signal: {e}")


@app.get("/api/favorite-signals/snapshots", response_model=list[FavoriteSnapshotSummary])
def list_favorite_snapshots() -> list[FavoriteSnapshotSummary]:
    """
    List all unique snapshots with computed favorite signals.

    Returns:
        List of snapshots with signal counts and last computed times
    """
    try:
        favorite_store = FavoriteSignalStore(settings.metadata_db_path)
        snapshots = favorite_store.list_snapshots()

        return [
            FavoriteSnapshotSummary(
                snapshot_days_to_exp=s["snapshot_days_to_exp"],
                signal_count=s["signal_count"],
                last_computed=s["last_computed"],
            )
            for s in snapshots
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {e}")


@app.delete("/api/favorite-signals")
def clear_favorite_signals(
    snapshot_days_to_exp: float = 90.0,
    tolerance: float = 0.5,
) -> dict[str, Any]:
    """
    Clear favorite signals for a specific snapshot.

    This is useful before recomputing signals with new parameters.

    Args:
        snapshot_days_to_exp: Target days to expiry for the snapshot
        tolerance: +/- tolerance for matching snapshot

    Returns:
        Number of signals deleted
    """
    try:
        favorite_store = FavoriteSignalStore(settings.metadata_db_path)
        deleted = favorite_store.clear_signals_for_snapshot(snapshot_days_to_exp, tolerance)

        return {
            "status": "success",
            "deleted_count": deleted,
            "snapshot_days_to_exp": snapshot_days_to_exp,
            "tolerance": tolerance,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear signals: {e}")


# =============================================================================
# Data Quality Check endpoints for validation and reporting
# =============================================================================


@app.post("/api/data-quality/check", response_model=DataQualityReportSchema)
def run_data_quality_check(
    request: DataQualityCheckRequest | None = None,
) -> DataQualityReportSchema:
    """
    Run a comprehensive data quality check on the data layer.

    This validates:
    1. Uniqueness - transaction_hash uniqueness for trades
    2. Range checks - price within 0-1, usd_amount non-negative
    3. Referential integrity - trades.market_id exists in markets
    4. Warnings - markets missing closedTime (needed for backtests)

    A report is generated and persisted for later retrieval.

    Args:
        request: Optional request with run_type (default: "bootstrap")

    Returns:
        Complete data quality report with all findings
    """
    run_type = request.run_type if request else "bootstrap"

    try:
        checker = DataQualityChecker(
            parquet_dir=settings.parquet_dir,
            db_path=settings.metadata_db_path,
            partitioned=settings.parquet_partitioning_enabled,
        )
        report = checker.run_full_check(run_type=run_type)

        return DataQualityReportSchema(
            report_id=report.report_id,
            entity=report.entity,
            run_type=report.run_type,
            started_at=report.started_at,
            completed_at=report.completed_at,
            uniqueness_valid=report.uniqueness_valid,
            uniqueness_issues=[
                UniquenessIssueSchema(
                    column=i.column,
                    duplicate_value=i.duplicate_value,
                    occurrence_count=i.occurrence_count,
                )
                for i in report.uniqueness_issues
            ],
            duplicate_count=report.duplicate_count,
            range_valid=report.range_valid,
            range_issues=[
                RangeIssueSchema(
                    column=i.column,
                    value=i.value,
                    expected_min=i.expected_min,
                    expected_max=i.expected_max,
                    row_identifier=i.row_identifier,
                )
                for i in report.range_issues
            ],
            out_of_range_count=report.out_of_range_count,
            referential_integrity_valid=report.referential_integrity_valid,
            referential_integrity_issues=[
                ReferentialIntegrityIssueSchema(
                    source_table=i.source_table,
                    source_column=i.source_column,
                    source_value=i.source_value,
                    target_table=i.target_table,
                    target_column=i.target_column,
                )
                for i in report.referential_integrity_issues
            ],
            orphaned_count=report.orphaned_count,
            markets_without_closed_time=report.markets_without_closed_time,
            total_issues=report.total_issues,
            is_valid=report.is_valid,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run data quality check: {e}")


@app.get("/api/data-quality/reports", response_model=DataQualityReportListResponse)
def list_data_quality_reports(
    limit: int = 50,
) -> DataQualityReportListResponse:
    """
    List recent data quality reports.

    Args:
        limit: Maximum number of reports to return (default 50)

    Returns:
        List of data quality reports ordered by completion time (newest first)
    """
    try:
        checker = DataQualityChecker(
            parquet_dir=settings.parquet_dir,
            db_path=settings.metadata_db_path,
            partitioned=settings.parquet_partitioning_enabled,
        )
        reports = checker.list_reports(limit=limit)

        return DataQualityReportListResponse(
            reports=[
                DataQualityReportSchema(
                    report_id=r.report_id,
                    entity=r.entity,
                    run_type=r.run_type,
                    started_at=r.started_at,
                    completed_at=r.completed_at,
                    uniqueness_valid=r.uniqueness_valid,
                    uniqueness_issues=[
                        UniquenessIssueSchema(
                            column=i.column,
                            duplicate_value=i.duplicate_value,
                            occurrence_count=i.occurrence_count,
                        )
                        for i in r.uniqueness_issues
                    ],
                    duplicate_count=r.duplicate_count,
                    range_valid=r.range_valid,
                    range_issues=[
                        RangeIssueSchema(
                            column=i.column,
                            value=i.value,
                            expected_min=i.expected_min,
                            expected_max=i.expected_max,
                            row_identifier=i.row_identifier,
                        )
                        for i in r.range_issues
                    ],
                    out_of_range_count=r.out_of_range_count,
                    referential_integrity_valid=r.referential_integrity_valid,
                    referential_integrity_issues=[
                        ReferentialIntegrityIssueSchema(
                            source_table=i.source_table,
                            source_column=i.source_column,
                            source_value=i.source_value,
                            target_table=i.target_table,
                            target_column=i.target_column,
                        )
                        for i in r.referential_integrity_issues
                    ],
                    orphaned_count=r.orphaned_count,
                    markets_without_closed_time=r.markets_without_closed_time,
                    total_issues=r.total_issues,
                    is_valid=r.is_valid,
                )
                for r in reports
            ],
            count=len(reports),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {e}")


@app.get("/api/data-quality/reports/{report_id}", response_model=DataQualityReportSchema)
def get_data_quality_report(report_id: str) -> DataQualityReportSchema:
    """
    Get a specific data quality report by ID.

    Args:
        report_id: The report UUID

    Returns:
        The complete data quality report
    """
    try:
        checker = DataQualityChecker(
            parquet_dir=settings.parquet_dir,
            db_path=settings.metadata_db_path,
            partitioned=settings.parquet_partitioning_enabled,
        )
        report = checker.get_report(report_id)

        if report is None:
            raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")

        return DataQualityReportSchema(
            report_id=report.report_id,
            entity=report.entity,
            run_type=report.run_type,
            started_at=report.started_at,
            completed_at=report.completed_at,
            uniqueness_valid=report.uniqueness_valid,
            uniqueness_issues=[
                UniquenessIssueSchema(
                    column=i.column,
                    duplicate_value=i.duplicate_value,
                    occurrence_count=i.occurrence_count,
                )
                for i in report.uniqueness_issues
            ],
            duplicate_count=report.duplicate_count,
            range_valid=report.range_valid,
            range_issues=[
                RangeIssueSchema(
                    column=i.column,
                    value=i.value,
                    expected_min=i.expected_min,
                    expected_max=i.expected_max,
                    row_identifier=i.row_identifier,
                )
                for i in report.range_issues
            ],
            out_of_range_count=report.out_of_range_count,
            referential_integrity_valid=report.referential_integrity_valid,
            referential_integrity_issues=[
                ReferentialIntegrityIssueSchema(
                    source_table=i.source_table,
                    source_column=i.source_column,
                    source_value=i.source_value,
                    target_table=i.target_table,
                    target_column=i.target_column,
                )
                for i in report.referential_integrity_issues
            ],
            orphaned_count=report.orphaned_count,
            markets_without_closed_time=report.markets_without_closed_time,
            total_issues=report.total_issues,
            is_valid=report.is_valid,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report: {e}")


# =============================================================================
# Performance benchmark endpoints
# =============================================================================


@app.post("/api/performance/benchmark", response_model=PerformanceBenchmarkReportSchema)
def run_performance_benchmark(
    request: PerformanceBenchmarkRequest | None = None,
) -> PerformanceBenchmarkReportSchema:
    """
    Run a complete performance benchmark suite.

    Benchmarks DuckDB-over-Parquet query performance for:
    - Single market_id query over 30-day window
    - 100+ market_ids query over 30-day window
    - Predicate pushdown/partition pruning verification
    - Memory-bounded queries with limits
    - days_to_exp filtering for backtests

    Records baseline numbers for regression tracking.
    """
    try:
        # Check if parquet files exist
        if not settings.parquet_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="No Parquet data found. Run bootstrap first.",
            )

        # Check if partitioned
        trades_path = settings.parquet_dir / "trades.parquet"
        trades_dir = settings.parquet_dir / "trades"
        partitioned = trades_dir.exists() and trades_dir.is_dir()

        duckdb_layer = DuckDBLayer(
            settings.duckdb_path,
            settings.parquet_dir,
            partitioned=partitioned,
        )

        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            include_plans = request.include_plans if request else True
            report = benchmarker.run_full_benchmark(include_plans=include_plans)

            return PerformanceBenchmarkReportSchema(
                report_id=report.report_id,
                created_at=report.created_at,
                total_trades=report.total_trades,
                total_markets=report.total_markets,
                partitioned=report.partitioned,
                benchmarks=[
                    QueryBenchmarkResultSchema(
                        query_name=b.query_name,
                        query_description=b.query_description,
                        execution_time_ms=b.execution_time_ms,
                        rows_returned=b.rows_returned,
                        parameters=b.parameters,
                        query_plan=b.query_plan,
                        memory_usage_bytes=b.memory_usage_bytes,
                    )
                    for b in report.benchmarks
                ],
                summary=report.summary,
            )
        finally:
            duckdb_layer.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run benchmark: {e}")


# =============================================================================
# Cost efficiency endpoints
# =============================================================================


@app.post("/api/cost-efficiency/analyze", response_model=CostEfficiencyReportSchema)
def analyze_cost_efficiency() -> CostEfficiencyReportSchema:
    """
    Analyze cost efficiency of the storage system.

    Verifies that:
    - Local dev is fully functional using filesystem Parquet + DuckDB + SQLite
    - No managed database server is required for core workflows
    - Storage footprint is reduced vs CSV due to Parquet compression
    - Estimates monthly costs if data were stored on S3

    Returns a detailed report with storage footprints, compression ratios,
    infrastructure requirements, and cost estimates.
    """
    try:
        analyzer = CostEfficiencyAnalyzer(
            data_dir=settings.data_dir,
            parquet_dir=settings.parquet_dir,
            duckdb_path=settings.duckdb_path,
            metadata_db_path=settings.metadata_db_path,
            csv_dir=settings.data_dir,
        )

        report = analyzer.run_analysis()

        return CostEfficiencyReportSchema(
            report_id=report.report_id,
            created_at=report.created_at,
            storage_footprints=[
                StorageFootprintSchema(
                    path=sf.path,
                    size_bytes=sf.size_bytes,
                    size_mb=round(sf.size_mb, 2),
                    file_count=sf.file_count,
                    format=sf.format,
                )
                for sf in report.storage_footprints
            ],
            compression_ratios=[
                CompressionRatioSchema(
                    original_format=cr.original_format,
                    compressed_format=cr.compressed_format,
                    original_size_bytes=cr.original_size_bytes,
                    compressed_size_bytes=cr.compressed_size_bytes,
                    ratio=round(cr.ratio, 2),
                    savings_percent=round(cr.savings_percent, 1),
                )
                for cr in report.compression_ratios
            ],
            infrastructure=InfrastructureRequirementsSchema(
                requires_managed_db=report.infrastructure.requires_managed_db,
                requires_external_services=report.infrastructure.requires_external_services,
                local_only_services=report.infrastructure.local_only_services,
                storage_backends=report.infrastructure.storage_backends,
                notes=report.infrastructure.notes,
            ),
            estimated_monthly_cost_usd=report.estimated_monthly_cost_usd,
            notes=report.notes,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze cost efficiency: {e}")
