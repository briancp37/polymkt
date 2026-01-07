"""FastAPI application for the Polymkt analytics platform."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from polymkt.config import settings
from polymkt.models.schemas import (
    BootstrapSummary,
    CurateSummary,
    MarketSearchResult,
    RunRecord,
    UpdateSummary,
)
from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.pipeline.curate import run_curate
from polymkt.pipeline.update import run_update
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore
from polymkt.storage.search import MarketSearchIndex

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


@app.get("/api/markets/search", response_model=MarketSearchResponse)
def search_markets(
    q: str,
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    closed_time_min: str | None = None,
    closed_time_max: str | None = None,
) -> MarketSearchResponse:
    """
    Search markets using BM25 full-text search.

    This endpoint searches over market.question, market.tags (derived from events),
    and market.description using DuckDB's FTS extension.

    Results are sorted by relevance score (BM25) by default.

    Args:
        q: Search query string (required)
        limit: Maximum results to return (default 50)
        offset: Number of results to skip for pagination (default 0)
        category: Filter by market category (optional)
        closed_time_min: Filter by minimum closed_time (optional)
        closed_time_max: Filter by maximum closed_time (optional)

    Returns:
        Paginated search results with relevance scores
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

        # Create search index and build it
        search_index = MarketSearchIndex(duckdb_layer.conn)
        search_index.build_index()

        # Perform the search
        results_raw, total_count = search_index.search(
            query=q,
            limit=limit,
            offset=offset,
            category=category,
            closed_time_min=closed_time_min,
            closed_time_max=closed_time_max,
        )

        # Convert to response model
        results = [
            MarketSearchResult(
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
        return MarketSearchResponse(
            results=results,
            count=len(results),
            total_count=total_count,
            has_more=has_more,
        )
    except RuntimeError as e:
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
