"""FastAPI application for the Polymkt analytics platform."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from polymkt.config import settings
from polymkt.models.schemas import BootstrapSummary, RunRecord
from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.metadata import MetadataStore

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
