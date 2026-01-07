"""Data schemas for markets, trades, and orders."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventSchema(BaseModel):
    """Schema for event data (parent entity for related markets)."""

    model_config = ConfigDict(populate_by_name=True)

    event_id: str = Field(..., alias="eventId", description="Event ID")
    tags: list[str] = Field(default_factory=list, description="Event tags/categories")
    title: str | None = Field(None, description="Event title")
    description: str | None = Field(None, description="Event description")
    created_at: datetime | None = Field(None, alias="createdAt", description="Event creation time")


class MarketSchema(BaseModel):
    """Schema for market data."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Market ID")
    question: str = Field(..., description="Market question")
    created_at: datetime = Field(..., alias="createdAt", description="Creation timestamp")
    answer1: str | None = Field(None, description="First answer option")
    answer2: str | None = Field(None, description="Second answer option")
    neg_risk: bool | None = Field(None, description="Negative risk flag")
    market_slug: str | None = Field(None, description="Market slug for URL")
    token1: str | None = Field(None, description="First token ID")
    token2: str | None = Field(None, description="Second token ID")
    condition_id: str | None = Field(None, description="Condition ID")
    volume: float | None = Field(None, description="Trading volume")
    ticker: str | None = Field(None, description="Market ticker")
    closed_time: datetime | None = Field(None, alias="closedTime", description="Market closure time")
    description: str | None = Field(None, description="Market description")
    category: str | None = Field(None, description="Market category")
    event_id: str | None = Field(None, alias="eventId", description="Parent event ID for grouping")
    tags: list[str] | None = Field(None, description="Tags derived from event")


class TradeSchema(BaseModel):
    """Schema for trade data."""

    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime = Field(..., description="Trade timestamp in UTC")
    market_id: str = Field(..., description="Market ID")
    maker: str = Field(..., description="Maker address")
    taker: str = Field(..., description="Taker address")
    nonusdc_side: str | None = Field(None, description="Non-USDC side of trade")
    maker_direction: str | None = Field(None, description="Maker direction (buy/sell)")
    taker_direction: str | None = Field(None, description="Taker direction (buy/sell)")
    price: float = Field(..., description="Trade price")
    usd_amount: float = Field(..., description="USD amount")
    token_amount: float = Field(..., description="Token amount")
    transaction_hash: str = Field(..., alias="transactionHash", description="Transaction hash")


class OrderFilledSchema(BaseModel):
    """Schema for order filled events."""

    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime = Field(..., description="Event timestamp in UTC")
    maker: str = Field(..., description="Maker address")
    maker_asset_id: str = Field(..., alias="makerAssetId", description="Maker asset ID")
    maker_amount_filled: float = Field(..., alias="makerAmountFilled", description="Maker amount filled")
    taker: str = Field(..., description="Taker address")
    taker_asset_id: str = Field(..., alias="takerAssetId", description="Taker asset ID")
    taker_amount_filled: float = Field(..., alias="takerAmountFilled", description="Taker amount filled")
    transaction_hash: str = Field(..., alias="transactionHash", description="Transaction hash")


class RunRecord(BaseModel):
    """Metadata record for a pipeline run."""

    run_id: str = Field(..., description="Unique run identifier")
    run_type: str = Field(..., description="Type of run (bootstrap, update)")
    start_time: datetime = Field(..., description="Run start time")
    end_time: datetime | None = Field(None, description="Run end time")
    status: str = Field("running", description="Run status")
    rows_read: dict[str, int] = Field(default_factory=dict, description="Rows read per entity")
    rows_written: dict[str, int] = Field(default_factory=dict, description="Rows written per entity")
    watermark_before: dict[str, Any] | None = Field(None, description="Watermarks before run")
    watermark_after: dict[str, Any] | None = Field(None, description="Watermarks after run")
    schema_version: str = Field("1.0.0", description="Schema version used")
    error_message: str | None = Field(None, description="Error message if failed")
    duration_seconds: float | None = Field(None, description="Run duration in seconds")


class BootstrapSummary(BaseModel):
    """Summary of a bootstrap import operation."""

    run_id: str
    status: str
    duration_seconds: float
    markets_rows: int
    trades_rows: int
    order_filled_rows: int
    events_rows: int = Field(default=0, description="Number of event rows processed")
    schema_version: str
    parquet_files: list[str]
    rows_quarantined: dict[str, int] = Field(
        default_factory=dict, description="Rows quarantined per entity due to validation errors"
    )


class CurateSummary(BaseModel):
    """Summary of a curate operation."""

    run_id: str
    status: str
    duration_seconds: float
    trades_rows_read: int
    trades_rows_written: int
    analytics_files: list[str]


class UpdateSummary(BaseModel):
    """Summary of an incremental update operation."""

    run_id: str
    status: str
    duration_seconds: float
    rows_read: dict[str, int] = Field(
        default_factory=dict, description="Rows read per entity from source"
    )
    rows_written: dict[str, int] = Field(
        default_factory=dict, description="New rows written per entity"
    )
    rows_skipped: dict[str, int] = Field(
        default_factory=dict, description="Rows skipped due to deduplication per entity"
    )
    rows_updated: dict[str, int] = Field(
        default_factory=dict, description="Rows updated (for markets upsert)"
    )
    rows_quarantined: dict[str, int] = Field(
        default_factory=dict, description="Rows quarantined due to validation errors"
    )
    watermark_before: dict[str, Any] = Field(
        default_factory=dict, description="Watermarks before the update"
    )
    watermark_after: dict[str, Any] = Field(
        default_factory=dict, description="Watermarks after the update"
    )


class MarketSearchResult(BaseModel):
    """A single market search result."""

    id: str = Field(..., description="Market ID")
    question: str = Field(..., description="Market question")
    tags: list[str] | None = Field(None, description="Market tags derived from event")
    category: str | None = Field(None, description="Market category")
    closed_time: datetime | None = Field(None, description="Market close time")
    event_id: str | None = Field(None, description="Parent event ID")
    score: float = Field(..., description="BM25 relevance score")


class UnifiedMarketSearchResult(BaseModel):
    """A unified market search result that works for all search modes."""

    id: str = Field(..., description="Market ID")
    question: str = Field(..., description="Market question")
    tags: list[str] | None = Field(None, description="Market tags derived from event")
    category: str | None = Field(None, description="Market category")
    closed_time: datetime | None = Field(None, description="Market close time")
    event_id: str | None = Field(None, description="Parent event ID")
    relevance_score: float = Field(..., description="Relevance score (interpretation depends on mode)")
    snippet: str | None = Field(None, description="Text snippet with query terms highlighted")
    # Mode-specific scores (optional, only populated for hybrid mode)
    bm25_score: float | None = Field(None, description="BM25 score (hybrid mode only)")
    semantic_score: float | None = Field(None, description="Semantic similarity score (hybrid mode only)")


class SemanticSearchResult(BaseModel):
    """A single semantic search result with cosine similarity score."""

    id: str = Field(..., description="Market ID")
    question: str = Field(..., description="Market question")
    tags: list[str] | None = Field(None, description="Market tags derived from event")
    category: str | None = Field(None, description="Market category")
    closed_time: datetime | None = Field(None, description="Market close time")
    event_id: str | None = Field(None, description="Parent event ID")
    score: float = Field(..., description="Cosine similarity score (0-1, higher is better)")


class EmbeddingStats(BaseModel):
    """Statistics about the embeddings index."""

    total_embeddings: int = Field(..., description="Total number of market embeddings")
    embedding_model: str | None = Field(None, description="OpenAI model used for embeddings")
    embedding_dim: int | None = Field(None, description="Dimension of embedding vectors")
    first_created: datetime | None = Field(None, description="First embedding creation time")
    last_created: datetime | None = Field(None, description="Last embedding creation time")


class HybridSearchResult(BaseModel):
    """A single hybrid search result combining BM25 and semantic scores."""

    id: str = Field(..., description="Market ID")
    question: str = Field(..., description="Market question")
    tags: list[str] | None = Field(None, description="Market tags derived from event")
    category: str | None = Field(None, description="Market category")
    closed_time: datetime | None = Field(None, description="Market close time")
    event_id: str | None = Field(None, description="Parent event ID")
    score: float = Field(..., description="Combined RRF score (higher is better)")
    bm25_score: float | None = Field(None, description="BM25 relevance score (if in BM25 results)")
    semantic_score: float | None = Field(
        None, description="Cosine similarity score (if in semantic results)"
    )
    bm25_rank: int | None = Field(None, description="Rank in BM25 results")
    semantic_rank: int | None = Field(None, description="Rank in semantic results")


class HybridIndexStats(BaseModel):
    """Statistics about the hybrid search indices."""

    bm25_available: bool = Field(..., description="Whether BM25 index is available")
    semantic_available: bool = Field(..., description="Whether semantic index is available")
    bm25_markets_indexed: int | None = Field(None, description="Markets in BM25 index")
    semantic_markets_indexed: int | None = Field(None, description="Markets in semantic index")
    semantic_embedding_model: str | None = Field(None, description="Embedding model used")
    semantic_embedding_dim: int | None = Field(None, description="Embedding dimensions")


class SearchIndexUpdateResult(BaseModel):
    """Result of an incremental search index update."""

    status: str = Field(..., description="Status of the update (success, no_changes)")
    mode: str = Field(..., description="Update mode (incremental, full_rebuild, specific)")
    bm25_updated: int = Field(0, description="Number of markets updated in BM25 index")
    semantic_updated: int = Field(0, description="Number of markets updated in semantic index")
    new_markets: int = Field(0, description="Number of new markets detected")
    changed_markets: int = Field(0, description="Number of changed markets detected")
    deleted_markets: int = Field(0, description="Number of deleted markets detected")
    new_events: int = Field(0, description="Number of new events detected")
    changed_events: int = Field(0, description="Number of changed events detected")
    deleted_events: int = Field(0, description="Number of deleted events detected")
    event_affected_markets: int = Field(
        0, description="Number of markets updated due to event tag changes"
    )


class SearchIndexUpdaterStats(BaseModel):
    """Statistics about the search index updater."""

    total_hashes: int = Field(0, description="Total market content hashes tracked")
    first_updated: datetime | None = Field(None, description="First market hash update time")
    last_updated: datetime | None = Field(None, description="Last market hash update time")
    total_event_hashes: int = Field(0, description="Total event content hashes tracked")
    event_first_updated: datetime | None = Field(None, description="First event hash update time")
    event_last_updated: datetime | None = Field(None, description="Last event hash update time")
    bm25_available: bool = Field(..., description="Whether BM25 index is available")
    semantic_available: bool = Field(..., description="Whether semantic index is available")
    bm25_markets_indexed: int | None = Field(None, description="Markets in BM25 index")
    semantic_markets_indexed: int | None = Field(None, description="Markets in semantic index")
    semantic_embedding_model: str | None = Field(None, description="Embedding model used")
    semantic_embedding_dim: int | None = Field(None, description="Embedding dimensions")


# =============================================================================
# Dataset schemas for persisting market sets with filters and market lists
# =============================================================================


class DatasetFilters(BaseModel):
    """Filters used to create a dataset."""

    query: str | None = Field(None, description="Search query used to find markets")
    category: str | None = Field(None, description="Category filter")
    tags: list[str] | None = Field(None, description="Tag filters")
    closed_time_min: datetime | None = Field(None, description="Minimum closed time")
    closed_time_max: datetime | None = Field(None, description="Maximum closed time")
    min_volume: float | None = Field(None, description="Minimum trading volume")


class DatasetSchema(BaseModel):
    """Schema for a saved dataset (market set with filters and market list)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Dataset ID (UUID)")
    name: str = Field(..., description="User-friendly dataset name")
    description: str | None = Field(None, description="Dataset description")
    filters: DatasetFilters | None = Field(None, description="Filters used to create dataset")
    market_ids: list[str] = Field(default_factory=list, description="Included market IDs")
    excluded_market_ids: list[str] = Field(
        default_factory=list, description="Explicitly excluded market IDs"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class DatasetCreateRequest(BaseModel):
    """Request to create a new dataset."""

    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    description: str | None = Field(None, max_length=2000, description="Dataset description")
    filters: DatasetFilters | None = Field(None, description="Filters used to create dataset")
    market_ids: list[str] = Field(..., min_length=1, description="Market IDs to include")
    excluded_market_ids: list[str] = Field(
        default_factory=list, description="Market IDs to explicitly exclude"
    )


class DatasetUpdateRequest(BaseModel):
    """Request to update an existing dataset."""

    name: str | None = Field(None, min_length=1, max_length=255, description="Dataset name")
    description: str | None = Field(None, max_length=2000, description="Dataset description")
    filters: DatasetFilters | None = Field(None, description="Filters used to create dataset")
    market_ids: list[str] | None = Field(None, description="Market IDs to include")
    excluded_market_ids: list[str] | None = Field(
        None, description="Market IDs to explicitly exclude"
    )


class DatasetSummary(BaseModel):
    """Summary of a dataset for list views."""

    id: str = Field(..., description="Dataset ID")
    name: str = Field(..., description="Dataset name")
    description: str | None = Field(None, description="Dataset description")
    market_count: int = Field(..., description="Number of included markets")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class DatasetListResponse(BaseModel):
    """Response for listing datasets."""

    datasets: list[DatasetSummary] = Field(..., description="List of dataset summaries")
    count: int = Field(..., description="Number of datasets in this response")
    total_count: int = Field(..., description="Total number of datasets")
    has_more: bool = Field(..., description="Whether more datasets exist")


# =============================================================================
# Backtest schemas for persisting backtest runs and results
# =============================================================================


class StrategyConfig(BaseModel):
    """Configuration for a backtest strategy."""

    name: str = Field(..., description="Strategy name (e.g., 'buy_favorite')")
    entry_days_to_exp: float | None = Field(
        None, description="Days to expiry for entry signal (e.g., 90)"
    )
    exit_rule: str | None = Field(
        None, description="Exit rule (e.g., 'expiry', 'stop_loss', 'take_profit')"
    )
    favorite_rule: str | None = Field(
        None, description="Rule for determining favorite (e.g., 'max_yes_price')"
    )
    fee_rate: float = Field(0.0, description="Fee rate (0.0 to 1.0)")
    slippage_rate: float = Field(0.0, description="Slippage rate (0.0 to 1.0)")
    position_size: float = Field(1.0, description="Position size as fraction of capital (0.0 to 1.0)")
    extra_params: dict[str, Any] | None = Field(None, description="Additional strategy parameters")


class BacktestMetrics(BaseModel):
    """Summary metrics for a backtest run."""

    total_return: float | None = Field(None, description="Total return as percentage")
    total_pnl: float | None = Field(None, description="Total profit/loss in USD")
    win_rate: float | None = Field(None, description="Win rate (0.0 to 1.0)")
    trade_count: int | None = Field(None, description="Total number of trades")
    winning_trades: int | None = Field(None, description="Number of winning trades")
    losing_trades: int | None = Field(None, description="Number of losing trades")
    max_drawdown: float | None = Field(None, description="Maximum drawdown as percentage")
    sharpe_ratio: float | None = Field(None, description="Sharpe ratio")
    avg_trade_pnl: float | None = Field(None, description="Average PnL per trade")
    avg_holding_period_days: float | None = Field(None, description="Average holding period in days")


class BacktestTradeRecord(BaseModel):
    """Record of a single trade in a backtest."""

    market_id: str = Field(..., description="Market ID")
    election_group_id: str | None = Field(None, description="Election group ID if applicable")
    entry_time: datetime = Field(..., description="Entry timestamp")
    entry_price: float = Field(..., description="Entry price")
    exit_time: datetime | None = Field(None, description="Exit timestamp")
    exit_price: float | None = Field(None, description="Exit price")
    position_size: float = Field(..., description="Position size")
    pnl: float | None = Field(None, description="Profit/loss for this trade")
    fees_paid: float | None = Field(None, description="Fees paid for this trade")
    slippage_cost: float | None = Field(None, description="Slippage cost for this trade")


class BacktestSchema(BaseModel):
    """Schema for a saved backtest run."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Backtest ID (UUID)")
    dataset_id: str = Field(..., description="Dataset ID this backtest ran on")
    strategy_config: StrategyConfig = Field(..., description="Strategy configuration used")
    status: str = Field(
        "pending", description="Backtest status (pending, running, completed, failed)"
    )
    metrics: BacktestMetrics | None = Field(None, description="Summary metrics (null if not completed)")
    trades: list[BacktestTradeRecord] | None = Field(
        None, description="Trade records (null if not completed)"
    )
    equity_curve: list[dict[str, Any]] | None = Field(
        None, description="Equity curve data points (null if not completed)"
    )
    error_message: str | None = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")


class BacktestCreateRequest(BaseModel):
    """Request to create a new backtest."""

    dataset_id: str = Field(..., description="Dataset ID to run backtest on")
    strategy_config: StrategyConfig = Field(..., description="Strategy configuration")


class BacktestUpdateRequest(BaseModel):
    """Request to update a backtest (e.g., mark as completed with results)."""

    status: str | None = Field(None, description="Backtest status")
    metrics: BacktestMetrics | None = Field(None, description="Summary metrics")
    trades: list[BacktestTradeRecord] | None = Field(None, description="Trade records")
    equity_curve: list[dict[str, Any]] | None = Field(None, description="Equity curve data points")
    error_message: str | None = Field(None, description="Error message if failed")


class BacktestSummary(BaseModel):
    """Summary of a backtest for list views."""

    id: str = Field(..., description="Backtest ID")
    dataset_id: str = Field(..., description="Dataset ID")
    strategy_name: str = Field(..., description="Strategy name from config")
    status: str = Field(..., description="Backtest status")
    total_return: float | None = Field(None, description="Total return percentage")
    trade_count: int | None = Field(None, description="Number of trades")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")


class BacktestListResponse(BaseModel):
    """Response for listing backtests."""

    backtests: list[BacktestSummary] = Field(..., description="List of backtest summaries")
    count: int = Field(..., description="Number of backtests in this response")
    total_count: int = Field(..., description="Total number of backtests")
    has_more: bool = Field(..., description="Whether more backtests exist")
