"""Data schemas for markets, trades, and orders."""

from datetime import datetime
from enum import Enum
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
    excluded_count: int = Field(..., description="Number of excluded markets")
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


# =============================================================================
# Election group schemas for grouping related markets
# =============================================================================


class ElectionGroupSchema(BaseModel):
    """Schema for an election group (collection of related markets)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Election group ID (UUID)")
    name: str = Field(..., description="Group name (e.g., '2024 Presidential Election')")
    description: str | None = Field(None, description="Group description")
    market_ids: list[str] = Field(default_factory=list, description="Market IDs in this group")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ElectionGroupCreateRequest(BaseModel):
    """Request to create a new election group."""

    name: str = Field(..., min_length=1, max_length=255, description="Group name")
    description: str | None = Field(None, max_length=2000, description="Group description")
    market_ids: list[str] = Field(
        default_factory=list, description="Initial market IDs to include"
    )


class ElectionGroupUpdateRequest(BaseModel):
    """Request to update an existing election group."""

    name: str | None = Field(None, min_length=1, max_length=255, description="Group name")
    description: str | None = Field(None, max_length=2000, description="Group description")
    market_ids: list[str] | None = Field(
        None, description="Market IDs (replaces existing if provided)"
    )


class ElectionGroupSummary(BaseModel):
    """Summary of an election group for list views."""

    id: str = Field(..., description="Election group ID")
    name: str = Field(..., description="Group name")
    description: str | None = Field(None, description="Group description")
    market_count: int = Field(..., description="Number of markets in the group")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ElectionGroupListResponse(BaseModel):
    """Response for listing election groups."""

    groups: list[ElectionGroupSummary] = Field(..., description="List of group summaries")
    count: int = Field(..., description="Number of groups in this response")
    total_count: int = Field(..., description="Total number of groups")
    has_more: bool = Field(..., description="Whether more groups exist")


class ElectionGroupImportResult(BaseModel):
    """Result of importing election group mappings."""

    groups_created: int = Field(..., description="Number of new groups created")
    markets_mapped: int = Field(..., description="Number of markets mapped to groups")
    errors: list[str] = Field(default_factory=list, description="Import errors")


class ElectionGroupValidationResult(BaseModel):
    """Result of validating election groups."""

    total_groups: int = Field(..., description="Total number of groups")
    valid_groups: int = Field(..., description="Groups meeting minimum requirements")
    invalid_groups: int = Field(..., description="Groups with issues")
    issues: list[dict[str, Any]] = Field(default_factory=list, description="Validation issues")
    min_markets_required: int = Field(..., description="Minimum markets required per group")


class UnmappedMarketsResult(BaseModel):
    """Result of finding unmapped markets."""

    unmapped_market_ids: list[str] = Field(..., description="Markets not in any group")
    total_checked: int = Field(..., description="Total markets checked")
    unmapped_count: int = Field(..., description="Count of unmapped markets")


# =============================================================================
# Favorite signal schemas for "buy the favorite" strategy
# =============================================================================


class FavoriteSignalSchema(BaseModel):
    """Schema for a favorite signal (highest YES price market at snapshot)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Signal ID (UUID)")
    election_group_id: str = Field(..., description="Election group ID")
    election_group_name: str = Field(..., description="Election group name")
    favorite_market_id: str = Field(..., description="Market ID of the favorite")
    favorite_price: float = Field(..., description="YES price of the favorite at snapshot")
    favorite_question: str | None = Field(None, description="Market question of the favorite")
    snapshot_days_to_exp: float = Field(
        ..., description="Days to expiry at which snapshot was taken"
    )
    all_market_prices: dict[str, float] = Field(
        ..., description="Prices of all markets in the group at snapshot"
    )
    computed_at: datetime = Field(..., description="When the signal was computed")
    created_at: datetime = Field(..., description="When the signal was persisted")


class FavoriteComputeRequest(BaseModel):
    """Request to compute favorite signals."""

    target_days_to_exp: float = Field(
        90.0, description="Target days to expiry for snapshot"
    )
    tolerance: float = Field(
        0.5, description="+/- tolerance around target days to expiry"
    )
    group_ids: list[str] | None = Field(
        None, description="Specific group IDs to process (None = all groups)"
    )
    clear_existing: bool = Field(
        True, description="Clear existing signals for this snapshot before computing"
    )


class FavoriteComputeResultSchema(BaseModel):
    """Result of computing favorite signals."""

    signals_computed: int = Field(..., description="Number of signals computed")
    signals_saved: int = Field(..., description="Number of signals saved to database")
    groups_processed: int = Field(..., description="Total groups processed")
    groups_with_data: int = Field(..., description="Groups with trade data at snapshot")
    groups_without_data: int = Field(..., description="Groups without trade data")
    total_markets: int = Field(..., description="Total markets across all groups")
    markets_with_trades: int = Field(..., description="Markets with trades at snapshot")
    markets_without_trades: int = Field(..., description="Markets without trades")
    snapshot_days_to_exp: float = Field(..., description="Snapshot days to expiry used")
    tolerance: float = Field(..., description="Tolerance used")


class FavoriteSignalListResponse(BaseModel):
    """Response for listing favorite signals."""

    signals: list[FavoriteSignalSchema] = Field(..., description="List of favorite signals")
    count: int = Field(..., description="Number of signals in this response")
    snapshot_days_to_exp: float = Field(..., description="Snapshot days to expiry queried")


class FavoriteSnapshotSummary(BaseModel):
    """Summary of signals for a specific snapshot."""

    snapshot_days_to_exp: float = Field(..., description="Days to expiry of the snapshot")
    signal_count: int = Field(..., description="Number of signals at this snapshot")
    last_computed: datetime = Field(..., description="When signals were last computed")


# =============================================================================
# Data quality schemas for validation reports
# =============================================================================


class UniquenessIssueSchema(BaseModel):
    """Details about a uniqueness violation."""

    column: str = Field(..., description="Column with duplicate values")
    duplicate_value: str = Field(..., description="Sample duplicate value")
    occurrence_count: int = Field(..., description="Number of occurrences")


class RangeIssueSchema(BaseModel):
    """Details about a value out of expected range."""

    column: str = Field(..., description="Column with out-of-range value")
    value: float = Field(..., description="The out-of-range value")
    expected_min: float | None = Field(None, description="Expected minimum")
    expected_max: float | None = Field(None, description="Expected maximum")
    row_identifier: str = Field(..., description="Identifier of the row (e.g., transaction_hash)")


class ReferentialIntegrityIssueSchema(BaseModel):
    """Details about a referential integrity violation."""

    source_table: str = Field(..., description="Table containing the orphan reference")
    source_column: str = Field(..., description="Column containing the orphan reference")
    source_value: str = Field(..., description="The orphaned value")
    target_table: str = Field(..., description="Table that should contain the referenced value")
    target_column: str = Field(..., description="Column that should contain the referenced value")


class DataQualityReportSchema(BaseModel):
    """Complete data quality report."""

    model_config = ConfigDict(populate_by_name=True)

    report_id: str = Field(..., description="Unique report identifier")
    entity: str = Field(..., description="Entity checked (trades, markets, all)")
    run_type: str = Field(..., description="Type of run (bootstrap, update)")
    started_at: datetime = Field(..., description="When the check started")
    completed_at: datetime = Field(..., description="When the check completed")

    # Uniqueness
    uniqueness_valid: bool = Field(..., description="Whether uniqueness checks passed")
    uniqueness_issues: list[UniquenessIssueSchema] = Field(
        default_factory=list, description="Sample uniqueness issues"
    )
    duplicate_count: int = Field(0, description="Total count of duplicate values")

    # Range
    range_valid: bool = Field(..., description="Whether range checks passed")
    range_issues: list[RangeIssueSchema] = Field(
        default_factory=list, description="Sample range issues"
    )
    out_of_range_count: int = Field(0, description="Total count of out-of-range values")

    # Referential integrity
    referential_integrity_valid: bool = Field(
        ..., description="Whether referential integrity checks passed"
    )
    referential_integrity_issues: list[ReferentialIntegrityIssueSchema] = Field(
        default_factory=list, description="Sample referential integrity issues"
    )
    orphaned_count: int = Field(0, description="Total count of orphaned references")

    # Market-specific
    markets_without_closed_time: int = Field(
        0, description="Markets missing closedTime (warning for backtests)"
    )

    # Summary
    total_issues: int = Field(..., description="Total number of issues found")
    is_valid: bool = Field(..., description="Whether all checks passed")


class DataQualityReportListResponse(BaseModel):
    """Response for listing data quality reports."""

    reports: list[DataQualityReportSchema] = Field(..., description="List of reports")
    count: int = Field(..., description="Number of reports in this response")


class DataQualityCheckRequest(BaseModel):
    """Request to run a data quality check."""

    run_type: str = Field("bootstrap", description="Type of run being checked")


# =============================================================================
# Performance benchmark schemas for query optimization tracking
# =============================================================================


class QueryBenchmarkResultSchema(BaseModel):
    """Result of a single query benchmark."""

    query_name: str = Field(..., description="Short name for the query type")
    query_description: str = Field(..., description="Description of what the query does")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    rows_returned: int = Field(..., description="Number of rows returned")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Query parameters used")
    query_plan: str | None = Field(None, description="Query execution plan")
    memory_usage_bytes: int | None = Field(None, description="Memory usage in bytes")


class PerformanceBenchmarkReportSchema(BaseModel):
    """Complete performance benchmark report."""

    report_id: str = Field(..., description="Unique report identifier")
    created_at: datetime = Field(..., description="When the benchmark was run")
    total_trades: int = Field(..., description="Total number of trades in database")
    total_markets: int = Field(..., description="Total number of markets in database")
    partitioned: bool = Field(..., description="Whether trades data is partitioned")
    benchmarks: list[QueryBenchmarkResultSchema] = Field(
        default_factory=list, description="Individual benchmark results"
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


class PerformanceBenchmarkRequest(BaseModel):
    """Request to run performance benchmarks."""

    include_plans: bool = Field(True, description="Include query plans in the report")


class PerformanceBenchmarkListResponse(BaseModel):
    """Response for listing performance benchmark reports."""

    reports: list[PerformanceBenchmarkReportSchema] = Field(
        ..., description="List of benchmark reports"
    )
    count: int = Field(..., description="Number of reports in this response")


# =============================================================================
# Cost efficiency schemas for infrastructure and storage analysis
# =============================================================================


class StorageFootprintSchema(BaseModel):
    """Storage footprint for a file or directory."""

    path: str = Field(..., description="Path to the file or directory")
    size_bytes: int = Field(..., description="Size in bytes")
    size_mb: float = Field(..., description="Size in megabytes")
    file_count: int = Field(..., description="Number of files")
    format: str = Field(..., description="Storage format (csv, parquet, sqlite, duckdb)")


class CompressionRatioSchema(BaseModel):
    """Compression ratio between storage formats."""

    original_format: str = Field(..., description="Original format (e.g., csv)")
    compressed_format: str = Field(..., description="Compressed format (e.g., parquet)")
    original_size_bytes: int = Field(..., description="Original size in bytes")
    compressed_size_bytes: int = Field(..., description="Compressed size in bytes")
    ratio: float = Field(..., description="Compression ratio (original / compressed)")
    savings_percent: float = Field(..., description="Storage savings percentage")


class InfrastructureRequirementsSchema(BaseModel):
    """Infrastructure requirements for the current setup."""

    requires_managed_db: bool = Field(False, description="Whether a managed DB is required")
    requires_external_services: list[str] = Field(
        default_factory=list, description="External services required"
    )
    local_only_services: list[str] = Field(
        default_factory=list, description="Services that run locally"
    )
    storage_backends: list[str] = Field(
        default_factory=list, description="Storage backends used"
    )
    notes: list[str] = Field(default_factory=list, description="Additional notes")


class CostEfficiencyReportSchema(BaseModel):
    """Complete cost efficiency report."""

    report_id: str = Field(..., description="Unique report identifier")
    created_at: datetime = Field(..., description="When the analysis was run")
    storage_footprints: list[StorageFootprintSchema] = Field(
        default_factory=list, description="Storage footprints for all files"
    )
    compression_ratios: list[CompressionRatioSchema] = Field(
        default_factory=list, description="Compression ratios between formats"
    )
    infrastructure: InfrastructureRequirementsSchema = Field(
        ..., description="Infrastructure requirements"
    )
    estimated_monthly_cost_usd: float = Field(
        ..., description="Estimated monthly cost in USD if on S3"
    )
    notes: list[str] = Field(default_factory=list, description="Analysis notes")


# =============================================================================
# Dataset Agent schemas for natural language dataset creation
# =============================================================================


class DatasetAgentMarketItemSchema(BaseModel):
    """A market in the dataset agent's result with inclusion flag."""

    market_id: str = Field(..., description="Market ID")
    question: str = Field(..., description="Market question")
    category: str | None = Field(None, description="Market category")
    tags: list[str] | None = Field(None, description="Market tags")
    closed_time: datetime | None = Field(None, description="Market close time")
    relevance_score: float = Field(..., description="Relevance score from search")
    included: bool = Field(True, description="Whether market is included in dataset")


class DatasetAgentRequestSchema(BaseModel):
    """Request for the dataset agent to process."""

    natural_language_query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language query (e.g., 'find election markets about senate control')",
    )
    max_results: int = Field(100, ge=1, le=1000, description="Maximum markets to return")
    category_filter: str | None = Field(None, description="Override category filter")
    closed_time_min: str | None = Field(None, description="Override min closed time (ISO format)")
    closed_time_max: str | None = Field(None, description="Override max closed time (ISO format)")


class DatasetAgentResponseSchema(BaseModel):
    """Response from the dataset agent."""

    session_id: str = Field(..., description="Session ID for subsequent modifications")
    query: str = Field(..., description="Original natural language query")
    parsed_query: str = Field(..., description="Parsed search query")
    category_filter: str | None = Field(None, description="Detected or applied category filter")
    closed_time_filter: dict[str, str | None] | None = Field(
        None, description="Detected or applied time filter"
    )
    market_count: int = Field(..., description="Number of included markets")
    markets: list[DatasetAgentMarketItemSchema] = Field(
        ..., description="Markets with inclusion flags"
    )
    summary: str = Field(..., description="Human-readable summary of results")


class DatasetAgentModifyRequestSchema(BaseModel):
    """Request to modify market inclusion in a session."""

    session_id: str = Field(..., description="Session ID from process query")
    market_id: str | None = Field(None, description="Single market ID to modify")
    market_ids: list[str] | None = Field(None, description="Multiple market IDs to modify")
    included: bool = Field(..., description="Whether to include the market(s)")


class DatasetAgentSaveRequestSchema(BaseModel):
    """Request to save the agent response as a dataset."""

    session_id: str = Field(..., description="Session ID from process query")
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name for the dataset",
    )
    description: str | None = Field(None, max_length=2000, description="Dataset description")


class DatasetAgentSaveResultSchema(BaseModel):
    """Result of saving a dataset from agent response."""

    dataset_id: str = Field(..., description="Created dataset ID")
    dataset_name: str = Field(..., description="Dataset name")
    market_count: int = Field(..., description="Number of included markets")
    excluded_count: int = Field(..., description="Number of excluded markets")


# =============================================================================
# Backtesting Agent schemas
# =============================================================================


class ParsedStrategySchema(BaseModel):
    """Parsed strategy configuration from natural language."""

    name: str = Field(..., description="Generated strategy name")
    entry_days_to_exp: float = Field(..., description="Entry days to expiry")
    exit_rule: str = Field(..., description="Exit rule (e.g., 'expiry', 'take_profit')")
    favorite_rule: str = Field(
        ..., description="Favorite rule (e.g., 'max_yes_price')"
    )
    fee_rate: float = Field(..., description="Fee rate (0.0 to 1.0)")
    slippage_rate: float = Field(..., description="Slippage rate (0.0 to 1.0)")
    position_size: float = Field(..., description="Position size (0.0 to 1.0)")
    extra_params: dict[str, Any] | None = Field(
        None, description="Additional strategy parameters"
    )


class BacktestAgentRequestSchema(BaseModel):
    """Request for the backtesting agent."""

    dataset_id: str = Field(..., description="Dataset ID to run backtest on")
    natural_language_strategy: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language strategy description (e.g., 'buy favorite 90 days out, hold to expiry')",
    )
    fee_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override fee rate (0.0 to 1.0)",
    )
    slippage_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override slippage rate (0.0 to 1.0)",
    )
    position_size: float | None = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Override position size (0.0 to 1.0)",
    )


class StrategyConfirmationSchema(BaseModel):
    """Confirmation summary of parsed strategy before execution."""

    session_id: str = Field(..., description="Session ID for this preparation")
    dataset_id: str = Field(..., description="Dataset ID")
    dataset_name: str = Field(..., description="Dataset name")
    market_count: int = Field(..., description="Number of markets in dataset")
    parsed_strategy: ParsedStrategySchema = Field(..., description="Parsed strategy")
    summary: str = Field(..., description="Human-readable summary of the strategy")
    warnings: list[str] = Field(
        default_factory=list, description="Warnings about missing prerequisites"
    )


class BacktestAgentModifyRequestSchema(BaseModel):
    """Request to modify a prepared strategy."""

    session_id: str = Field(..., description="Session ID from prepare_backtest")
    entry_days_to_exp: float | None = Field(
        None,
        gt=0,
        description="New entry days to expiry",
    )
    exit_rule: str | None = Field(
        None,
        description="New exit rule (e.g., 'expiry', 'take_profit', 'stop_loss')",
    )
    favorite_rule: str | None = Field(
        None,
        description="New favorite rule (e.g., 'max_yes_price', 'min_yes_price')",
    )
    fee_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="New fee rate (0.0 to 1.0)",
    )
    slippage_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="New slippage rate (0.0 to 1.0)",
    )
    position_size: float | None = Field(
        None,
        gt=0.0,
        le=1.0,
        description="New position size (0.0 to 1.0)",
    )


class BacktestAgentResultSchema(BaseModel):
    """Result of executing a backtest via the agent."""

    backtest_id: str = Field(..., description="Created backtest ID")
    status: str = Field(..., description="Backtest status ('completed' or 'failed')")
    metrics: BacktestMetrics | None = Field(None, description="Backtest metrics")
    trades: list[BacktestTradeRecord] = Field(
        default_factory=list, description="List of trades"
    )
    equity_curve: list[dict[str, Any]] = Field(
        default_factory=list, description="Equity curve data points"
    )
    error_message: str | None = Field(None, description="Error message if failed")


# =============================================================================
# Ops Metadata schemas for watchlists, alerts, and analytics sessions
# =============================================================================


class WatchlistItemSchema(BaseModel):
    """A wallet address in a watchlist."""

    watchlist_id: str = Field(..., description="Parent watchlist ID")
    wallet_address: str = Field(..., description="Wallet address (lowercase)")
    added_at: datetime = Field(..., description="When the wallet was added")
    notes: str | None = Field(None, description="Optional notes about this wallet")


class WatchlistSchema(BaseModel):
    """A watchlist for tracking wallet addresses."""

    id: str = Field(..., description="Watchlist ID")
    name: str = Field(..., description="Watchlist name")
    description: str | None = Field(None, description="Watchlist description")
    wallet_addresses: list[str] = Field(
        default_factory=list, description="Wallet addresses in this watchlist"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class WatchlistCreateRequest(BaseModel):
    """Request to create a new watchlist."""

    name: str = Field(..., min_length=1, max_length=255, description="Watchlist name")
    description: str | None = Field(None, max_length=1000, description="Description")
    wallet_addresses: list[str] = Field(
        default_factory=list, description="Initial wallet addresses"
    )


class WatchlistAddWalletRequest(BaseModel):
    """Request to add a wallet to a watchlist."""

    wallet_address: str = Field(
        ..., pattern=r"^0x[a-fA-F0-9]{40}$", description="Ethereum wallet address"
    )
    notes: str | None = Field(None, max_length=500, description="Optional notes")


class AlertSubscriptionSchema(BaseModel):
    """An alert subscription tied to a watchlist."""

    id: str = Field(..., description="Subscription ID")
    watchlist_id: str = Field(..., description="Associated watchlist ID")
    rule_type: str = Field(
        ..., description="Alert rule type (e.g., 'trade', 'large_trade')"
    )
    rule_config: dict[str, Any] = Field(
        default_factory=dict, description="Rule configuration (thresholds, etc.)"
    )
    cooldown_seconds: int = Field(
        300, description="Minimum seconds between alerts for same wallet/market"
    )
    is_active: bool = Field(True, description="Whether the subscription is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class AlertSubscriptionCreateRequest(BaseModel):
    """Request to create an alert subscription."""

    watchlist_id: str = Field(..., description="Watchlist to monitor")
    rule_type: str = Field(
        "trade", description="Alert rule type (trade, large_trade, price_change)"
    )
    rule_config: dict[str, Any] = Field(
        default_factory=dict, description="Rule configuration"
    )
    cooldown_seconds: int = Field(
        300, ge=0, le=86400, description="Cooldown between alerts (seconds)"
    )


class AlertSchema(BaseModel):
    """An alert triggered by a subscription."""

    id: str = Field(..., description="Alert ID")
    subscription_id: str = Field(..., description="Subscription that triggered this")
    event_id: str = Field(..., description="Event ID that triggered the alert")
    market_id: str = Field(..., description="Market ID")
    wallet_address: str = Field(..., description="Wallet address that triggered")
    trade_data: dict[str, Any] | None = Field(
        None, description="Trade details (price, amount, etc.)"
    )
    acknowledged: bool = Field(False, description="Whether alert has been acknowledged")
    acknowledged_at: datetime | None = Field(
        None, description="When alert was acknowledged"
    )
    triggered_at: datetime = Field(..., description="When alert was triggered")


class AlertListResponse(BaseModel):
    """Response for listing alerts."""

    alerts: list[AlertSchema] = Field(..., description="List of alerts")
    count: int = Field(..., description="Number of alerts returned")
    has_more: bool = Field(..., description="Whether more alerts exist")


class AnalyticsSessionSchema(BaseModel):
    """An analytics session with idle TTL tracking."""

    session_id: str = Field(..., description="Session ID")
    started_at: datetime = Field(..., description="Session start time")
    ended_at: datetime | None = Field(None, description="Session end time")
    last_activity_at: datetime = Field(..., description="Last activity timestamp")
    idle_timeout_minutes: int = Field(
        120, description="Minutes of inactivity before auto-termination"
    )
    status: str = Field(
        "active", description="Session status (active, ended, expired)"
    )
    queries_run: int = Field(0, description="Number of queries executed")
    rows_accessed: int = Field(0, description="Total rows accessed")
    updated_at: datetime = Field(..., description="Last update timestamp")


class AnalyticsSessionCreateRequest(BaseModel):
    """Request to create an analytics session."""

    idle_timeout_minutes: int = Field(
        120, ge=1, le=1440, description="Idle timeout in minutes (max 24 hours)"
    )


# =============================================================================
# Runtime Mode Schemas
# =============================================================================


class IngestMode(str, Enum):
    """Ingestion mode controlling data pipeline behavior."""

    OFF = "off"  # Disables all ingestion except manual backfill
    BATCHED = "batched"  # Runs update_to_now() on 5-minute cadence
    LIVE = "live"  # Long-lived ingest/consumer loop


class AnalyticsMode(str, Enum):
    """Analytics mode controlling query infrastructure behavior."""

    OFF = "off"  # Analytics infrastructure disabled
    ON_DEMAND = "on_demand"  # Start when needed, stop after TTL
    LIVE = "live"  # Always available


class ModeStateSchema(BaseModel):
    """Current runtime mode state."""

    mode_name: str = Field(..., description="Mode identifier (ingest_mode or analytics_mode)")
    mode_value: str = Field(..., description="Current mode value")
    previous_value: str | None = Field(None, description="Previous mode value")
    transitioned_at: datetime = Field(..., description="When the mode was last changed")
    transitioned_by: str = Field("system", description="Who initiated the transition")
    is_transitioning: bool = Field(
        False, description="Whether a transition is in progress"
    )
    transition_started_at: datetime | None = Field(
        None, description="When the current transition started"
    )
    updated_at: datetime = Field(..., description="Last update timestamp")


class ModeTransitionRequest(BaseModel):
    """Request to transition a mode to a new value."""

    target_value: str = Field(..., description="Target mode value")
    initiated_by: str = Field("api", description="Who is initiating the transition")
    force: bool = Field(
        False, description="Force transition even if one is in progress"
    )


class ModeTransitionResponse(BaseModel):
    """Response from a mode transition request."""

    success: bool = Field(..., description="Whether transition was successful")
    mode_name: str = Field(..., description="Mode that was transitioned")
    previous_value: str = Field(..., description="Previous mode value")
    new_value: str = Field(..., description="New mode value")
    transitioned_at: datetime = Field(..., description="Transition timestamp")
    message: str = Field(..., description="Transition result message")


class RuntimeStatusSchema(BaseModel):
    """Current runtime status including all modes."""

    ingest_mode: ModeStateSchema = Field(..., description="Ingestion mode state")
    analytics_mode: ModeStateSchema = Field(..., description="Analytics mode state")


# =============================================================================
# Wallet Position Tracking Schemas
# =============================================================================


class PositionSchema(BaseModel):
    """A wallet's position in a specific market and outcome."""

    id: str = Field(..., description="Position ID (wallet_market_outcome hash)")
    wallet_address: str = Field(..., description="Wallet address (lowercase)")
    market_id: str = Field(..., description="Market ID")
    outcome: str = Field(..., description="Outcome token (YES or NO)")
    current_size: float = Field(..., description="Current position size in tokens")
    total_cost_basis: float = Field(
        ..., description="Total USD spent to build position"
    )
    average_cost: float | None = Field(
        None, description="Average cost per token (cost_basis / size)"
    )
    realized_pnl: float = Field(0.0, description="Realized P&L from closed portions")
    last_trade_price: float | None = Field(
        None, description="Last trade price for this outcome"
    )
    last_price_timestamp: datetime | None = Field(
        None, description="When last_trade_price was observed"
    )
    mtm_pnl: float | None = Field(
        None, description="Mark-to-market unrealized P&L"
    )
    mtm_window_start: datetime | None = Field(
        None, description="Start of current 5-minute MTM window"
    )
    first_trade_at: datetime = Field(..., description="When position was first opened")
    updated_at: datetime = Field(..., description="Last update timestamp")


class PositionUpdateEvent(BaseModel):
    """Event representing a trade that updates a position."""

    wallet_address: str = Field(..., description="Wallet address")
    market_id: str = Field(..., description="Market ID")
    outcome: str = Field(..., description="Outcome token (YES or NO)")
    is_buy: bool = Field(..., description="True if buying tokens, False if selling")
    quantity: float = Field(..., description="Token quantity traded")
    price: float = Field(..., description="Price per token (0-1)")
    timestamp: datetime = Field(..., description="Trade timestamp")
    transaction_hash: str = Field(..., description="Transaction hash for dedup")


class MTMSnapshotSchema(BaseModel):
    """A mark-to-market snapshot for a position at a 5-minute boundary."""

    id: str = Field(..., description="Snapshot ID")
    position_id: str = Field(..., description="Position being snapshotted")
    wallet_address: str = Field(..., description="Wallet address")
    market_id: str = Field(..., description="Market ID")
    outcome: str = Field(..., description="Outcome token")
    window_start: datetime = Field(..., description="5-minute window start")
    position_size: float = Field(..., description="Position size at snapshot")
    last_trade_price: float | None = Field(
        None, description="Last trade price in window (or carried forward)"
    )
    mtm_pnl: float | None = Field(None, description="MTM P&L at snapshot")
    average_cost: float | None = Field(None, description="Average cost at snapshot")
    created_at: datetime = Field(..., description="Snapshot creation time")


class ClosedPositionSchema(BaseModel):
    """A record of a position that was fully closed (size returned to zero)."""

    id: str = Field(..., description="Closed position record ID")
    wallet_address: str = Field(..., description="Wallet address")
    market_id: str = Field(..., description="Market ID")
    outcome: str = Field(..., description="Outcome token")
    realized_pnl: float = Field(..., description="Total realized P&L")
    average_cost: float = Field(..., description="Average entry cost")
    exit_price: float = Field(..., description="Price at which position closed")
    first_trade_at: datetime = Field(..., description="When position opened")
    closed_at: datetime = Field(..., description="When position closed")


class WalletMetricsSchema(BaseModel):
    """Aggregated metrics for a wallet's trading performance."""

    wallet_address: str = Field(..., description="Wallet address")
    win_count: int = Field(0, description="Number of winning closed positions")
    loss_count: int = Field(0, description="Number of losing closed positions")
    win_percentage: float | None = Field(
        None, description="Win rate (excluding zero P&L positions)"
    )
    total_realized_pnl: float = Field(0.0, description="Sum of realized P&L")
    total_unrealized_pnl: float | None = Field(
        None, description="Sum of MTM P&L for open positions"
    )
    open_positions_count: int = Field(0, description="Number of open positions")
    closed_positions_count: int = Field(0, description="Number of closed positions")
    total_volume_usd: float = Field(0.0, description="Total notional traded")


class PositionListResponse(BaseModel):
    """Response for listing positions."""

    positions: list[PositionSchema] = Field(..., description="List of positions")
    count: int = Field(..., description="Number of positions returned")


class MTMProcessingResult(BaseModel):
    """Result of processing a 5-minute MTM window."""

    window_start: datetime = Field(..., description="Window start time")
    window_end: datetime = Field(..., description="Window end time")
    positions_updated: int = Field(0, description="Positions with price updates")
    positions_carried_forward: int = Field(
        0, description="Positions with carried-forward prices"
    )
    snapshots_created: int = Field(0, description="MTM snapshots created")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
