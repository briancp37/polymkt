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
