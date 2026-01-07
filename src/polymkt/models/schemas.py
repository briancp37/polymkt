"""Data schemas for markets, trades, and orders."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    schema_version: str
    parquet_files: list[str]
    rows_quarantined: dict[str, int] = Field(
        default_factory=dict, description="Rows quarantined per entity due to validation errors"
    )
