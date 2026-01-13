"""Position tracking service with average-cost accounting and 5-minute MTM.

Implements wallet position tracking per the PRD requirements:
- Maintain per-wallet per-market per-outcome positions with average cost
- Compute realized P&L on position size returning to zero
- Compute mark-to-market P&L every 5 minutes using last trade price
- Carry forward last known price if no trade in window; null until first print
"""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import structlog

from polymkt.models.schemas import (
    PositionSchema,
    PositionUpdateEvent,
    MTMProcessingResult,
)
from polymkt.storage.metadata import MetadataStore

logger = structlog.get_logger()


class PositionTracker:
    """
    Tracks wallet positions with average-cost accounting.

    This service processes trades and maintains position state:
    - On buy: increase position size, add to cost basis, recalculate average cost
    - On sell: decrease position size, compute realized P&L if closing
    - When size reaches zero: record closed position with realized P&L
    """

    def __init__(self, metadata_store: MetadataStore) -> None:
        self.store = metadata_store

    def process_trade(
        self,
        wallet_address: str,
        market_id: str,
        outcome: str,
        is_buy: bool,
        quantity: float,
        price: float,
        timestamp: datetime,
        transaction_hash: str,
    ) -> PositionSchema | None:
        """
        Process a trade and update the corresponding position.

        Uses average-cost accounting:
        - Buys add to position at traded price
        - Sells reduce position and realize P&L based on average cost

        Args:
            wallet_address: Wallet address (will be normalized to lowercase)
            market_id: Market ID
            outcome: Outcome token (YES or NO)
            is_buy: True if buying tokens, False if selling
            quantity: Token quantity traded
            price: Price per token (0-1)
            timestamp: Trade timestamp
            transaction_hash: Transaction hash for deduplication

        Returns:
            Updated position, or None if trade was already processed
        """
        wallet = wallet_address.lower()
        outcome = outcome.upper()

        # Check for duplicate trade
        if self.store.position_trade_exists(transaction_hash):
            logger.debug(
                "position_trade_duplicate",
                transaction_hash=transaction_hash[:10],
            )
            return None

        # Calculate USD amount
        usd_amount = quantity * price

        # Get or create position
        position = self.store.get_position(wallet, market_id, outcome)
        now = datetime.now(timezone.utc)

        if position is None:
            # Create new position
            position_id = self.store._generate_position_id(wallet, market_id, outcome)

            if is_buy:
                # Opening a new long position
                position = PositionSchema(
                    id=position_id,
                    wallet_address=wallet,
                    market_id=market_id,
                    outcome=outcome,
                    current_size=quantity,
                    total_cost_basis=usd_amount,
                    average_cost=price,
                    realized_pnl=0.0,
                    last_trade_price=price,
                    last_price_timestamp=timestamp,
                    mtm_pnl=None,  # Null until first MTM window
                    mtm_window_start=None,
                    first_trade_at=timestamp,
                    updated_at=now,
                )
            else:
                # Selling without existing position (short sale - create negative position)
                position = PositionSchema(
                    id=position_id,
                    wallet_address=wallet,
                    market_id=market_id,
                    outcome=outcome,
                    current_size=-quantity,
                    total_cost_basis=-usd_amount,  # Negative for short
                    average_cost=price,
                    realized_pnl=0.0,
                    last_trade_price=price,
                    last_price_timestamp=timestamp,
                    mtm_pnl=None,
                    mtm_window_start=None,
                    first_trade_at=timestamp,
                    updated_at=now,
                )
        else:
            # Update existing position
            old_size = position.current_size
            old_cost_basis = position.total_cost_basis
            old_avg_cost = position.average_cost or 0.0

            if is_buy:
                # Adding to position
                new_size = old_size + quantity
                new_cost_basis = old_cost_basis + usd_amount

                if abs(new_size) > 0.000001:
                    # Weighted average cost
                    new_avg_cost = new_cost_basis / new_size
                else:
                    new_avg_cost = None

                position = PositionSchema(
                    id=position.id,
                    wallet_address=position.wallet_address,
                    market_id=position.market_id,
                    outcome=position.outcome,
                    current_size=new_size,
                    total_cost_basis=new_cost_basis,
                    average_cost=new_avg_cost,
                    realized_pnl=position.realized_pnl,
                    last_trade_price=price,
                    last_price_timestamp=timestamp,
                    mtm_pnl=position.mtm_pnl,
                    mtm_window_start=position.mtm_window_start,
                    first_trade_at=position.first_trade_at,
                    updated_at=now,
                )
            else:
                # Selling/reducing position
                new_size = old_size - quantity

                # Compute realized P&L on the portion being sold
                # P&L = (sale_price - avg_cost) * quantity_sold
                realized_on_trade = (price - old_avg_cost) * quantity
                new_realized_pnl = position.realized_pnl + realized_on_trade

                # Reduce cost basis proportionally
                if abs(old_size) > 0.000001:
                    proportion_remaining = new_size / old_size
                    new_cost_basis = old_cost_basis * proportion_remaining
                else:
                    new_cost_basis = 0.0

                if abs(new_size) > 0.000001:
                    new_avg_cost = old_avg_cost  # Average cost doesn't change on sale
                else:
                    new_avg_cost = None

                position = PositionSchema(
                    id=position.id,
                    wallet_address=position.wallet_address,
                    market_id=position.market_id,
                    outcome=position.outcome,
                    current_size=new_size,
                    total_cost_basis=new_cost_basis,
                    average_cost=new_avg_cost,
                    realized_pnl=new_realized_pnl,
                    last_trade_price=price,
                    last_price_timestamp=timestamp,
                    mtm_pnl=position.mtm_pnl,
                    mtm_window_start=position.mtm_window_start,
                    first_trade_at=position.first_trade_at,
                    updated_at=now,
                )

        # Record the trade
        self.store.record_position_trade(
            position_id=position.id,
            wallet_address=wallet,
            market_id=market_id,
            outcome=outcome,
            is_buy=is_buy,
            quantity=quantity,
            price=price,
            usd_amount=usd_amount,
            transaction_hash=transaction_hash,
            timestamp=timestamp,
        )

        # Check if position is closed (size returned to zero)
        if abs(position.current_size) < 0.000001:
            # Position closed - record and delete
            logger.info(
                "position_closed",
                wallet_address=wallet[:10],
                market_id=market_id[:10],
                outcome=outcome,
                realized_pnl=position.realized_pnl,
            )

            self.store.record_closed_position(
                wallet_address=wallet,
                market_id=market_id,
                outcome=outcome,
                realized_pnl=position.realized_pnl,
                average_cost=position.average_cost or price,
                exit_price=price,
                first_trade_at=position.first_trade_at,
                closed_at=timestamp,
            )

            self.store.delete_position(wallet, market_id, outcome)
            return None  # Position no longer exists
        else:
            # Update position in store
            self.store.upsert_position(position)
            return position


class MTMProcessor:
    """
    Processes mark-to-market snapshots every 5 minutes.

    Per PRD requirements:
    - Compute MTM P&L using last trade price in window
    - Carry forward last known price if no trade in window
    - Null until first price is observed
    """

    def __init__(self, metadata_store: MetadataStore) -> None:
        self.store = metadata_store

    def process_window(
        self,
        window_start: datetime,
        window_end: datetime,
        trade_prices: dict[str, float] | None = None,
    ) -> MTMProcessingResult:
        """
        Process a 5-minute MTM window for all open positions.

        Args:
            window_start: Start of the 5-minute window
            window_end: End of the 5-minute window
            trade_prices: Optional dict of position_id -> last_trade_price
                         If not provided, carries forward existing prices

        Returns:
            MTMProcessingResult with processing statistics
        """
        start_time = perf_counter()

        positions = self.store.list_all_positions()
        positions_updated = 0
        positions_carried_forward = 0
        snapshots_created = 0

        now = datetime.now(timezone.utc)

        for position in positions:
            # Check if we have a new price for this position
            new_price = None
            if trade_prices:
                new_price = trade_prices.get(position.id)

            if new_price is not None:
                # Update with new price
                updated_position = self._update_with_price(
                    position, new_price, window_start, now
                )
                positions_updated += 1
            elif position.last_trade_price is not None:
                # Carry forward existing price
                updated_position = self._carry_forward_price(
                    position, window_start, now
                )
                positions_carried_forward += 1
            else:
                # No price available yet - skip MTM
                continue

            # Record snapshot
            self.store.record_mtm_snapshot(updated_position, window_start)
            snapshots_created += 1

        processing_time_ms = (perf_counter() - start_time) * 1000

        logger.info(
            "mtm_window_processed",
            window_start=window_start.isoformat(),
            positions_updated=positions_updated,
            positions_carried_forward=positions_carried_forward,
            snapshots_created=snapshots_created,
            processing_time_ms=round(processing_time_ms, 2),
        )

        return MTMProcessingResult(
            window_start=window_start,
            window_end=window_end,
            positions_updated=positions_updated,
            positions_carried_forward=positions_carried_forward,
            snapshots_created=snapshots_created,
            processing_time_ms=processing_time_ms,
        )

    def _update_with_price(
        self,
        position: PositionSchema,
        new_price: float,
        window_start: datetime,
        now: datetime,
    ) -> PositionSchema:
        """Update position with new trade price and recalculate MTM."""
        # Compute MTM P&L: (current_price - avg_cost) * position_size
        mtm_pnl = None
        if position.average_cost is not None:
            mtm_pnl = (new_price - position.average_cost) * position.current_size

        updated = PositionSchema(
            id=position.id,
            wallet_address=position.wallet_address,
            market_id=position.market_id,
            outcome=position.outcome,
            current_size=position.current_size,
            total_cost_basis=position.total_cost_basis,
            average_cost=position.average_cost,
            realized_pnl=position.realized_pnl,
            last_trade_price=new_price,
            last_price_timestamp=now,
            mtm_pnl=mtm_pnl,
            mtm_window_start=window_start,
            first_trade_at=position.first_trade_at,
            updated_at=now,
        )

        self.store.upsert_position(updated)
        return updated

    def _carry_forward_price(
        self,
        position: PositionSchema,
        window_start: datetime,
        now: datetime,
    ) -> PositionSchema:
        """Carry forward last known price and recalculate MTM."""
        # Recompute MTM with carried-forward price
        mtm_pnl = None
        if position.average_cost is not None and position.last_trade_price is not None:
            mtm_pnl = (
                position.last_trade_price - position.average_cost
            ) * position.current_size

        updated = PositionSchema(
            id=position.id,
            wallet_address=position.wallet_address,
            market_id=position.market_id,
            outcome=position.outcome,
            current_size=position.current_size,
            total_cost_basis=position.total_cost_basis,
            average_cost=position.average_cost,
            realized_pnl=position.realized_pnl,
            last_trade_price=position.last_trade_price,  # Carried forward
            last_price_timestamp=position.last_price_timestamp,  # Keep original
            mtm_pnl=mtm_pnl,
            mtm_window_start=window_start,
            first_trade_at=position.first_trade_at,
            updated_at=now,
        )

        self.store.upsert_position(updated)
        return updated


def get_5min_window_boundaries(timestamp: datetime) -> tuple[datetime, datetime]:
    """
    Get the 5-minute window boundaries for a given timestamp.

    Returns (window_start, window_end) aligned to 5-minute intervals.
    """
    # Truncate to 5-minute boundary
    minute = (timestamp.minute // 5) * 5
    window_start = timestamp.replace(minute=minute, second=0, microsecond=0)
    window_end = window_start.replace(minute=minute + 5) if minute < 55 else (
        window_start.replace(hour=window_start.hour + 1, minute=0)
        if window_start.hour < 23
        else window_start.replace(day=window_start.day + 1, hour=0, minute=0)
    )

    return window_start, window_end


def get_rollup_window_boundaries(
    timestamp: datetime, interval: str
) -> tuple[datetime, datetime]:
    """
    Get the window boundaries for a given timestamp and interval.

    Args:
        timestamp: The timestamp to align
        interval: The rollup interval (1m, 1h, 1d)

    Returns:
        (window_start, window_end) tuple aligned to the interval
    """
    from datetime import timedelta

    if interval == "1m":
        # Align to minute boundary
        window_start = timestamp.replace(second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=1)
    elif interval == "1h":
        # Align to hour boundary
        window_start = timestamp.replace(minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(hours=1)
    elif interval == "1d":
        # Align to day boundary (UTC)
        window_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(days=1)
    else:
        raise ValueError(f"Invalid interval: {interval}. Must be 1m, 1h, or 1d")

    return window_start, window_end
