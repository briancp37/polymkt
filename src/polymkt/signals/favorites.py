"""Favorite signal computation for election groups.

This module computes the "favorite" market within each election group at a
specified days-to-expiry snapshot. The favorite is defined as the market with
the highest YES price at the snapshot time.

PRD Reference:
- "Define 'favorite' as highest YES price at the 90-days-to-exp snapshot per election group"
- For each market_id, compute snapshot price at (days_to_exp ~= 90) using last trade price
- For each election_group_id, select the market_id with max YES price at that snapshot
- Ties are handled deterministically (by market_id alphabetically)
- Favorites are persisted as a signal table for backtests
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.election_groups import ElectionGroupStore

logger = structlog.get_logger()


@dataclass
class SnapshotPrice:
    """Price data for a market at a specific snapshot."""

    market_id: str
    price: float  # YES price (0-1)
    timestamp: datetime
    days_to_exp: float
    question: str | None = None


@dataclass
class FavoriteSignal:
    """A favorite signal for an election group at a specific snapshot."""

    election_group_id: str
    election_group_name: str
    favorite_market_id: str
    favorite_price: float
    favorite_question: str | None
    snapshot_days_to_exp: float
    computed_at: datetime
    all_market_prices: dict[str, float]  # market_id -> price for all markets in group


@dataclass
class FavoriteComputeResult:
    """Result of computing favorites for all election groups."""

    signals: list[FavoriteSignal]
    groups_processed: int
    groups_with_data: int
    groups_without_data: int  # Groups where no trades at snapshot
    total_markets: int
    markets_with_trades: int
    markets_without_trades: int
    snapshot_days_to_exp: float
    tolerance: float  # +/- days around target snapshot


class FavoriteSignalStore:
    """SQLite-backed storage for favorite signals.

    Persists computed favorites so they can be used in backtests.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS favorite_signals (
                    id TEXT PRIMARY KEY,
                    election_group_id TEXT NOT NULL,
                    election_group_name TEXT NOT NULL,
                    favorite_market_id TEXT NOT NULL,
                    favorite_price REAL NOT NULL,
                    favorite_question TEXT,
                    snapshot_days_to_exp REAL NOT NULL,
                    all_market_prices TEXT NOT NULL,
                    computed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # Index for efficient lookups by group and snapshot
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fs_group_snapshot
                ON favorite_signals(election_group_id, snapshot_days_to_exp)
            """)
            # Index for lookups by snapshot
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fs_snapshot
                ON favorite_signals(snapshot_days_to_exp)
            """)
            conn.commit()
        finally:
            conn.close()

    def save_signals(self, signals: list[FavoriteSignal]) -> int:
        """Save computed favorite signals to the database.

        Args:
            signals: List of favorite signals to save

        Returns:
            Number of signals saved
        """
        if not signals:
            return 0

        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        saved = 0

        try:
            for signal in signals:
                signal_id = str(uuid.uuid4())
                # Serialize all_market_prices as JSON string
                import json

                prices_json = json.dumps(signal.all_market_prices)

                conn.execute(
                    """
                    INSERT INTO favorite_signals
                    (id, election_group_id, election_group_name, favorite_market_id,
                     favorite_price, favorite_question, snapshot_days_to_exp,
                     all_market_prices, computed_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal_id,
                        signal.election_group_id,
                        signal.election_group_name,
                        signal.favorite_market_id,
                        signal.favorite_price,
                        signal.favorite_question,
                        signal.snapshot_days_to_exp,
                        prices_json,
                        signal.computed_at.isoformat(),
                        now,
                    ),
                )
                saved += 1

            conn.commit()
            logger.info("favorite_signals_saved", count=saved)
        finally:
            conn.close()

        return saved

    def get_signals_for_snapshot(
        self,
        snapshot_days_to_exp: float,
        tolerance: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Get all favorite signals for a specific snapshot.

        Args:
            snapshot_days_to_exp: Target days to expiry
            tolerance: +/- tolerance for matching snapshot

        Returns:
            List of favorite signal dicts
        """
        import json

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT * FROM favorite_signals
                WHERE snapshot_days_to_exp >= ? AND snapshot_days_to_exp <= ?
                ORDER BY election_group_name
                """,
                (snapshot_days_to_exp - tolerance, snapshot_days_to_exp + tolerance),
            )
            return [
                {
                    "id": row["id"],
                    "election_group_id": row["election_group_id"],
                    "election_group_name": row["election_group_name"],
                    "favorite_market_id": row["favorite_market_id"],
                    "favorite_price": row["favorite_price"],
                    "favorite_question": row["favorite_question"],
                    "snapshot_days_to_exp": row["snapshot_days_to_exp"],
                    "all_market_prices": json.loads(row["all_market_prices"]),
                    "computed_at": datetime.fromisoformat(row["computed_at"]),
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_signal_for_group(
        self,
        election_group_id: str,
        snapshot_days_to_exp: float,
        tolerance: float = 0.5,
    ) -> dict[str, Any] | None:
        """Get the favorite signal for a specific group at a snapshot.

        Args:
            election_group_id: The election group ID
            snapshot_days_to_exp: Target days to expiry
            tolerance: +/- tolerance for matching snapshot

        Returns:
            Favorite signal dict or None if not found
        """
        import json

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT * FROM favorite_signals
                WHERE election_group_id = ?
                  AND snapshot_days_to_exp >= ?
                  AND snapshot_days_to_exp <= ?
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                (
                    election_group_id,
                    snapshot_days_to_exp - tolerance,
                    snapshot_days_to_exp + tolerance,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return {
                "id": row["id"],
                "election_group_id": row["election_group_id"],
                "election_group_name": row["election_group_name"],
                "favorite_market_id": row["favorite_market_id"],
                "favorite_price": row["favorite_price"],
                "favorite_question": row["favorite_question"],
                "snapshot_days_to_exp": row["snapshot_days_to_exp"],
                "all_market_prices": json.loads(row["all_market_prices"]),
                "computed_at": datetime.fromisoformat(row["computed_at"]),
                "created_at": datetime.fromisoformat(row["created_at"]),
            }
        finally:
            conn.close()

    def clear_signals_for_snapshot(self, snapshot_days_to_exp: float, tolerance: float = 0.5) -> int:
        """Clear all signals for a specific snapshot (before recomputing).

        Args:
            snapshot_days_to_exp: Target days to expiry
            tolerance: +/- tolerance for matching snapshot

        Returns:
            Number of signals deleted
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                DELETE FROM favorite_signals
                WHERE snapshot_days_to_exp >= ? AND snapshot_days_to_exp <= ?
                """,
                (snapshot_days_to_exp - tolerance, snapshot_days_to_exp + tolerance),
            )
            deleted = cursor.rowcount
            conn.commit()
            logger.info("favorite_signals_cleared", count=deleted, snapshot=snapshot_days_to_exp)
            return deleted
        finally:
            conn.close()

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all unique snapshot days with signal counts.

        Returns:
            List of snapshots with counts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT snapshot_days_to_exp,
                       COUNT(*) as signal_count,
                       MAX(computed_at) as last_computed
                FROM favorite_signals
                GROUP BY snapshot_days_to_exp
                ORDER BY snapshot_days_to_exp
            """)
            return [
                {
                    "snapshot_days_to_exp": row["snapshot_days_to_exp"],
                    "signal_count": row["signal_count"],
                    "last_computed": datetime.fromisoformat(row["last_computed"]),
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()


def compute_snapshot_prices(
    duckdb_layer: DuckDBLayer,
    market_ids: list[str],
    target_days_to_exp: float = 90.0,
    tolerance: float = 0.5,
) -> dict[str, SnapshotPrice]:
    """Compute the snapshot price for each market at the target days-to-exp.

    The snapshot price is the last trade price before the snapshot time.
    Uses the trade closest to the target days_to_exp within tolerance.

    Args:
        duckdb_layer: DuckDB layer for querying trades
        market_ids: List of market IDs to get prices for
        target_days_to_exp: Target days to expiry (default 90)
        tolerance: +/- tolerance around target (default 0.5 days)

    Returns:
        Dict mapping market_id -> SnapshotPrice for markets with trades
    """
    if not market_ids:
        return {}

    # Query trades within the snapshot window, ordered by timestamp DESC
    # to get the most recent trade at the snapshot time
    trades, _ = duckdb_layer.query_trades_with_markets(
        market_ids=market_ids,
        days_to_exp_min=target_days_to_exp - tolerance,
        days_to_exp_max=target_days_to_exp + tolerance,
        order_by="timestamp",
        order_dir="DESC",
        limit=100000,  # Should be enough for reasonable market counts
    )

    # For each market, keep the most recent trade (first in DESC order)
    snapshot_prices: dict[str, SnapshotPrice] = {}
    for trade in trades:
        market_id = trade["market_id"]
        if market_id not in snapshot_prices:
            snapshot_prices[market_id] = SnapshotPrice(
                market_id=market_id,
                price=trade["price"],
                timestamp=trade["timestamp"],
                days_to_exp=trade["days_to_exp"],
                question=trade.get("question"),
            )

    logger.info(
        "snapshot_prices_computed",
        markets_requested=len(market_ids),
        markets_with_trades=len(snapshot_prices),
        target_days_to_exp=target_days_to_exp,
        tolerance=tolerance,
    )

    return snapshot_prices


def select_favorite(
    snapshot_prices: dict[str, SnapshotPrice],
) -> tuple[str, float] | None:
    """Select the favorite market (highest YES price) from snapshot prices.

    Ties are broken deterministically by selecting the market with the
    lexicographically smallest market_id.

    Args:
        snapshot_prices: Dict of market_id -> SnapshotPrice

    Returns:
        Tuple of (favorite_market_id, favorite_price) or None if no prices
    """
    if not snapshot_prices:
        return None

    # Sort by price DESC, then by market_id ASC for deterministic tie-breaking
    sorted_markets = sorted(
        snapshot_prices.items(),
        key=lambda x: (-x[1].price, x[0]),  # -price for DESC, market_id for ASC
    )

    favorite_market_id, favorite_snapshot = sorted_markets[0]
    return favorite_market_id, favorite_snapshot.price


def compute_favorites_for_groups(
    duckdb_layer: DuckDBLayer,
    election_group_store: ElectionGroupStore,
    target_days_to_exp: float = 90.0,
    tolerance: float = 0.5,
    group_ids: list[str] | None = None,
) -> FavoriteComputeResult:
    """Compute favorite signals for all election groups (or specified groups).

    For each election group:
    1. Get all market IDs in the group
    2. Query trades at the snapshot time (target_days_to_exp +/- tolerance)
    3. For each market, get the last trade price before snapshot
    4. Select the favorite as the market with highest YES price
    5. Handle ties deterministically (by market_id)

    Args:
        duckdb_layer: DuckDB layer for querying trades
        election_group_store: Store for election groups
        target_days_to_exp: Target days to expiry for snapshot (default 90)
        tolerance: +/- tolerance around target (default 0.5 days)
        group_ids: Optional list of specific group IDs to process

    Returns:
        FavoriteComputeResult with all computed signals and statistics
    """
    computed_at = datetime.now(timezone.utc)
    signals: list[FavoriteSignal] = []
    groups_with_data = 0
    groups_without_data = 0
    total_markets = 0
    markets_with_trades = 0

    # Get groups to process
    if group_ids:
        groups = [election_group_store.get_group(gid) for gid in group_ids]
    else:
        groups, _ = election_group_store.list_groups(limit=10000)
        # list_groups returns summaries, need to get full group data
        groups = [election_group_store.get_group(g["id"]) for g in groups]

    groups_processed = len(groups)

    for group in groups:
        group_id = group["id"]
        group_name = group["name"]
        market_ids = group["market_ids"]

        if not market_ids:
            logger.warning("election_group_empty", group_id=group_id, group_name=group_name)
            groups_without_data += 1
            continue

        total_markets += len(market_ids)

        # Compute snapshot prices for all markets in the group
        snapshot_prices = compute_snapshot_prices(
            duckdb_layer,
            market_ids,
            target_days_to_exp,
            tolerance,
        )

        markets_with_trades += len(snapshot_prices)

        if not snapshot_prices:
            logger.warning(
                "no_trades_at_snapshot",
                group_id=group_id,
                group_name=group_name,
                markets_count=len(market_ids),
                target_days_to_exp=target_days_to_exp,
            )
            groups_without_data += 1
            continue

        # Select the favorite
        favorite_result = select_favorite(snapshot_prices)
        if favorite_result is None:
            groups_without_data += 1
            continue

        favorite_market_id, favorite_price = favorite_result
        favorite_snapshot = snapshot_prices[favorite_market_id]

        # Build all_market_prices dict
        all_market_prices = {mid: sp.price for mid, sp in snapshot_prices.items()}

        signal = FavoriteSignal(
            election_group_id=group_id,
            election_group_name=group_name,
            favorite_market_id=favorite_market_id,
            favorite_price=favorite_price,
            favorite_question=favorite_snapshot.question,
            snapshot_days_to_exp=target_days_to_exp,
            computed_at=computed_at,
            all_market_prices=all_market_prices,
        )
        signals.append(signal)
        groups_with_data += 1

        logger.info(
            "favorite_computed",
            group_id=group_id,
            group_name=group_name,
            favorite_market_id=favorite_market_id,
            favorite_price=favorite_price,
            markets_with_prices=len(snapshot_prices),
        )

    return FavoriteComputeResult(
        signals=signals,
        groups_processed=groups_processed,
        groups_with_data=groups_with_data,
        groups_without_data=groups_without_data,
        total_markets=total_markets,
        markets_with_trades=markets_with_trades,
        markets_without_trades=total_markets - markets_with_trades,
        snapshot_days_to_exp=target_days_to_exp,
        tolerance=tolerance,
    )
