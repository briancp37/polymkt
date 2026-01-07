"""Backtest engine for executing trading strategies.

This module implements the core backtest execution logic for the "buy the favorite"
strategy. The strategy:
1. At 90 days to expiry, identify the favorite market per election group (highest YES price)
2. Enter a long position in the favorite at the entry price
3. Hold until market expiry (closedTime)
4. Exit at the final price (assumed to be 1.0 for winners, 0.0 for losers)

PRD Reference:
- "Backtest v1: buy the favorite (highest YES price) at 90 days to expiry and hold to expiry"
- "Select a dataset/universe containing 100+ markets with valid election_group_id and closedTime"
- "Generate favorite signals at the 90-day snapshot per group (highest YES price)"
- "Simulate entry (one position per group) and hold until closedTime"
- "Compute per-trade and aggregate PnL and summary metrics"
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from polymkt.models.schemas import (
    BacktestMetrics,
    BacktestTradeRecord,
    BacktestUpdateRequest,
    StrategyConfig,
)
from polymkt.signals.favorites import FavoriteSignalStore
from polymkt.storage.backtests import BacktestStore
from polymkt.storage.datasets import DatasetStore
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.election_groups import ElectionGroupStore

logger = structlog.get_logger()


@dataclass
class BacktestExecutionResult:
    """Result of executing a backtest."""

    backtest_id: str
    status: str  # "completed" or "failed"
    metrics: BacktestMetrics | None
    trades: list[BacktestTradeRecord]
    equity_curve: list[dict[str, Any]]
    error_message: str | None = None


class BacktestEngine:
    """Engine for executing backtests on datasets using pre-computed signals.

    The engine orchestrates:
    1. Loading the dataset and its markets
    2. Loading pre-computed favorite signals at the specified snapshot
    3. For each election group with a favorite signal:
       - Find the entry trade (at snapshot days-to-exp)
       - Find the exit price (at market expiry)
       - Calculate PnL with fees and slippage
    4. Aggregate metrics across all trades
    5. Persist results to the backtest store
    """

    def __init__(
        self,
        duckdb_layer: DuckDBLayer,
        dataset_store: DatasetStore,
        backtest_store: BacktestStore,
        election_group_store: ElectionGroupStore,
        favorite_signal_store: FavoriteSignalStore,
    ) -> None:
        self.duckdb = duckdb_layer
        self.datasets = dataset_store
        self.backtests = backtest_store
        self.election_groups = election_group_store
        self.favorite_signals = favorite_signal_store

    def execute(self, backtest_id: str) -> BacktestExecutionResult:
        """Execute a backtest by ID.

        The backtest must already exist in the store with status='pending'.
        This method:
        1. Updates status to 'running'
        2. Executes the strategy
        3. Updates status to 'completed' or 'failed' with results

        Args:
            backtest_id: ID of the backtest to execute

        Returns:
            BacktestExecutionResult with the execution outcome
        """
        # Get the backtest configuration
        backtest = self.backtests.get_backtest(backtest_id)

        if backtest.status != "pending":
            raise ValueError(
                f"Backtest {backtest_id} is not pending (status={backtest.status})"
            )

        # Update to running
        self.backtests.update_backtest(
            backtest_id, BacktestUpdateRequest(status="running")
        )

        try:
            # Execute the strategy
            result = self._execute_strategy(
                backtest.dataset_id, backtest.strategy_config
            )

            # Update with results
            self.backtests.update_backtest(
                backtest_id,
                BacktestUpdateRequest(
                    status="completed",
                    metrics=result.metrics,
                    trades=result.trades,
                    equity_curve=result.equity_curve,
                ),
            )

            logger.info(
                "backtest_completed",
                backtest_id=backtest_id,
                trade_count=len(result.trades),
                total_return=result.metrics.total_return if result.metrics else None,
            )

            return BacktestExecutionResult(
                backtest_id=backtest_id,
                status="completed",
                metrics=result.metrics,
                trades=result.trades,
                equity_curve=result.equity_curve,
            )

        except Exception as e:
            error_msg = str(e)
            self.backtests.update_backtest(
                backtest_id,
                BacktestUpdateRequest(status="failed", error_message=error_msg),
            )

            logger.error(
                "backtest_failed", backtest_id=backtest_id, error=error_msg
            )

            return BacktestExecutionResult(
                backtest_id=backtest_id,
                status="failed",
                metrics=None,
                trades=[],
                equity_curve=[],
                error_message=error_msg,
            )

    def _execute_strategy(
        self, dataset_id: str, config: StrategyConfig
    ) -> BacktestExecutionResult:
        """Execute the backtest strategy on a dataset.

        Args:
            dataset_id: ID of the dataset containing markets to trade
            config: Strategy configuration

        Returns:
            BacktestExecutionResult with trades and metrics
        """
        # Get the dataset
        dataset = self.datasets.get_dataset(dataset_id)
        market_ids = set(dataset.market_ids) - set(dataset.excluded_market_ids)

        if not market_ids:
            raise ValueError(f"Dataset {dataset_id} has no markets after exclusions")

        # Get target days to expiry from config (default 90)
        entry_days_to_exp = config.entry_days_to_exp or 90.0
        tolerance = 0.5  # Default tolerance for snapshot matching

        # Get favorite signals at the snapshot
        signals = self.favorite_signals.get_signals_for_snapshot(
            snapshot_days_to_exp=entry_days_to_exp, tolerance=tolerance
        )

        if not signals:
            raise ValueError(
                f"No favorite signals found at {entry_days_to_exp} days to expiry. "
                "Run /api/favorite-signals/compute first."
            )

        logger.info(
            "executing_strategy",
            dataset_id=dataset_id,
            market_count=len(market_ids),
            signal_count=len(signals),
            entry_days_to_exp=entry_days_to_exp,
        )

        # Build trades for each signal where the favorite is in our dataset
        trades: list[BacktestTradeRecord] = []
        equity_curve: list[dict[str, Any]] = []
        cumulative_pnl = 0.0

        for signal in signals:
            favorite_market_id = signal["favorite_market_id"]

            # Only trade if the favorite is in our dataset
            if favorite_market_id not in market_ids:
                logger.debug(
                    "skipping_signal_not_in_dataset",
                    group_id=signal["election_group_id"],
                    favorite_market_id=favorite_market_id,
                )
                continue

            # Find entry trade
            entry_result = self._find_entry_trade(
                favorite_market_id, entry_days_to_exp, tolerance
            )

            if entry_result is None:
                logger.warning(
                    "no_entry_trade_found",
                    group_id=signal["election_group_id"],
                    market_id=favorite_market_id,
                )
                continue

            entry_time, entry_price = entry_result

            # Find exit (market close)
            exit_result = self._find_exit_at_expiry(favorite_market_id)

            if exit_result is None:
                logger.warning(
                    "no_exit_found",
                    group_id=signal["election_group_id"],
                    market_id=favorite_market_id,
                )
                continue

            exit_time, exit_price = exit_result

            # Calculate PnL with fees and slippage
            position_size = config.position_size
            gross_pnl = (exit_price - entry_price) * position_size

            # Apply fees (as percentage of position)
            fees = position_size * config.fee_rate * 2  # Entry + exit fees

            # Apply slippage (as percentage impact on prices)
            slippage_cost = position_size * config.slippage_rate * 2

            net_pnl = gross_pnl - fees - slippage_cost

            trade = BacktestTradeRecord(
                market_id=favorite_market_id,
                election_group_id=signal["election_group_id"],
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                position_size=position_size,
                pnl=net_pnl,
                fees_paid=fees,
                slippage_cost=slippage_cost,
            )
            trades.append(trade)

            cumulative_pnl += net_pnl
            equity_curve.append(
                {
                    "time": exit_time.isoformat() if exit_time else None,
                    "pnl": cumulative_pnl,
                    "trade_index": len(trades),
                }
            )

            logger.debug(
                "trade_recorded",
                market_id=favorite_market_id,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=net_pnl,
            )

        # Compute aggregate metrics
        metrics = self._compute_metrics(trades, config)

        return BacktestExecutionResult(
            backtest_id="",  # Will be set by caller
            status="completed",
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _find_entry_trade(
        self, market_id: str, target_days_to_exp: float, tolerance: float
    ) -> tuple[datetime, float] | None:
        """Find the entry trade for a market at the target days-to-exp.

        Returns the most recent trade within the tolerance window.

        Args:
            market_id: Market ID to find entry for
            target_days_to_exp: Target days to expiry
            tolerance: +/- tolerance around target

        Returns:
            Tuple of (entry_time, entry_price) or None if no trade found
        """
        trades, total = self.duckdb.query_trades_with_markets(
            market_id=market_id,
            days_to_exp_min=target_days_to_exp - tolerance,
            days_to_exp_max=target_days_to_exp + tolerance,
            order_by="timestamp",
            order_dir="DESC",
            limit=1,
        )

        if not trades:
            return None

        trade = trades[0]
        return trade["timestamp"], trade["price"]

    def _find_exit_at_expiry(
        self, market_id: str
    ) -> tuple[datetime, float] | None:
        """Find the exit trade at market expiry.

        For prediction markets, we assume the final price resolves to:
        - 1.0 if the market outcome is YES
        - 0.0 if the market outcome is NO

        We find the last trade before expiry and use its price to infer resolution.
        If price > 0.5, the market likely resolved YES (1.0).
        If price <= 0.5, the market likely resolved NO (0.0).

        Args:
            market_id: Market ID to find exit for

        Returns:
            Tuple of (exit_time, exit_price) or None if no trade found
        """
        # Get the last trade for this market (closest to expiry)
        trades, total = self.duckdb.query_trades_with_markets(
            market_id=market_id,
            days_to_exp_min=0,  # Only trades before or at expiry
            days_to_exp_max=1,  # Last day before expiry
            order_by="timestamp",
            order_dir="DESC",
            limit=1,
        )

        if not trades:
            # Fallback: get the very last trade for this market
            trades, _ = self.duckdb.query_trades_with_markets(
                market_id=market_id,
                order_by="timestamp",
                order_dir="DESC",
                limit=1,
            )

        if not trades:
            return None

        trade = trades[0]
        closed_time = trade.get("closed_time")

        # Determine exit price based on final trade price
        # In prediction markets, if price > 0.5 near expiry, outcome is likely YES (1.0)
        # If price <= 0.5, outcome is likely NO (0.0)
        final_price = trade["price"]
        if final_price > 0.5:
            exit_price = 1.0  # Market resolved YES
        else:
            exit_price = 0.0  # Market resolved NO

        exit_time = closed_time if closed_time else trade["timestamp"]
        return exit_time, exit_price

    def _compute_metrics(
        self, trades: list[BacktestTradeRecord], config: StrategyConfig
    ) -> BacktestMetrics:
        """Compute aggregate metrics from trade records.

        Args:
            trades: List of trade records
            config: Strategy configuration

        Returns:
            BacktestMetrics with computed values
        """
        if not trades:
            return BacktestMetrics(
                total_return=0.0,
                total_pnl=0.0,
                win_rate=0.0,
                trade_count=0,
                winning_trades=0,
                losing_trades=0,
                max_drawdown=0.0,
                sharpe_ratio=None,
                avg_trade_pnl=0.0,
                avg_holding_period_days=0.0,
            )

        # Count wins/losses
        pnls = [t.pnl or 0.0 for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        trade_count = len(trades)
        winning_trades = len(winning)
        losing_trades = len(losing)

        # Win rate
        win_rate = winning_trades / trade_count if trade_count > 0 else 0.0

        # Total return (as percentage of capital deployed)
        # Assume position_size is the capital per trade
        total_capital = trade_count * config.position_size
        total_return = (total_pnl / total_capital * 100) if total_capital > 0 else 0.0

        # Average PnL per trade
        avg_trade_pnl = total_pnl / trade_count if trade_count > 0 else 0.0

        # Average holding period
        holding_periods: list[float] = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                holding_periods.append(delta.total_seconds() / 86400.0)

        avg_holding_period_days = (
            sum(holding_periods) / len(holding_periods) if holding_periods else 0.0
        )

        # Max drawdown (from equity curve)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified: mean / std of daily returns)
        # Since we have per-trade PnL, not daily, this is a rough approximation
        sharpe_ratio = None
        if len(pnls) > 1:
            import statistics

            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                # Annualize assuming ~252 trading days, ~12 trades per year
                annualization = (252 / len(trades)) ** 0.5 if len(trades) < 252 else 1.0
                sharpe_ratio = (mean_pnl / std_pnl) * annualization

        return BacktestMetrics(
            total_return=round(total_return, 4),
            total_pnl=round(total_pnl, 6),
            win_rate=round(win_rate, 4),
            trade_count=trade_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=round(max_dd * 100, 4),  # As percentage
            sharpe_ratio=round(sharpe_ratio, 4) if sharpe_ratio else None,
            avg_trade_pnl=round(avg_trade_pnl, 6),
            avg_holding_period_days=round(avg_holding_period_days, 2),
        )


def create_backtest_engine(
    duckdb_path: Path,
    parquet_dir: Path,
    metadata_db_path: Path,
    partitioned: bool = False,
) -> BacktestEngine:
    """Create a backtest engine with all required dependencies.

    Args:
        duckdb_path: Path to DuckDB database file
        parquet_dir: Directory containing Parquet files
        metadata_db_path: Path to SQLite metadata database
        partitioned: Whether trades data is partitioned

    Returns:
        Configured BacktestEngine instance
    """
    duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=partitioned)
    duckdb_layer.create_views()

    dataset_store = DatasetStore(metadata_db_path)
    backtest_store = BacktestStore(metadata_db_path)
    election_group_store = ElectionGroupStore(metadata_db_path)
    favorite_signal_store = FavoriteSignalStore(metadata_db_path)

    return BacktestEngine(
        duckdb_layer=duckdb_layer,
        dataset_store=dataset_store,
        backtest_store=backtest_store,
        election_group_store=election_group_store,
        favorite_signal_store=favorite_signal_store,
    )
