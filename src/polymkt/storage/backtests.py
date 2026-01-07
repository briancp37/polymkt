"""Backtest storage for persisting backtest runs and results."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from polymkt.models.schemas import (
    BacktestCreateRequest,
    BacktestMetrics,
    BacktestSchema,
    BacktestSummary,
    BacktestTradeRecord,
    BacktestUpdateRequest,
    StrategyConfig,
)

logger = structlog.get_logger()


class BacktestNotFoundError(Exception):
    """Raised when a backtest is not found."""

    pass


class BacktestStore:
    """SQLite-backed storage for backtests (strategy runs and results)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    strategy_config TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    metrics TEXT,
                    trades TEXT,
                    equity_curve TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            # Create indices for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtests_dataset_id
                ON backtests(dataset_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtests_created_at
                ON backtests(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtests_status
                ON backtests(status)
            """)
            conn.commit()
        finally:
            conn.close()

    def create_backtest(self, request: BacktestCreateRequest) -> BacktestSchema:
        """Create a new backtest and return the created backtest."""
        now = datetime.now(timezone.utc)
        backtest_id = str(uuid.uuid4())

        backtest = BacktestSchema(
            id=backtest_id,
            dataset_id=request.dataset_id,
            strategy_config=request.strategy_config,
            status="pending",
            metrics=None,
            trades=None,
            equity_curve=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO backtests (
                    id, dataset_id, strategy_config, status, metrics, trades,
                    equity_curve, error_message, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    backtest.id,
                    backtest.dataset_id,
                    json.dumps(backtest.strategy_config.model_dump(mode="json")),
                    backtest.status,
                    None,
                    None,
                    None,
                    None,
                    backtest.created_at.isoformat(),
                    backtest.updated_at.isoformat(),
                    None,
                ),
            )
            conn.commit()
            logger.info(
                "backtest_created",
                backtest_id=backtest.id,
                dataset_id=backtest.dataset_id,
                strategy=backtest.strategy_config.name,
            )
        finally:
            conn.close()

        return backtest

    def get_backtest(self, backtest_id: str) -> BacktestSchema:
        """Get a backtest by ID. Raises BacktestNotFoundError if not found."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM backtests WHERE id = ?", (backtest_id,))
            row = cursor.fetchone()
            if row is None:
                raise BacktestNotFoundError(f"Backtest not found: {backtest_id}")
            return self._row_to_backtest(row)
        finally:
            conn.close()

    def update_backtest(
        self, backtest_id: str, request: BacktestUpdateRequest
    ) -> BacktestSchema:
        """Update a backtest. Returns the updated backtest. Raises BacktestNotFoundError if not found."""
        # First get the existing backtest
        existing = self.get_backtest(backtest_id)

        now = datetime.now(timezone.utc)

        # Determine if we're completing the backtest
        completed_at = existing.completed_at
        new_status = request.status if request.status is not None else existing.status
        if new_status == "completed" and existing.status != "completed":
            completed_at = now

        # Apply updates only for provided fields
        updated = BacktestSchema(
            id=existing.id,
            dataset_id=existing.dataset_id,
            strategy_config=existing.strategy_config,
            status=new_status,
            metrics=request.metrics if request.metrics is not None else existing.metrics,
            trades=request.trades if request.trades is not None else existing.trades,
            equity_curve=(
                request.equity_curve if request.equity_curve is not None else existing.equity_curve
            ),
            error_message=(
                request.error_message if request.error_message is not None else existing.error_message
            ),
            created_at=existing.created_at,
            updated_at=now,
            completed_at=completed_at,
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE backtests SET
                    status = ?,
                    metrics = ?,
                    trades = ?,
                    equity_curve = ?,
                    error_message = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (
                    updated.status,
                    json.dumps(updated.metrics.model_dump(mode="json")) if updated.metrics else None,
                    json.dumps([t.model_dump(mode="json") for t in updated.trades]) if updated.trades else None,
                    json.dumps(updated.equity_curve) if updated.equity_curve else None,
                    updated.error_message,
                    updated.updated_at.isoformat(),
                    updated.completed_at.isoformat() if updated.completed_at else None,
                    backtest_id,
                ),
            )
            conn.commit()
            logger.info("backtest_updated", backtest_id=backtest_id, status=updated.status)
        finally:
            conn.close()

        return updated

    def delete_backtest(self, backtest_id: str) -> None:
        """Delete a backtest. Raises BacktestNotFoundError if not found."""
        # First check that the backtest exists
        self.get_backtest(backtest_id)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM backtests WHERE id = ?", (backtest_id,))
            conn.commit()
            logger.info("backtest_deleted", backtest_id=backtest_id)
        finally:
            conn.close()

    def list_backtests(
        self,
        limit: int = 50,
        offset: int = 0,
        dataset_id: str | None = None,
    ) -> tuple[list[BacktestSummary], int]:
        """List backtests with pagination. Returns (summaries, total_count)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Build query based on filters
            where_clause = ""
            params: list[str | int] = []
            if dataset_id is not None:
                where_clause = "WHERE dataset_id = ?"
                params.append(dataset_id)

            # Get total count
            cursor = conn.execute(f"SELECT COUNT(*) FROM backtests {where_clause}", params)
            total_count = cursor.fetchone()[0]

            # Get paginated results
            params.extend([limit, offset])
            cursor = conn.execute(
                f"""
                SELECT id, dataset_id, strategy_config, status, metrics,
                       created_at, completed_at
                FROM backtests
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )
            summaries = [self._row_to_summary(row) for row in cursor.fetchall()]

            return summaries, total_count
        finally:
            conn.close()

    def _row_to_backtest(self, row: sqlite3.Row) -> BacktestSchema:
        """Convert a database row to a BacktestSchema."""
        strategy_config_data = json.loads(row["strategy_config"])
        metrics_data = json.loads(row["metrics"]) if row["metrics"] else None
        trades_data = json.loads(row["trades"]) if row["trades"] else None
        equity_curve_data = json.loads(row["equity_curve"]) if row["equity_curve"] else None

        return BacktestSchema(
            id=row["id"],
            dataset_id=row["dataset_id"],
            strategy_config=StrategyConfig(**strategy_config_data),
            status=row["status"],
            metrics=BacktestMetrics(**metrics_data) if metrics_data else None,
            trades=[BacktestTradeRecord(**t) for t in trades_data] if trades_data else None,
            equity_curve=equity_curve_data,
            error_message=row["error_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    def _row_to_summary(self, row: sqlite3.Row) -> BacktestSummary:
        """Convert a database row to a BacktestSummary."""
        strategy_config_data = json.loads(row["strategy_config"])
        metrics_data = json.loads(row["metrics"]) if row["metrics"] else None

        return BacktestSummary(
            id=row["id"],
            dataset_id=row["dataset_id"],
            strategy_name=strategy_config_data.get("name", "unknown"),
            status=row["status"],
            total_return=metrics_data.get("total_return") if metrics_data else None,
            trade_count=metrics_data.get("trade_count") if metrics_data else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )
