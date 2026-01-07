"""DuckDB layer for querying Parquet files."""

from pathlib import Path
from typing import Any

import duckdb
import structlog

logger = structlog.get_logger()

VIEW_DEFINITIONS = {
    "v_markets": """
        CREATE OR REPLACE VIEW v_markets AS
        SELECT
            id,
            question,
            created_at,
            answer1,
            answer2,
            neg_risk,
            market_slug,
            token1,
            token2,
            condition_id,
            volume,
            ticker,
            closed_time,
            description,
            category
        FROM read_parquet('{parquet_dir}/markets.parquet')
    """,
    "v_trades": """
        CREATE OR REPLACE VIEW v_trades AS
        SELECT
            timestamp,
            market_id,
            maker,
            taker,
            nonusdc_side,
            maker_direction,
            taker_direction,
            price,
            usd_amount,
            token_amount,
            transaction_hash
        FROM read_parquet('{parquet_dir}/trades.parquet')
    """,
    "v_order_filled": """
        CREATE OR REPLACE VIEW v_order_filled AS
        SELECT
            timestamp,
            maker,
            maker_asset_id,
            maker_amount_filled,
            taker,
            taker_asset_id,
            taker_amount_filled,
            transaction_hash
        FROM read_parquet('{parquet_dir}/order_filled.parquet')
    """,
    "v_trades_with_markets": """
        CREATE OR REPLACE VIEW v_trades_with_markets AS
        SELECT
            t.timestamp,
            t.market_id,
            t.maker,
            t.taker,
            t.nonusdc_side,
            t.maker_direction,
            t.taker_direction,
            t.price,
            t.usd_amount,
            t.token_amount,
            t.transaction_hash,
            m.question,
            m.category,
            m.closed_time,
            -- Derived field: days to expiry
            CASE
                WHEN m.closed_time IS NOT NULL
                THEN EXTRACT(EPOCH FROM (m.closed_time - t.timestamp)) / 86400.0
                ELSE NULL
            END AS days_to_exp
        FROM v_trades t
        LEFT JOIN v_markets m ON t.market_id = m.id
    """,
}


class DuckDBLayer:
    """DuckDB layer for querying Parquet files with pre-defined views."""

    def __init__(self, db_path: Path, parquet_dir: Path) -> None:
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get or create the database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.db_path))
            logger.info("duckdb_connected", path=str(self.db_path))
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("duckdb_closed")

    def create_views(self) -> list[str]:
        """Create all views over Parquet files."""
        created_views: list[str] = []
        parquet_dir_str = str(self.parquet_dir.absolute())

        for view_name, view_sql in VIEW_DEFINITIONS.items():
            sql = view_sql.format(parquet_dir=parquet_dir_str)
            self.conn.execute(sql)
            created_views.append(view_name)
            logger.info("view_created", view=view_name)

        return created_views

    def verify_views(self) -> dict[str, int]:
        """Verify all views are queryable and return row counts."""
        counts: dict[str, int] = {}
        for view_name in VIEW_DEFINITIONS:
            result = self.conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()
            counts[view_name] = result[0] if result else 0
            logger.info("view_verified", view=view_name, rows=counts[view_name])
        return counts

    def execute(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Execute a SQL query."""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)

    def query_trades(
        self,
        market_id: str | None = None,
        market_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        order_by: str = "timestamp",
        order_dir: str = "ASC",
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Query trades with optional filters.

        Args:
            market_id: Filter by single market ID
            market_ids: Filter by multiple market IDs
            start_time: Filter trades >= this timestamp
            end_time: Filter trades <= this timestamp
            limit: Max rows to return (default 1000)
            offset: Number of rows to skip (default 0)
            order_by: Column to sort by (default "timestamp").
                      Options: "timestamp", "price", "usd_amount", "token_amount"
                      Use "timestamp,transaction_hash" for stable ordering.
            order_dir: Sort direction "ASC" or "DESC" (default "ASC")

        Returns:
            Tuple of (trades list, total_count) where total_count is the
            total number of matching rows before pagination.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if market_id:
            conditions.append("market_id = ?")
            params.append(market_id)
        elif market_ids:
            placeholders = ", ".join(["?" for _ in market_ids])
            conditions.append(f"market_id IN ({placeholders})")
            params.extend(market_ids)

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Validate order_by to prevent SQL injection
        allowed_order_columns = {
            "timestamp",
            "price",
            "usd_amount",
            "token_amount",
            "market_id",
            "transaction_hash",
        }
        # Support composite ordering like "timestamp,transaction_hash"
        order_columns = [col.strip() for col in order_by.split(",")]
        for col in order_columns:
            if col not in allowed_order_columns:
                raise ValueError(
                    f"Invalid order_by column: {col}. "
                    f"Allowed: {allowed_order_columns}"
                )
        order_by_clause = ", ".join(order_columns)

        # Validate order_dir
        order_dir_upper = order_dir.upper()
        if order_dir_upper not in ("ASC", "DESC"):
            raise ValueError(f"Invalid order_dir: {order_dir}. Must be ASC or DESC")

        # First, get total count for pagination
        count_sql = f"""
            SELECT COUNT(*) FROM v_trades
            WHERE {where_clause}
        """
        count_result = self.conn.execute(count_sql, params).fetchone()
        total_count = count_result[0] if count_result else 0

        # Now get the paginated results
        sql = f"""
            SELECT * FROM v_trades
            WHERE {where_clause}
            ORDER BY {order_by_clause} {order_dir_upper}
            LIMIT ? OFFSET ?
        """
        query_params = params + [limit, offset]

        result = self.conn.execute(sql, query_params)
        columns = [desc[0] for desc in result.description]
        trades = [dict(zip(columns, row)) for row in result.fetchall()]

        return trades, total_count

    def query_trades_with_markets(
        self,
        market_id: str | None = None,
        market_ids: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        days_to_exp_min: float | None = None,
        days_to_exp_max: float | None = None,
        limit: int = 1000,
        offset: int = 0,
        order_by: str = "timestamp",
        order_dir: str = "ASC",
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Query trades joined with markets, including the days_to_exp derived field.

        Args:
            market_id: Filter by single market ID
            market_ids: Filter by multiple market IDs
            start_time: Filter trades >= this timestamp
            end_time: Filter trades <= this timestamp
            days_to_exp_min: Filter trades where days_to_exp >= this value
            days_to_exp_max: Filter trades where days_to_exp <= this value
            limit: Max rows to return (default 1000)
            offset: Number of rows to skip (default 0)
            order_by: Column to sort by (default "timestamp").
                      Options: "timestamp", "price", "usd_amount", "token_amount",
                               "market_id", "transaction_hash", "days_to_exp"
                      Use "timestamp,transaction_hash" for stable ordering.
            order_dir: Sort direction "ASC" or "DESC" (default "ASC")

        Returns:
            Tuple of (trades list, total_count) where total_count is the
            total number of matching rows before pagination.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if market_id:
            conditions.append("market_id = ?")
            params.append(market_id)
        elif market_ids:
            placeholders = ", ".join(["?" for _ in market_ids])
            conditions.append(f"market_id IN ({placeholders})")
            params.extend(market_ids)

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        if days_to_exp_min is not None:
            conditions.append("days_to_exp >= ?")
            params.append(days_to_exp_min)

        if days_to_exp_max is not None:
            conditions.append("days_to_exp <= ?")
            params.append(days_to_exp_max)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Validate order_by to prevent SQL injection
        allowed_order_columns = {
            "timestamp",
            "price",
            "usd_amount",
            "token_amount",
            "market_id",
            "transaction_hash",
            "days_to_exp",
            "question",
            "category",
            "closed_time",
        }
        # Support composite ordering like "timestamp,transaction_hash"
        order_columns = [col.strip() for col in order_by.split(",")]
        for col in order_columns:
            if col not in allowed_order_columns:
                raise ValueError(
                    f"Invalid order_by column: {col}. "
                    f"Allowed: {allowed_order_columns}"
                )
        order_by_clause = ", ".join(order_columns)

        # Validate order_dir
        order_dir_upper = order_dir.upper()
        if order_dir_upper not in ("ASC", "DESC"):
            raise ValueError(f"Invalid order_dir: {order_dir}. Must be ASC or DESC")

        # First, get total count for pagination
        count_sql = f"""
            SELECT COUNT(*) FROM v_trades_with_markets
            WHERE {where_clause}
        """
        count_result = self.conn.execute(count_sql, params).fetchone()
        total_count = count_result[0] if count_result else 0

        # Now get the paginated results
        sql = f"""
            SELECT * FROM v_trades_with_markets
            WHERE {where_clause}
            ORDER BY {order_by_clause} {order_dir_upper}
            LIMIT ? OFFSET ?
        """
        query_params = params + [limit, offset]

        result = self.conn.execute(sql, query_params)
        columns = [desc[0] for desc in result.description]
        trades = [dict(zip(columns, row)) for row in result.fetchall()]

        return trades, total_count
