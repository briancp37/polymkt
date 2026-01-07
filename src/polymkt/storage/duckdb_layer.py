"""DuckDB layer for querying Parquet files."""

from pathlib import Path
from typing import Any

import duckdb
import structlog

logger = structlog.get_logger()


def get_view_definitions(parquet_dir: str, partitioned: bool = False) -> dict[str, str]:
    """
    Get view definitions for DuckDB (legacy single-layer mode).

    Args:
        parquet_dir: Absolute path to parquet directory
        partitioned: Whether trades data is partitioned

    Returns:
        Dictionary of view name to SQL definition
    """
    # For partitioned trades, read from directory with hive partitioning
    if partitioned:
        trades_source = f"read_parquet('{parquet_dir}/trades/**/*.parquet', hive_partitioning=true)"
    else:
        trades_source = f"read_parquet('{parquet_dir}/trades.parquet')"

    return {
        "v_events": f"""
            CREATE OR REPLACE VIEW v_events AS
            SELECT
                event_id,
                tags,
                title,
                description,
                created_at
            FROM read_parquet('{parquet_dir}/events.parquet')
        """,
        "v_markets": f"""
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
                category,
                event_id,
                tags
            FROM read_parquet('{parquet_dir}/markets.parquet')
        """,
        "v_trades": f"""
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
            FROM {trades_source}
        """,
        "v_order_filled": f"""
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
                m.tags,
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


def get_layered_view_definitions(
    raw_dir: str,
    analytics_dir: str,
    partitioned: bool = False,
) -> dict[str, str]:
    """
    Get view definitions for DuckDB with raw/analytics layer separation.

    Creates views for both raw layer (immutable source data) and analytics
    layer (derived fields like days_to_exp).

    Args:
        raw_dir: Absolute path to raw parquet directory
        analytics_dir: Absolute path to analytics parquet directory
        partitioned: Whether trades data is partitioned

    Returns:
        Dictionary of view name to SQL definition
    """
    # Raw layer sources
    if partitioned:
        raw_trades_source = f"read_parquet('{raw_dir}/trades/**/*.parquet', hive_partitioning=true)"
        analytics_trades_source = f"read_parquet('{analytics_dir}/trades_analytics/**/*.parquet', hive_partitioning=true)"
    else:
        raw_trades_source = f"read_parquet('{raw_dir}/trades.parquet')"
        analytics_trades_source = f"read_parquet('{analytics_dir}/trades_analytics.parquet')"

    return {
        # Raw layer views (immutable, no derived fields)
        "v_events_raw": f"""
            CREATE OR REPLACE VIEW v_events_raw AS
            SELECT
                event_id,
                tags,
                title,
                description,
                created_at
            FROM read_parquet('{raw_dir}/events.parquet')
        """,
        "v_markets_raw": f"""
            CREATE OR REPLACE VIEW v_markets_raw AS
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
                category,
                event_id,
                tags
            FROM read_parquet('{raw_dir}/markets.parquet')
        """,
        "v_trades_raw": f"""
            CREATE OR REPLACE VIEW v_trades_raw AS
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
            FROM {raw_trades_source}
        """,
        "v_order_filled_raw": f"""
            CREATE OR REPLACE VIEW v_order_filled_raw AS
            SELECT
                timestamp,
                maker,
                maker_asset_id,
                maker_amount_filled,
                taker,
                taker_asset_id,
                taker_amount_filled,
                transaction_hash
            FROM read_parquet('{raw_dir}/order_filled.parquet')
        """,
        # Analytics layer views (with derived fields)
        "v_trades_analytics": f"""
            CREATE OR REPLACE VIEW v_trades_analytics AS
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
                transaction_hash,
                days_to_exp
            FROM {analytics_trades_source}
        """,
        # Convenience views that join analytics trades with market metadata
        "v_trades_with_markets": f"""
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
                t.days_to_exp,
                m.question,
                m.category,
                m.closed_time,
                m.tags
            FROM v_trades_analytics t
            LEFT JOIN v_markets_raw m ON t.market_id = m.id
        """,
        # Alias views for backward compatibility
        "v_events": f"""
            CREATE OR REPLACE VIEW v_events AS
            SELECT * FROM v_events_raw
        """,
        "v_markets": f"""
            CREATE OR REPLACE VIEW v_markets AS
            SELECT * FROM v_markets_raw
        """,
        "v_trades": f"""
            CREATE OR REPLACE VIEW v_trades AS
            SELECT * FROM v_trades_raw
        """,
        "v_order_filled": f"""
            CREATE OR REPLACE VIEW v_order_filled AS
            SELECT * FROM v_order_filled_raw
        """,
    }


class DuckDBLayer:
    """DuckDB layer for querying Parquet files with pre-defined views."""

    def __init__(
        self,
        db_path: Path,
        parquet_dir: Path,
        partitioned: bool = False,
        layered: bool = False,
        raw_dir: Path | None = None,
        analytics_dir: Path | None = None,
    ) -> None:
        """
        Initialize DuckDB layer.

        Args:
            db_path: Path to DuckDB database file
            parquet_dir: Directory for Parquet files (legacy single-layer mode)
            partitioned: Whether trades data is partitioned
            layered: Enable raw/analytics layer separation
            raw_dir: Directory for raw Parquet files (layered mode)
            analytics_dir: Directory for analytics Parquet files (layered mode)
        """
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        self.partitioned = partitioned
        self.layered = layered
        self.raw_dir = raw_dir
        self.analytics_dir = analytics_dir
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

        if self.layered and self.raw_dir and self.analytics_dir:
            # Use layered view definitions (raw + analytics)
            raw_dir_str = str(self.raw_dir.absolute())
            analytics_dir_str = str(self.analytics_dir.absolute())
            view_definitions = get_layered_view_definitions(
                raw_dir_str, analytics_dir_str, self.partitioned
            )
            logger.info(
                "creating_layered_views",
                raw_dir=raw_dir_str,
                analytics_dir=analytics_dir_str,
                partitioned=self.partitioned,
            )
        else:
            # Use legacy single-layer view definitions
            parquet_dir_str = str(self.parquet_dir.absolute())
            view_definitions = get_view_definitions(parquet_dir_str, self.partitioned)

        for view_name, view_sql in view_definitions.items():
            try:
                self.conn.execute(view_sql)
                created_views.append(view_name)
                logger.info("view_created", view=view_name, partitioned=self.partitioned)
            except Exception as e:
                error_str = str(e)
                # Events view is optional - continue if parquet file doesn't exist
                # or if v_events_raw doesn't exist (for alias views)
                is_events_view = "events" in view_name.lower()
                is_missing_file = "No files found" in error_str
                is_missing_base_view = "v_events_raw does not exist" in error_str

                if is_events_view and (is_missing_file or is_missing_base_view):
                    logger.info(
                        "view_skipped_no_parquet",
                        view=view_name,
                        reason="events.parquet not found",
                    )
                else:
                    raise

        return created_views

    def verify_views(self) -> dict[str, int]:
        """Verify all views are queryable and return row counts."""
        counts: dict[str, int] = {}

        if self.layered:
            # Layered mode has more views
            view_names = [
                "v_events_raw",
                "v_markets_raw",
                "v_trades_raw",
                "v_order_filled_raw",
                "v_trades_analytics",
                "v_trades_with_markets",
                "v_events",
                "v_markets",
                "v_trades",
                "v_order_filled",
            ]
        else:
            view_names = ["v_events", "v_markets", "v_trades", "v_order_filled", "v_trades_with_markets"]

        for view_name in view_names:
            try:
                result = self.conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()
                counts[view_name] = result[0] if result else 0
                logger.info("view_verified", view=view_name, rows=counts[view_name])
            except Exception as e:
                # View may not exist if corresponding parquet file doesn't exist
                logger.warning("view_not_available", view=view_name, error=str(e))
                counts[view_name] = 0
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

    def explain_query(
        self,
        sql: str,
        params: list[Any] | None = None,
        analyze: bool = False,
    ) -> str:
        """
        Get the query execution plan.

        Args:
            sql: The SQL query to explain
            params: Query parameters
            analyze: If True, use EXPLAIN ANALYZE (actually run query)

        Returns:
            The query plan as a string
        """
        explain_cmd = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
        explain_sql = f"{explain_cmd} {sql}"

        if params:
            result = self.conn.execute(explain_sql, params)
        else:
            result = self.conn.execute(explain_sql)

        rows = result.fetchall()
        return "\n".join(str(row[0]) for row in rows)
