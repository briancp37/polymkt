"""LLM-driven analytics with conservative guardrails.

Implements PRD requirements for safe LLM query interface:
- Restrict queries to ClickHouse rollup tables only
- Restrict queries to wallets present in watchlist
- Apply default 7-day time window
- Enforce hard limits: max 10k rows, 30s execution time
- Reject queries that access raw event tables
- Log all queries to ops metadata store
"""

import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from polymkt.config import settings

logger = structlog.get_logger()


# Allowed tables for LLM queries (rollup tables only)
ALLOWED_TABLES = frozenset({
    "wallet_agg_1m",
    "wallet_agg_1h",
    "wallet_agg_1d",
    "wallet_positions_current",
})

# Blocked tables (raw event tables)
BLOCKED_TABLES = frozenset({
    "raw_trades",
    "raw_events",
    "trades",
    "events",
    "order_filled",
})

# Query hard limits
MAX_ROWS_RETURNED = 10_000
MAX_EXECUTION_TIME_SECONDS = 30
DEFAULT_TIME_WINDOW_DAYS = 7


class QueryValidationError(Exception):
    """Raised when a query fails validation."""

    pass


class QueryGuardrails:
    """
    Validates and enforces guardrails on analytics queries.

    Ensures queries:
    - Only access allowed rollup tables
    - Only query wallets in the watchlist
    - Have time bounds within allowed limits
    - Return no more than max rows
    """

    def __init__(
        self,
        max_rows: int = MAX_ROWS_RETURNED,
        max_execution_time: int = MAX_EXECUTION_TIME_SECONDS,
        default_time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
    ) -> None:
        self.max_rows = max_rows
        self.max_execution_time = max_execution_time
        self.default_time_window_days = default_time_window_days

    def validate_table_access(self, sql: str) -> None:
        """
        Validate that the SQL query only accesses allowed tables.

        Raises:
            QueryValidationError: If query accesses blocked tables
        """
        sql_lower = sql.lower()

        # Check for blocked tables
        for blocked in BLOCKED_TABLES:
            # Match table name as word boundary
            pattern = rf"\b{blocked}\b"
            if re.search(pattern, sql_lower):
                raise QueryValidationError(
                    f"Access to table '{blocked}' is not allowed. "
                    f"Only rollup tables are permitted: {', '.join(sorted(ALLOWED_TABLES))}"
                )

        # Check that at least one allowed table is referenced
        has_allowed = False
        for allowed in ALLOWED_TABLES:
            pattern = rf"\b{allowed}\b"
            if re.search(pattern, sql_lower):
                has_allowed = True
                break

        if not has_allowed:
            raise QueryValidationError(
                f"Query must reference at least one allowed table: "
                f"{', '.join(sorted(ALLOWED_TABLES))}"
            )

    def validate_wallet_access(
        self,
        wallet_addresses: list[str],
        watched_wallets: set[str],
    ) -> None:
        """
        Validate that all queried wallets are in the watchlist.

        Args:
            wallet_addresses: Wallet addresses being queried
            watched_wallets: Set of watched wallet addresses (lowercase)

        Raises:
            QueryValidationError: If any wallet is not in watchlist
        """
        if not watched_wallets:
            raise QueryValidationError(
                "No wallets in watchlist. Add wallets to a watchlist before querying."
            )

        for addr in wallet_addresses:
            normalized = addr.lower()
            if normalized not in watched_wallets:
                raise QueryValidationError(
                    f"Wallet '{addr}' is not in any watchlist. "
                    "Only watched wallets can be queried."
                )

    def apply_time_window(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> tuple[datetime, datetime]:
        """
        Apply default time window if not specified.

        Args:
            start_time: Query start time (optional)
            end_time: Query end time (optional)

        Returns:
            Tuple of (start_time, end_time) with defaults applied
        """
        now = datetime.now(timezone.utc)

        if end_time is None:
            end_time = now

        if start_time is None:
            start_time = end_time - timedelta(days=self.default_time_window_days)

        return start_time, end_time

    def apply_row_limit(self, limit: int | None) -> int:
        """
        Apply row limit, enforcing maximum.

        Args:
            limit: Requested row limit (optional)

        Returns:
            Effective row limit (capped at max_rows)
        """
        if limit is None or limit > self.max_rows:
            return self.max_rows
        return max(1, limit)


class LLMAnalyticsEngine:
    """
    LLM-driven analytics engine with query guardrails.

    Provides a safe interface for LLM and UI analytics queries
    with enforced restrictions and logging.
    """

    def __init__(
        self,
        metadata_db_path: str | None = None,
        guardrails: QueryGuardrails | None = None,
    ) -> None:
        self.metadata_db_path = metadata_db_path or str(settings.metadata_db_path)
        self.guardrails = guardrails or QueryGuardrails()
        self._watched_wallets_cache: set[str] | None = None
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_seconds = 60  # Refresh cache every 60 seconds

    def _get_watched_wallets(self) -> set[str]:
        """Get all watched wallets from all watchlists, with caching."""
        now = datetime.now(timezone.utc)

        # Return cached value if still valid
        if (
            self._watched_wallets_cache is not None
            and self._cache_timestamp is not None
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
        ):
            return self._watched_wallets_cache

        import sqlite3

        conn = sqlite3.connect(self.metadata_db_path)
        try:
            cursor = conn.execute(
                "SELECT DISTINCT wallet_address FROM watchlist_items"
            )
            self._watched_wallets_cache = {row[0].lower() for row in cursor.fetchall()}
            self._cache_timestamp = now
            return self._watched_wallets_cache
        finally:
            conn.close()

    def _log_query(
        self,
        query_id: str,
        sql: str,
        parameters: dict[str, Any],
        wallet_addresses: list[str],
        execution_time_ms: float,
        rows_read: int,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Log query execution to ops metadata store."""
        import sqlite3

        conn = sqlite3.connect(self.metadata_db_path)
        try:
            # Ensure query log table exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_query_log (
                    id TEXT PRIMARY KEY,
                    sql TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    wallet_addresses TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    rows_read INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    executed_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_query_log_executed_at
                ON llm_query_log(executed_at)
            """)

            import json

            conn.execute(
                """
                INSERT INTO llm_query_log
                (id, sql, parameters, wallet_addresses, execution_time_ms,
                 rows_read, status, error_message, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    sql,
                    json.dumps(parameters),
                    json.dumps(wallet_addresses),
                    execution_time_ms,
                    rows_read,
                    status,
                    error_message,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def query_wallet_rollups(
        self,
        wallet_address: str,
        interval: str = "1d",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Query wallet rollups with guardrails.

        Args:
            wallet_address: Wallet address to query
            interval: Rollup interval (1m, 1h, 1d)
            start_time: Query start time (default: 7 days ago)
            end_time: Query end time (default: now)
            limit: Max rows to return (default/max: 10k)

        Returns:
            Dict with query results and metadata

        Raises:
            QueryValidationError: If guardrails are violated
        """
        query_id = str(uuid.uuid4())
        start_execution = time.time()

        try:
            # Validate wallet is in watchlist
            watched_wallets = self._get_watched_wallets()
            self.guardrails.validate_wallet_access([wallet_address], watched_wallets)

            # Apply time window defaults
            start_time, end_time = self.guardrails.apply_time_window(start_time, end_time)

            # Apply row limit
            effective_limit = self.guardrails.apply_row_limit(limit)

            # Validate interval
            table_map = {
                "1m": "wallet_agg_1m",
                "1h": "wallet_agg_1h",
                "1d": "wallet_agg_1d",
            }
            if interval not in table_map:
                raise QueryValidationError(
                    f"Invalid interval '{interval}'. Must be one of: 1m, 1h, 1d"
                )

            # Execute query via ClickHouse layer
            from polymkt.storage.clickhouse import ClickHouseLayer

            layer = ClickHouseLayer()
            if not layer.connect():
                raise QueryValidationError(
                    "Failed to connect to ClickHouse. Ensure it is enabled and running."
                )

            try:
                results = layer.query_wallet_rollups(
                    wallet_address=wallet_address,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=effective_limit,
                )
            finally:
                layer.close()

            execution_time_ms = (time.time() - start_execution) * 1000

            # Log successful query
            self._log_query(
                query_id=query_id,
                sql=f"SELECT * FROM {table_map[interval]} WHERE wallet_address = ?",
                parameters={
                    "wallet_address": wallet_address,
                    "interval": interval,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "limit": effective_limit,
                },
                wallet_addresses=[wallet_address],
                execution_time_ms=execution_time_ms,
                rows_read=len(results),
                status="success",
            )

            return {
                "query_id": query_id,
                "wallet_address": wallet_address,
                "interval": interval,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "limit_applied": effective_limit,
                "rows_returned": len(results),
                "execution_time_ms": round(execution_time_ms, 2),
                "rollups": results,
            }

        except QueryValidationError:
            execution_time_ms = (time.time() - start_execution) * 1000
            self._log_query(
                query_id=query_id,
                sql="",
                parameters={"wallet_address": wallet_address, "interval": interval},
                wallet_addresses=[wallet_address],
                execution_time_ms=execution_time_ms,
                rows_read=0,
                status="validation_error",
                error_message=str(QueryValidationError),
            )
            raise

        except Exception as e:
            execution_time_ms = (time.time() - start_execution) * 1000
            self._log_query(
                query_id=query_id,
                sql="",
                parameters={"wallet_address": wallet_address, "interval": interval},
                wallet_addresses=[wallet_address],
                execution_time_ms=execution_time_ms,
                rows_read=0,
                status="error",
                error_message=str(e),
            )
            raise

    def query_wallet_positions(
        self,
        wallet_address: str,
        market_id: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Query current positions for a wallet with guardrails.

        Args:
            wallet_address: Wallet address to query
            market_id: Optional market filter
            limit: Max rows to return (default/max: 10k)

        Returns:
            Dict with position results and metadata

        Raises:
            QueryValidationError: If guardrails are violated
        """
        query_id = str(uuid.uuid4())
        start_execution = time.time()

        try:
            # Validate wallet is in watchlist
            watched_wallets = self._get_watched_wallets()
            self.guardrails.validate_wallet_access([wallet_address], watched_wallets)

            # Apply row limit
            effective_limit = self.guardrails.apply_row_limit(limit)

            # Execute query via ClickHouse layer
            from polymkt.storage.clickhouse import ClickHouseLayer

            layer = ClickHouseLayer()
            if not layer.connect():
                raise QueryValidationError(
                    "Failed to connect to ClickHouse. Ensure it is enabled and running."
                )

            try:
                results = layer.query_wallet_positions(
                    wallet_address=wallet_address,
                    market_id=market_id,
                )
                # Apply limit manually since query_wallet_positions doesn't have limit
                if len(results) > effective_limit:
                    results = results[:effective_limit]
            finally:
                layer.close()

            execution_time_ms = (time.time() - start_execution) * 1000

            # Log successful query
            self._log_query(
                query_id=query_id,
                sql="SELECT * FROM wallet_positions_current WHERE wallet_address = ?",
                parameters={
                    "wallet_address": wallet_address,
                    "market_id": market_id,
                    "limit": effective_limit,
                },
                wallet_addresses=[wallet_address],
                execution_time_ms=execution_time_ms,
                rows_read=len(results),
                status="success",
            )

            return {
                "query_id": query_id,
                "wallet_address": wallet_address,
                "market_id": market_id,
                "limit_applied": effective_limit,
                "rows_returned": len(results),
                "execution_time_ms": round(execution_time_ms, 2),
                "positions": results,
            }

        except QueryValidationError:
            execution_time_ms = (time.time() - start_execution) * 1000
            self._log_query(
                query_id=query_id,
                sql="",
                parameters={"wallet_address": wallet_address, "market_id": market_id},
                wallet_addresses=[wallet_address],
                execution_time_ms=execution_time_ms,
                rows_read=0,
                status="validation_error",
                error_message=str(QueryValidationError),
            )
            raise

        except Exception as e:
            execution_time_ms = (time.time() - start_execution) * 1000
            self._log_query(
                query_id=query_id,
                sql="",
                parameters={"wallet_address": wallet_address, "market_id": market_id},
                wallet_addresses=[wallet_address],
                execution_time_ms=execution_time_ms,
                rows_read=0,
                status="error",
                error_message=str(e),
            )
            raise

    def execute_raw_sql(
        self,
        sql: str,
        wallet_addresses: list[str],
        parameters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a raw SQL query with guardrails.

        This is the most flexible query method but has strict validation:
        - SQL must only reference allowed tables
        - All wallet addresses must be in watchlist
        - Row limit is enforced

        Args:
            sql: SQL query to execute
            wallet_addresses: Wallet addresses being queried (for validation)
            parameters: Query parameters
            limit: Max rows to return (default/max: 10k)

        Returns:
            Dict with query results and metadata

        Raises:
            QueryValidationError: If guardrails are violated
        """
        query_id = str(uuid.uuid4())
        start_execution = time.time()
        parameters = parameters or {}

        try:
            # Validate table access
            self.guardrails.validate_table_access(sql)

            # Validate wallet access
            watched_wallets = self._get_watched_wallets()
            self.guardrails.validate_wallet_access(wallet_addresses, watched_wallets)

            # Apply row limit
            effective_limit = self.guardrails.apply_row_limit(limit)

            # Inject LIMIT if not present
            sql_lower = sql.lower().strip()
            if "limit" not in sql_lower:
                sql = f"{sql.rstrip().rstrip(';')} LIMIT {effective_limit}"

            # Execute query via ClickHouse layer
            from polymkt.storage.clickhouse import ClickHouseLayer

            layer = ClickHouseLayer()
            if not layer.connect():
                raise QueryValidationError(
                    "Failed to connect to ClickHouse. Ensure it is enabled and running."
                )

            try:
                # Execute with timeout
                result = layer._client.query(
                    sql,
                    parameters=parameters,
                    settings={
                        "max_execution_time": self.guardrails.max_execution_time,
                        "max_result_rows": effective_limit,
                    },
                )
                results = [
                    dict(zip(result.column_names, row))
                    for row in result.result_rows
                ]
            finally:
                layer.close()

            execution_time_ms = (time.time() - start_execution) * 1000

            # Log successful query
            self._log_query(
                query_id=query_id,
                sql=sql,
                parameters=parameters,
                wallet_addresses=wallet_addresses,
                execution_time_ms=execution_time_ms,
                rows_read=len(results),
                status="success",
            )

            return {
                "query_id": query_id,
                "sql": sql,
                "limit_applied": effective_limit,
                "rows_returned": len(results),
                "execution_time_ms": round(execution_time_ms, 2),
                "results": results,
            }

        except QueryValidationError:
            execution_time_ms = (time.time() - start_execution) * 1000
            self._log_query(
                query_id=query_id,
                sql=sql,
                parameters=parameters,
                wallet_addresses=wallet_addresses,
                execution_time_ms=execution_time_ms,
                rows_read=0,
                status="validation_error",
                error_message=str(QueryValidationError),
            )
            raise

        except Exception as e:
            execution_time_ms = (time.time() - start_execution) * 1000
            self._log_query(
                query_id=query_id,
                sql=sql,
                parameters=parameters,
                wallet_addresses=wallet_addresses,
                execution_time_ms=execution_time_ms,
                rows_read=0,
                status="error",
                error_message=str(e),
            )
            raise

    def get_query_log(
        self,
        limit: int = 100,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get recent query log entries.

        Args:
            limit: Maximum entries to return
            status: Filter by status (success, error, validation_error)

        Returns:
            List of query log entries
        """
        import json
        import sqlite3

        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM llm_query_log"
            params: list[Any] = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            query += " ORDER BY executed_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "sql": row["sql"],
                    "parameters": json.loads(row["parameters"]),
                    "wallet_addresses": json.loads(row["wallet_addresses"]),
                    "execution_time_ms": row["execution_time_ms"],
                    "rows_read": row["rows_read"],
                    "status": row["status"],
                    "error_message": row["error_message"],
                    "executed_at": row["executed_at"],
                })
            return results
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return []
        finally:
            conn.close()

    def get_guardrails_config(self) -> dict[str, Any]:
        """Get the current guardrails configuration."""
        return {
            "max_rows_returned": self.guardrails.max_rows,
            "max_execution_time_seconds": self.guardrails.max_execution_time,
            "default_time_window_days": self.guardrails.default_time_window_days,
            "allowed_tables": sorted(ALLOWED_TABLES),
            "blocked_tables": sorted(BLOCKED_TABLES),
        }


def get_llm_analytics_engine() -> LLMAnalyticsEngine:
    """Get an LLM analytics engine instance with default settings."""
    return LLMAnalyticsEngine()
