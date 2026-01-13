"""ClickHouse serving layer for analytics rollups and recent drilldowns.

Implements the PRD requirements for ClickHouse serving layer:
- Rollup tables: wallet_agg_1m, wallet_agg_1h, wallet_agg_1d
- wallet_positions_current table for current positions
- Optional raw events retention for last 30 days
- S3 Parquet remains source of truth for historical data
"""

from datetime import datetime, timezone, timedelta
from typing import Any

import structlog

from polymkt.config import settings

logger = structlog.get_logger()


# ClickHouse DDL for rollup tables
WALLET_AGG_1M_DDL = """
CREATE TABLE IF NOT EXISTS wallet_agg_1m (
    wallet_address LowCardinality(String),
    window_start DateTime64(3, 'UTC'),
    window_end DateTime64(3, 'UTC'),
    trade_count UInt32 DEFAULT 0,
    buy_count UInt32 DEFAULT 0,
    sell_count UInt32 DEFAULT 0,
    volume_usd Float64 DEFAULT 0.0,
    positions_opened UInt32 DEFAULT 0,
    positions_closed UInt32 DEFAULT 0,
    realized_pnl Float64 DEFAULT 0.0,
    win_count UInt32 DEFAULT 0,
    loss_count UInt32 DEFAULT 0,
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYYYYMM(window_start)
ORDER BY (wallet_address, window_start)
TTL window_start + INTERVAL 90 DAY
"""

WALLET_AGG_1H_DDL = """
CREATE TABLE IF NOT EXISTS wallet_agg_1h (
    wallet_address LowCardinality(String),
    window_start DateTime64(3, 'UTC'),
    window_end DateTime64(3, 'UTC'),
    trade_count UInt32 DEFAULT 0,
    buy_count UInt32 DEFAULT 0,
    sell_count UInt32 DEFAULT 0,
    volume_usd Float64 DEFAULT 0.0,
    positions_opened UInt32 DEFAULT 0,
    positions_closed UInt32 DEFAULT 0,
    realized_pnl Float64 DEFAULT 0.0,
    win_count UInt32 DEFAULT 0,
    loss_count UInt32 DEFAULT 0,
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYYYYMM(window_start)
ORDER BY (wallet_address, window_start)
TTL window_start + INTERVAL 1 YEAR
"""

WALLET_AGG_1D_DDL = """
CREATE TABLE IF NOT EXISTS wallet_agg_1d (
    wallet_address LowCardinality(String),
    window_start DateTime64(3, 'UTC'),
    window_end DateTime64(3, 'UTC'),
    trade_count UInt32 DEFAULT 0,
    buy_count UInt32 DEFAULT 0,
    sell_count UInt32 DEFAULT 0,
    volume_usd Float64 DEFAULT 0.0,
    positions_opened UInt32 DEFAULT 0,
    positions_closed UInt32 DEFAULT 0,
    realized_pnl Float64 DEFAULT 0.0,
    win_count UInt32 DEFAULT 0,
    loss_count UInt32 DEFAULT 0,
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYear(window_start)
ORDER BY (wallet_address, window_start)
"""

WALLET_POSITIONS_CURRENT_DDL = """
CREATE TABLE IF NOT EXISTS wallet_positions_current (
    position_id String,
    wallet_address LowCardinality(String),
    market_id String,
    outcome LowCardinality(String),
    current_size Float64,
    total_cost_basis Float64,
    average_cost Nullable(Float64),
    realized_pnl Float64 DEFAULT 0.0,
    last_trade_price Nullable(Float64),
    last_price_timestamp Nullable(DateTime64(3, 'UTC')),
    mtm_pnl Nullable(Float64),
    mtm_window_start Nullable(DateTime64(3, 'UTC')),
    first_trade_at DateTime64(3, 'UTC'),
    updated_at DateTime64(3, 'UTC'),
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(inserted_at)
ORDER BY (wallet_address, market_id, outcome)
"""

RAW_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS raw_trades (
    timestamp DateTime64(3, 'UTC'),
    market_id String,
    maker LowCardinality(String),
    taker LowCardinality(String),
    nonusdc_side Nullable(String),
    maker_direction Nullable(String),
    taker_direction Nullable(String),
    price Float64,
    usd_amount Float64,
    token_amount Float64,
    transaction_hash String,
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
) ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (market_id, timestamp, transaction_hash)
TTL timestamp + INTERVAL {retention_days} DAY
"""

# All DDL statements for initialization
ALL_TABLES_DDL = [
    ("wallet_agg_1m", WALLET_AGG_1M_DDL),
    ("wallet_agg_1h", WALLET_AGG_1H_DDL),
    ("wallet_agg_1d", WALLET_AGG_1D_DDL),
    ("wallet_positions_current", WALLET_POSITIONS_CURRENT_DDL),
]


def _try_import_clickhouse() -> Any:
    """Try to import clickhouse-connect, return None if not available."""
    try:
        import clickhouse_connect
        return clickhouse_connect
    except ImportError:
        return None


class ClickHouseLayer:
    """
    ClickHouse serving layer for analytics queries.

    Provides:
    - Connection management
    - Table initialization
    - Data sync from SQLite metadata store
    - Query interface for rollup tables
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self.host = host or settings.clickhouse_host
        self.port = port or settings.clickhouse_http_port
        self.database = database or settings.clickhouse_database
        self.user = user or settings.clickhouse_user
        self.password = password or settings.clickhouse_password
        self._client: Any = None
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to ClickHouse.

        Returns:
            True if connection successful, False otherwise
        """
        clickhouse_connect = _try_import_clickhouse()
        if clickhouse_connect is None:
            logger.warning(
                "clickhouse_connect_not_available",
                message="Install clickhouse-connect to enable ClickHouse",
            )
            return False

        try:
            self._client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                database=self.database,
                username=self.user,
                password=self.password,
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(
                "clickhouse_connected",
                host=self.host,
                port=self.port,
                database=self.database,
            )
            return True
        except Exception as e:
            logger.error("clickhouse_connection_failed", error=str(e))
            self._connected = False
            return False

    def close(self) -> None:
        """Close ClickHouse connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to ClickHouse."""
        return self._connected and self._client is not None

    def initialize_tables(self, include_raw_trades: bool = True) -> list[str]:
        """
        Create all required tables in ClickHouse.

        Args:
            include_raw_trades: Whether to create raw_trades table

        Returns:
            List of tables created
        """
        if not self.is_connected():
            if not self.connect():
                return []

        created_tables = []

        # Create database if not exists
        try:
            self._client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        except Exception as e:
            logger.warning("database_create_failed", error=str(e))

        # Create rollup tables
        for table_name, ddl in ALL_TABLES_DDL:
            try:
                self._client.command(ddl)
                created_tables.append(table_name)
                logger.info("clickhouse_table_created", table=table_name)
            except Exception as e:
                logger.error(
                    "clickhouse_table_create_failed",
                    table=table_name,
                    error=str(e),
                )

        # Create raw trades table with TTL
        if include_raw_trades:
            try:
                retention_days = settings.clickhouse_raw_retention_days
                raw_ddl = RAW_TRADES_DDL.format(retention_days=retention_days)
                self._client.command(raw_ddl)
                created_tables.append("raw_trades")
                logger.info(
                    "clickhouse_table_created",
                    table="raw_trades",
                    retention_days=retention_days,
                )
            except Exception as e:
                logger.error(
                    "clickhouse_table_create_failed",
                    table="raw_trades",
                    error=str(e),
                )

        return created_tables

    def sync_rollups_from_sqlite(
        self,
        metadata_db_path: str | None = None,
        interval: str = "1d",
        wallet_address: str | None = None,
    ) -> int:
        """
        Sync rollup data from SQLite metadata store to ClickHouse.

        Args:
            metadata_db_path: Path to SQLite metadata database
            interval: Rollup interval to sync (1m, 1h, 1d)
            wallet_address: Optional wallet to sync (None = all wallets)

        Returns:
            Number of rows synced
        """
        if not self.is_connected():
            if not self.connect():
                return 0

        metadata_db_path = metadata_db_path or str(settings.metadata_db_path)

        # Determine target table
        table_map = {
            "1m": "wallet_agg_1m",
            "1h": "wallet_agg_1h",
            "1d": "wallet_agg_1d",
        }
        table_name = table_map.get(interval)
        if not table_name:
            logger.error("invalid_interval", interval=interval)
            return 0

        # Get rollups from SQLite
        import sqlite3

        conn = sqlite3.connect(metadata_db_path)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT * FROM wallet_metrics_rollups
            WHERE interval = ?
        """
        params: list[Any] = [interval]

        if wallet_address:
            query += " AND wallet_address = ?"
            params.append(wallet_address.lower())

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return 0

        # Convert to ClickHouse format
        data = []
        for row in rows:
            data.append({
                "wallet_address": row["wallet_address"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "trade_count": row["trade_count"],
                "buy_count": row["buy_count"],
                "sell_count": row["sell_count"],
                "volume_usd": row["volume_usd"],
                "positions_opened": row["positions_opened"],
                "positions_closed": row["positions_closed"],
                "realized_pnl": row["realized_pnl"],
                "win_count": row["win_count"],
                "loss_count": row["loss_count"],
            })

        # Insert into ClickHouse
        try:
            self._client.insert(
                table_name,
                data,
                column_names=[
                    "wallet_address",
                    "window_start",
                    "window_end",
                    "trade_count",
                    "buy_count",
                    "sell_count",
                    "volume_usd",
                    "positions_opened",
                    "positions_closed",
                    "realized_pnl",
                    "win_count",
                    "loss_count",
                ],
            )
            logger.info(
                "clickhouse_rollups_synced",
                table=table_name,
                rows=len(data),
                interval=interval,
            )
            return len(data)
        except Exception as e:
            logger.error("clickhouse_sync_failed", error=str(e))
            return 0

    def sync_positions_from_sqlite(
        self,
        metadata_db_path: str | None = None,
        wallet_address: str | None = None,
    ) -> int:
        """
        Sync current positions from SQLite to ClickHouse.

        Args:
            metadata_db_path: Path to SQLite metadata database
            wallet_address: Optional wallet to sync (None = all wallets)

        Returns:
            Number of positions synced
        """
        if not self.is_connected():
            if not self.connect():
                return 0

        metadata_db_path = metadata_db_path or str(settings.metadata_db_path)

        import sqlite3

        conn = sqlite3.connect(metadata_db_path)
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM positions"
        params: list[Any] = []

        if wallet_address:
            query += " WHERE wallet_address = ?"
            params.append(wallet_address.lower())

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return 0

        # Convert to ClickHouse format
        data = []
        for row in rows:
            data.append({
                "position_id": row["id"],
                "wallet_address": row["wallet_address"],
                "market_id": row["market_id"],
                "outcome": row["outcome"],
                "current_size": row["current_size"],
                "total_cost_basis": row["total_cost_basis"],
                "average_cost": row["average_cost"],
                "realized_pnl": row["realized_pnl"] or 0.0,
                "last_trade_price": row["last_trade_price"],
                "last_price_timestamp": row["last_price_timestamp"],
                "mtm_pnl": row["mtm_pnl"],
                "mtm_window_start": row["mtm_window_start"],
                "first_trade_at": row["first_trade_at"],
                "updated_at": row["updated_at"],
            })

        # Insert into ClickHouse
        try:
            self._client.insert(
                "wallet_positions_current",
                data,
                column_names=[
                    "position_id",
                    "wallet_address",
                    "market_id",
                    "outcome",
                    "current_size",
                    "total_cost_basis",
                    "average_cost",
                    "realized_pnl",
                    "last_trade_price",
                    "last_price_timestamp",
                    "mtm_pnl",
                    "mtm_window_start",
                    "first_trade_at",
                    "updated_at",
                ],
            )
            logger.info(
                "clickhouse_positions_synced",
                rows=len(data),
            )
            return len(data)
        except Exception as e:
            logger.error("clickhouse_positions_sync_failed", error=str(e))
            return 0

    def query_wallet_rollups(
        self,
        wallet_address: str,
        interval: str = "1d",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query wallet rollups from ClickHouse.

        Args:
            wallet_address: Wallet address to query
            interval: Rollup interval (1m, 1h, 1d)
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum rows to return

        Returns:
            List of rollup records
        """
        if not self.is_connected():
            if not self.connect():
                return []

        table_map = {
            "1m": "wallet_agg_1m",
            "1h": "wallet_agg_1h",
            "1d": "wallet_agg_1d",
        }
        table_name = table_map.get(interval)
        if not table_name:
            return []

        query = f"""
            SELECT *
            FROM {table_name} FINAL
            WHERE wallet_address = {{wallet:String}}
        """
        params: dict[str, Any] = {"wallet": wallet_address.lower()}

        if start_time:
            query += " AND window_start >= {start:DateTime64(3)}"
            params["start"] = start_time.isoformat()
        if end_time:
            query += " AND window_start < {end:DateTime64(3)}"
            params["end"] = end_time.isoformat()

        query += f" ORDER BY window_start DESC LIMIT {limit}"

        try:
            result = self._client.query(query, parameters=params)
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        except Exception as e:
            logger.error("clickhouse_query_failed", error=str(e))
            return []

    def query_wallet_positions(
        self,
        wallet_address: str | None = None,
        market_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query current positions from ClickHouse.

        Args:
            wallet_address: Optional wallet filter
            market_id: Optional market filter

        Returns:
            List of position records
        """
        if not self.is_connected():
            if not self.connect():
                return []

        query = "SELECT * FROM wallet_positions_current FINAL WHERE 1=1"
        params: dict[str, Any] = {}

        if wallet_address:
            query += " AND wallet_address = {wallet:String}"
            params["wallet"] = wallet_address.lower()
        if market_id:
            query += " AND market_id = {market:String}"
            params["market"] = market_id

        query += " ORDER BY updated_at DESC"

        try:
            result = self._client.query(query, parameters=params)
            return [dict(zip(result.column_names, row)) for row in result.result_rows]
        except Exception as e:
            logger.error("clickhouse_query_failed", error=str(e))
            return []

    def get_table_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all ClickHouse tables.

        Returns:
            Dict of table name to stats (row_count, size, partitions)
        """
        if not self.is_connected():
            if not self.connect():
                return {}

        tables = [
            "wallet_agg_1m",
            "wallet_agg_1h",
            "wallet_agg_1d",
            "wallet_positions_current",
            "raw_trades",
        ]
        stats: dict[str, dict[str, Any]] = {}

        for table in tables:
            try:
                result = self._client.query(f"""
                    SELECT
                        count() as row_count,
                        sum(bytes) as bytes,
                        count(DISTINCT partition) as partitions
                    FROM system.parts
                    WHERE database = '{self.database}'
                    AND table = '{table}'
                    AND active = 1
                """)
                if result.result_rows:
                    row = result.result_rows[0]
                    stats[table] = {
                        "row_count": row[0],
                        "bytes": row[1],
                        "partitions": row[2],
                        "exists": True,
                    }
                else:
                    stats[table] = {"exists": False}
            except Exception:
                stats[table] = {"exists": False}

        return stats


def get_clickhouse_layer() -> ClickHouseLayer:
    """Get a ClickHouse layer instance with default settings."""
    return ClickHouseLayer()


def is_clickhouse_available() -> bool:
    """Check if ClickHouse is available and configured."""
    if not settings.clickhouse_enabled:
        return False

    layer = ClickHouseLayer()
    connected = layer.connect()
    layer.close()
    return connected
