"""Metadata storage for run history, watermarks, watchlists, and alerts using SQLite."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from polymkt.models.schemas import (
    RunRecord,
    WatchlistSchema,
    WatchlistItemSchema,
    AlertSubscriptionSchema,
    AlertSchema,
    AnalyticsSessionSchema,
    ModeStateSchema,
    IngestMode,
    AnalyticsMode,
)

logger = structlog.get_logger()


class MetadataStore:
    """SQLite-backed metadata storage for pipeline run history and watermarks."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    rows_read TEXT NOT NULL,
                    rows_written TEXT NOT NULL,
                    watermark_before TEXT,
                    watermark_after TEXT,
                    schema_version TEXT NOT NULL,
                    error_message TEXT,
                    duration_seconds REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watermarks (
                    entity TEXT PRIMARY KEY,
                    watermark_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Watchlist tables for Sharp Money tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    watchlist_id TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    notes TEXT,
                    PRIMARY KEY (watchlist_id, wallet_address),
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_watchlist_items_address
                ON watchlist_items(wallet_address)
            """)

            # Alert subscription tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_subscriptions (
                    id TEXT PRIMARY KEY,
                    watchlist_id TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    rule_config TEXT NOT NULL,
                    cooldown_seconds INTEGER DEFAULT 300,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    subscription_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    trade_data TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_at TEXT,
                    triggered_at TEXT NOT NULL,
                    FOREIGN KEY (subscription_id) REFERENCES alert_subscriptions(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_subscription_triggered
                ON alerts(subscription_id, triggered_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_event_id
                ON alerts(event_id)
            """)

            # Analytics session lifecycle tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    last_activity_at TEXT NOT NULL,
                    idle_timeout_minutes INTEGER DEFAULT 120,
                    status TEXT NOT NULL,
                    queries_run INTEGER DEFAULT 0,
                    rows_accessed INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status_activity
                ON analytics_sessions(status, last_activity_at DESC)
            """)

            # Runtime mode state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mode_state (
                    mode_name TEXT PRIMARY KEY,
                    mode_value TEXT NOT NULL,
                    previous_value TEXT,
                    transitioned_at TEXT NOT NULL,
                    transitioned_by TEXT DEFAULT 'system',
                    is_transitioning INTEGER DEFAULT 0,
                    transition_started_at TEXT,
                    updated_at TEXT NOT NULL
                )
            """)

            # Initialize default modes if they don't exist
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                INSERT OR IGNORE INTO mode_state
                (mode_name, mode_value, transitioned_at, transitioned_by, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("ingest_mode", IngestMode.OFF.value, now, "system", now))
            conn.execute("""
                INSERT OR IGNORE INTO mode_state
                (mode_name, mode_value, transitioned_at, transitioned_by, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, ("analytics_mode", AnalyticsMode.OFF.value, now, "system", now))

            conn.commit()
        finally:
            conn.close()

    def create_run(self, run: RunRecord) -> None:
        """Insert a new run record."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, run_type, start_time, end_time, status,
                    rows_read, rows_written, watermark_before, watermark_after,
                    schema_version, error_message, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.run_type,
                    run.start_time.isoformat(),
                    run.end_time.isoformat() if run.end_time else None,
                    run.status,
                    json.dumps(run.rows_read),
                    json.dumps(run.rows_written),
                    json.dumps(run.watermark_before) if run.watermark_before else None,
                    json.dumps(run.watermark_after) if run.watermark_after else None,
                    run.schema_version,
                    run.error_message,
                    run.duration_seconds,
                ),
            )
            conn.commit()
            logger.info("run_created", run_id=run.run_id, run_type=run.run_type)
        finally:
            conn.close()

    def update_run(self, run: RunRecord) -> None:
        """Update an existing run record."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE runs SET
                    end_time = ?,
                    status = ?,
                    rows_read = ?,
                    rows_written = ?,
                    watermark_before = ?,
                    watermark_after = ?,
                    error_message = ?,
                    duration_seconds = ?
                WHERE run_id = ?
                """,
                (
                    run.end_time.isoformat() if run.end_time else None,
                    run.status,
                    json.dumps(run.rows_read),
                    json.dumps(run.rows_written),
                    json.dumps(run.watermark_before) if run.watermark_before else None,
                    json.dumps(run.watermark_after) if run.watermark_after else None,
                    run.error_message,
                    run.duration_seconds,
                    run.run_id,
                ),
            )
            conn.commit()
            logger.info("run_updated", run_id=run.run_id, status=run.status)
        finally:
            conn.close()

    def get_run(self, run_id: str) -> RunRecord | None:
        """Get a run record by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_run(row)
        finally:
            conn.close()

    def list_runs(self, limit: int = 100) -> list[RunRecord]:
        """List recent runs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM runs ORDER BY start_time DESC LIMIT ?", (limit,)
            )
            return [self._row_to_run(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        """Convert a database row to a RunRecord."""
        return RunRecord(
            run_id=row["run_id"],
            run_type=row["run_type"],
            start_time=datetime.fromisoformat(row["start_time"]),
            end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
            status=row["status"],
            rows_read=json.loads(row["rows_read"]),
            rows_written=json.loads(row["rows_written"]),
            watermark_before=json.loads(row["watermark_before"]) if row["watermark_before"] else None,
            watermark_after=json.loads(row["watermark_after"]) if row["watermark_after"] else None,
            schema_version=row["schema_version"],
            error_message=row["error_message"],
            duration_seconds=row["duration_seconds"],
        )

    def get_watermark(self, entity: str) -> Any | None:
        """Get the current watermark for an entity."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT watermark_value FROM watermarks WHERE entity = ?", (entity,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def set_watermark(self, entity: str, value: Any) -> None:
        """Set the watermark for an entity."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO watermarks (entity, watermark_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(entity) DO UPDATE SET
                    watermark_value = excluded.watermark_value,
                    updated_at = excluded.updated_at
                """,
                (entity, json.dumps(value), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            logger.info("watermark_set", entity=entity)
        finally:
            conn.close()

    def get_all_watermarks(self) -> dict[str, Any]:
        """Get all current watermarks."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT entity, watermark_value FROM watermarks")
            return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        finally:
            conn.close()

    # =========================================================================
    # Watchlist methods
    # =========================================================================

    def create_watchlist(self, name: str, description: str | None = None) -> WatchlistSchema:
        """Create a new watchlist."""
        watchlist_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO watchlists (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (watchlist_id, name, description, now, now),
            )
            conn.commit()
            logger.info("watchlist_created", watchlist_id=watchlist_id, name=name)
            return WatchlistSchema(
                id=watchlist_id,
                name=name,
                description=description,
                wallet_addresses=[],
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now),
            )
        finally:
            conn.close()

    def get_watchlist(self, watchlist_id: str) -> WatchlistSchema | None:
        """Get a watchlist by ID including all wallet addresses."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM watchlists WHERE id = ?", (watchlist_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None

            # Get wallet addresses
            cursor = conn.execute(
                "SELECT wallet_address FROM watchlist_items WHERE watchlist_id = ?",
                (watchlist_id,),
            )
            addresses = [r[0] for r in cursor.fetchall()]

            return WatchlistSchema(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                wallet_addresses=addresses,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
        finally:
            conn.close()

    def list_watchlists(self) -> list[WatchlistSchema]:
        """List all watchlists."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM watchlists ORDER BY created_at DESC")
            watchlists = []
            for row in cursor.fetchall():
                # Get wallet addresses for each watchlist
                addr_cursor = conn.execute(
                    "SELECT wallet_address FROM watchlist_items WHERE watchlist_id = ?",
                    (row["id"],),
                )
                addresses = [r[0] for r in addr_cursor.fetchall()]
                watchlists.append(
                    WatchlistSchema(
                        id=row["id"],
                        name=row["name"],
                        description=row["description"],
                        wallet_addresses=addresses,
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
            return watchlists
        finally:
            conn.close()

    def list_watchlist_items(self, watchlist_id: str) -> list[WatchlistItemSchema]:
        """List all items (wallet addresses) in a watchlist."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT watchlist_id, wallet_address, added_at, notes
                FROM watchlist_items WHERE watchlist_id = ?
                ORDER BY added_at DESC
                """,
                (watchlist_id,),
            )
            items = []
            for row in cursor.fetchall():
                items.append(
                    WatchlistItemSchema(
                        watchlist_id=row["watchlist_id"],
                        wallet_address=row["wallet_address"],
                        added_at=datetime.fromisoformat(row["added_at"]),
                        notes=row["notes"],
                    )
                )
            return items
        finally:
            conn.close()

    def add_wallet_to_watchlist(
        self, watchlist_id: str, wallet_address: str, notes: str | None = None
    ) -> WatchlistItemSchema:
        """Add a wallet address to a watchlist."""
        # Normalize address to lowercase
        normalized_address = wallet_address.lower()
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO watchlist_items
                (watchlist_id, wallet_address, added_at, notes)
                VALUES (?, ?, ?, ?)
                """,
                (watchlist_id, normalized_address, now, notes),
            )
            # Update watchlist updated_at
            conn.execute(
                "UPDATE watchlists SET updated_at = ? WHERE id = ?",
                (now, watchlist_id),
            )
            conn.commit()
            logger.info(
                "wallet_added_to_watchlist",
                watchlist_id=watchlist_id,
                wallet_address=normalized_address,
            )
            return WatchlistItemSchema(
                watchlist_id=watchlist_id,
                wallet_address=normalized_address,
                added_at=datetime.fromisoformat(now),
                notes=notes,
            )
        finally:
            conn.close()

    def remove_wallet_from_watchlist(
        self, watchlist_id: str, wallet_address: str
    ) -> bool:
        """Remove a wallet address from a watchlist."""
        normalized_address = wallet_address.lower()
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM watchlist_items WHERE watchlist_id = ? AND wallet_address = ?",
                (watchlist_id, normalized_address),
            )
            # Update watchlist updated_at
            conn.execute(
                "UPDATE watchlists SET updated_at = ? WHERE id = ?",
                (now, watchlist_id),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(
                    "wallet_removed_from_watchlist",
                    watchlist_id=watchlist_id,
                    wallet_address=normalized_address,
                )
            return deleted
        finally:
            conn.close()

    def is_wallet_watched(self, wallet_address: str) -> bool:
        """Check if a wallet address is in any watchlist."""
        normalized_address = wallet_address.lower()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT 1 FROM watchlist_items WHERE wallet_address = ? LIMIT 1",
                (normalized_address,),
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def get_watchlists_for_wallet(self, wallet_address: str) -> list[str]:
        """Get all watchlist IDs that contain a wallet address."""
        normalized_address = wallet_address.lower()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT watchlist_id FROM watchlist_items WHERE wallet_address = ?",
                (normalized_address,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_watchlist(self, watchlist_id: str) -> bool:
        """Delete a watchlist and all its items."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Delete items first (cascade should handle this, but be explicit)
            conn.execute(
                "DELETE FROM watchlist_items WHERE watchlist_id = ?", (watchlist_id,)
            )
            cursor = conn.execute(
                "DELETE FROM watchlists WHERE id = ?", (watchlist_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("watchlist_deleted", watchlist_id=watchlist_id)
            return deleted
        finally:
            conn.close()

    # =========================================================================
    # Alert subscription methods
    # =========================================================================

    def create_alert_subscription(
        self,
        watchlist_id: str,
        rule_type: str,
        rule_config: dict[str, Any],
        cooldown_seconds: int = 300,
    ) -> AlertSubscriptionSchema:
        """Create an alert subscription for a watchlist."""
        subscription_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO alert_subscriptions
                (id, watchlist_id, rule_type, rule_config, cooldown_seconds,
                 is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    subscription_id,
                    watchlist_id,
                    rule_type,
                    json.dumps(rule_config),
                    cooldown_seconds,
                    now,
                    now,
                ),
            )
            conn.commit()
            logger.info(
                "alert_subscription_created",
                subscription_id=subscription_id,
                watchlist_id=watchlist_id,
                rule_type=rule_type,
            )
            return AlertSubscriptionSchema(
                id=subscription_id,
                watchlist_id=watchlist_id,
                rule_type=rule_type,
                rule_config=rule_config,
                cooldown_seconds=cooldown_seconds,
                is_active=True,
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now),
            )
        finally:
            conn.close()

    def get_alert_subscription(
        self, subscription_id: str
    ) -> AlertSubscriptionSchema | None:
        """Get an alert subscription by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM alert_subscriptions WHERE id = ?", (subscription_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_alert_subscription(row)
        finally:
            conn.close()

    def list_alert_subscriptions(
        self,
        watchlist_id: str | None = None,
        rule_type: str | None = None,
        is_active: bool | None = True,
    ) -> list[AlertSubscriptionSchema]:
        """List alert subscriptions, optionally filtered by watchlist and/or rule type."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM alert_subscriptions WHERE 1=1"
            params: list[Any] = []

            if watchlist_id:
                query += " AND watchlist_id = ?"
                params.append(watchlist_id)

            if rule_type:
                query += " AND rule_type = ?"
                params.append(rule_type)

            if is_active is not None:
                query += " AND is_active = ?"
                params.append(1 if is_active else 0)

            query += " ORDER BY created_at DESC"
            cursor = conn.execute(query, params)
            return [self._row_to_alert_subscription(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _row_to_alert_subscription(
        self, row: sqlite3.Row
    ) -> AlertSubscriptionSchema:
        """Convert a database row to an AlertSubscriptionSchema."""
        return AlertSubscriptionSchema(
            id=row["id"],
            watchlist_id=row["watchlist_id"],
            rule_type=row["rule_type"],
            rule_config=json.loads(row["rule_config"]),
            cooldown_seconds=row["cooldown_seconds"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def set_alert_subscription_active(
        self, subscription_id: str, is_active: bool
    ) -> bool:
        """Enable or disable an alert subscription."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "UPDATE alert_subscriptions SET is_active = ?, updated_at = ? WHERE id = ?",
                (1 if is_active else 0, now, subscription_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # =========================================================================
    # Alert methods
    # =========================================================================

    def create_alert(
        self,
        subscription_id: str,
        event_id: str,
        market_id: str,
        wallet_address: str,
        trade_data: dict[str, Any] | None = None,
    ) -> AlertSchema:
        """Create a new alert."""
        alert_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO alerts
                (id, subscription_id, event_id, market_id, wallet_address,
                 trade_data, triggered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    subscription_id,
                    event_id,
                    market_id,
                    wallet_address.lower(),
                    json.dumps(trade_data) if trade_data else None,
                    now,
                ),
            )
            conn.commit()
            logger.info(
                "alert_created",
                alert_id=alert_id,
                subscription_id=subscription_id,
                event_id=event_id,
            )
            return AlertSchema(
                id=alert_id,
                subscription_id=subscription_id,
                event_id=event_id,
                market_id=market_id,
                wallet_address=wallet_address.lower(),
                trade_data=trade_data,
                acknowledged=False,
                acknowledged_at=None,
                triggered_at=datetime.fromisoformat(now),
            )
        finally:
            conn.close()

    def alert_exists_for_event(self, event_id: str) -> bool:
        """Check if an alert already exists for an event (deduplication)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT 1 FROM alerts WHERE event_id = ? LIMIT 1", (event_id,)
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def list_alerts(
        self,
        subscription_id: str | None = None,
        unacknowledged_only: bool = False,
        limit: int = 100,
    ) -> list[AlertSchema]:
        """List alerts, optionally filtered."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM alerts WHERE 1=1"
            params: list[Any] = []

            if subscription_id:
                query += " AND subscription_id = ?"
                params.append(subscription_id)

            if unacknowledged_only:
                query += " AND acknowledged = 0"

            query += " ORDER BY triggered_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [self._row_to_alert(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "UPDATE alerts SET acknowledged = 1, acknowledged_at = ? WHERE id = ?",
                (now, alert_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_last_alert_time(
        self,
        subscription_id: str,
        wallet_address: str,
        market_id: str,
    ) -> datetime | None:
        """
        Get the timestamp of the last alert for a specific wallet/market combination.

        Used for implementing per-rule cooldown windows to prevent alert spam.

        Args:
            subscription_id: The subscription to check
            wallet_address: The wallet address (will be normalized to lowercase)
            market_id: The market ID

        Returns:
            The triggered_at timestamp of the last alert, or None if no alerts exist
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT triggered_at FROM alerts
                WHERE subscription_id = ?
                AND wallet_address = ?
                AND market_id = ?
                ORDER BY triggered_at DESC
                LIMIT 1
                """,
                (subscription_id, wallet_address.lower(), market_id),
            )
            row = cursor.fetchone()
            if row:
                return datetime.fromisoformat(row["triggered_at"])
            return None
        finally:
            conn.close()

    def is_within_cooldown(
        self,
        subscription_id: str,
        wallet_address: str,
        market_id: str,
        cooldown_seconds: int,
    ) -> bool:
        """
        Check if alerting for a wallet/market combination is within the cooldown window.

        Args:
            subscription_id: The subscription to check
            wallet_address: The wallet address
            market_id: The market ID
            cooldown_seconds: The cooldown duration in seconds

        Returns:
            True if an alert was triggered within the cooldown window, False otherwise
        """
        if cooldown_seconds <= 0:
            return False

        last_alert_time = self.get_last_alert_time(
            subscription_id, wallet_address, market_id
        )
        if not last_alert_time:
            return False

        now = datetime.now(timezone.utc)
        elapsed_seconds = (now - last_alert_time).total_seconds()
        return elapsed_seconds < cooldown_seconds

    def _row_to_alert(self, row: sqlite3.Row) -> AlertSchema:
        """Convert a database row to an AlertSchema."""
        return AlertSchema(
            id=row["id"],
            subscription_id=row["subscription_id"],
            event_id=row["event_id"],
            market_id=row["market_id"],
            wallet_address=row["wallet_address"],
            trade_data=json.loads(row["trade_data"]) if row["trade_data"] else None,
            acknowledged=bool(row["acknowledged"]),
            acknowledged_at=(
                datetime.fromisoformat(row["acknowledged_at"])
                if row["acknowledged_at"]
                else None
            ),
            triggered_at=datetime.fromisoformat(row["triggered_at"]),
        )

    # =========================================================================
    # Analytics session methods
    # =========================================================================

    def create_analytics_session(
        self, idle_timeout_minutes: int = 120
    ) -> AnalyticsSessionSchema:
        """Create a new analytics session."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO analytics_sessions
                (session_id, started_at, last_activity_at, idle_timeout_minutes,
                 status, queries_run, rows_accessed, updated_at)
                VALUES (?, ?, ?, ?, 'active', 0, 0, ?)
                """,
                (session_id, now, now, idle_timeout_minutes, now),
            )
            conn.commit()
            logger.info("analytics_session_created", session_id=session_id)
            return AnalyticsSessionSchema(
                session_id=session_id,
                started_at=datetime.fromisoformat(now),
                ended_at=None,
                last_activity_at=datetime.fromisoformat(now),
                idle_timeout_minutes=idle_timeout_minutes,
                status="active",
                queries_run=0,
                rows_accessed=0,
                updated_at=datetime.fromisoformat(now),
            )
        finally:
            conn.close()

    def get_analytics_session(
        self, session_id: str
    ) -> AnalyticsSessionSchema | None:
        """Get an analytics session by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM analytics_sessions WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_analytics_session(row)
        finally:
            conn.close()

    def update_analytics_session_activity(
        self,
        session_id: str,
        queries_increment: int = 1,
        rows_increment: int = 0,
    ) -> bool:
        """Update session activity (touch last_activity_at, increment counters)."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                UPDATE analytics_sessions
                SET last_activity_at = ?,
                    queries_run = queries_run + ?,
                    rows_accessed = rows_accessed + ?,
                    updated_at = ?
                WHERE session_id = ? AND status = 'active'
                """,
                (now, queries_increment, rows_increment, now, session_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def end_analytics_session(self, session_id: str) -> bool:
        """End an analytics session."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                UPDATE analytics_sessions
                SET ended_at = ?, status = 'ended', updated_at = ?
                WHERE session_id = ? AND status = 'active'
                """,
                (now, now, session_id),
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info("analytics_session_ended", session_id=session_id)
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_active_analytics_sessions(self) -> list[AnalyticsSessionSchema]:
        """List all active analytics sessions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM analytics_sessions WHERE status = 'active' "
                "ORDER BY last_activity_at DESC"
            )
            return [self._row_to_analytics_session(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def cleanup_expired_analytics_sessions(self) -> int:
        """End sessions that have exceeded their idle timeout. Returns count cleaned."""
        now = datetime.now(timezone.utc)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Find expired sessions
            cursor = conn.execute(
                "SELECT session_id, last_activity_at, idle_timeout_minutes "
                "FROM analytics_sessions WHERE status = 'active'"
            )
            expired_ids = []
            for row in cursor.fetchall():
                last_activity = datetime.fromisoformat(row["last_activity_at"])
                timeout_minutes = row["idle_timeout_minutes"]
                elapsed_minutes = (now - last_activity).total_seconds() / 60
                if elapsed_minutes > timeout_minutes:
                    expired_ids.append(row["session_id"])

            # End expired sessions
            if expired_ids:
                now_str = now.isoformat()
                for session_id in expired_ids:
                    conn.execute(
                        "UPDATE analytics_sessions "
                        "SET ended_at = ?, status = 'expired', updated_at = ? "
                        "WHERE session_id = ?",
                        (now_str, now_str, session_id),
                    )
                conn.commit()
                logger.info(
                    "analytics_sessions_expired", count=len(expired_ids)
                )

            return len(expired_ids)
        finally:
            conn.close()

    def _row_to_analytics_session(
        self, row: sqlite3.Row
    ) -> AnalyticsSessionSchema:
        """Convert a database row to an AnalyticsSessionSchema."""
        return AnalyticsSessionSchema(
            session_id=row["session_id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=(
                datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
            ),
            last_activity_at=datetime.fromisoformat(row["last_activity_at"]),
            idle_timeout_minutes=row["idle_timeout_minutes"],
            status=row["status"],
            queries_run=row["queries_run"],
            rows_accessed=row["rows_accessed"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # =========================================================================
    # Runtime Mode Management
    # =========================================================================

    def get_mode_state(self, mode_name: str) -> ModeStateSchema | None:
        """Get the current state of a runtime mode."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM mode_state WHERE mode_name = ?", (mode_name,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_mode_state(row)
        finally:
            conn.close()

    def get_all_mode_states(self) -> dict[str, ModeStateSchema]:
        """Get all runtime mode states."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM mode_state")
            return {
                row["mode_name"]: self._row_to_mode_state(row)
                for row in cursor.fetchall()
            }
        finally:
            conn.close()

    def get_ingest_mode(self) -> IngestMode:
        """Get the current ingestion mode."""
        state = self.get_mode_state("ingest_mode")
        if not state:
            return IngestMode.OFF
        return IngestMode(state.mode_value)

    def get_analytics_mode(self) -> AnalyticsMode:
        """Get the current analytics mode."""
        state = self.get_mode_state("analytics_mode")
        if not state:
            return AnalyticsMode.OFF
        return AnalyticsMode(state.mode_value)

    def set_mode(
        self,
        mode_name: str,
        new_value: str,
        initiated_by: str = "system",
        force: bool = False,
    ) -> tuple[bool, str]:
        """
        Set a runtime mode to a new value with safe transition semantics.

        Args:
            mode_name: The mode to change (ingest_mode or analytics_mode)
            new_value: The target mode value
            initiated_by: Who is initiating this transition
            force: Force transition even if one is in progress

        Returns:
            Tuple of (success, message)
        """
        now = datetime.now(timezone.utc).isoformat()

        # Validate mode name
        if mode_name not in ("ingest_mode", "analytics_mode"):
            return False, f"Unknown mode: {mode_name}"

        # Validate mode value
        if mode_name == "ingest_mode":
            try:
                IngestMode(new_value)
            except ValueError:
                valid = [m.value for m in IngestMode]
                return False, f"Invalid ingest_mode value: {new_value}. Valid: {valid}"
        else:
            try:
                AnalyticsMode(new_value)
            except ValueError:
                valid = [m.value for m in AnalyticsMode]
                return False, f"Invalid analytics_mode value: {new_value}. Valid: {valid}"

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Get current state
            cursor = conn.execute(
                "SELECT * FROM mode_state WHERE mode_name = ?", (mode_name,)
            )
            row = cursor.fetchone()

            if not row:
                # Mode doesn't exist, create it
                conn.execute(
                    """
                    INSERT INTO mode_state
                    (mode_name, mode_value, transitioned_at, transitioned_by, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (mode_name, new_value, now, initiated_by, now),
                )
                conn.commit()
                logger.info(
                    "mode_created",
                    mode_name=mode_name,
                    value=new_value,
                    initiated_by=initiated_by,
                )
                return True, f"Mode {mode_name} created with value {new_value}"

            current_value = row["mode_value"]
            is_transitioning = bool(row["is_transitioning"])

            # Check if already at target value
            if current_value == new_value:
                return True, f"Mode {mode_name} is already {new_value}"

            # Check for in-progress transition
            if is_transitioning and not force:
                return (
                    False,
                    f"Mode {mode_name} has a transition in progress. Use force=True to override.",
                )

            # Begin transition
            conn.execute(
                """
                UPDATE mode_state SET
                    is_transitioning = 1,
                    transition_started_at = ?,
                    updated_at = ?
                WHERE mode_name = ?
                """,
                (now, now, mode_name),
            )
            conn.commit()

            # Complete transition (in real system, this might involve stopping/starting services)
            conn.execute(
                """
                UPDATE mode_state SET
                    mode_value = ?,
                    previous_value = ?,
                    transitioned_at = ?,
                    transitioned_by = ?,
                    is_transitioning = 0,
                    transition_started_at = NULL,
                    updated_at = ?
                WHERE mode_name = ?
                """,
                (new_value, current_value, now, initiated_by, now, mode_name),
            )
            conn.commit()

            logger.info(
                "mode_transitioned",
                mode_name=mode_name,
                from_value=current_value,
                to_value=new_value,
                initiated_by=initiated_by,
            )
            return True, f"Mode {mode_name} transitioned from {current_value} to {new_value}"

        finally:
            conn.close()

    def set_ingest_mode(
        self, mode: IngestMode, initiated_by: str = "system", force: bool = False
    ) -> tuple[bool, str]:
        """Set the ingestion mode with safe transition semantics."""
        return self.set_mode("ingest_mode", mode.value, initiated_by, force)

    def set_analytics_mode(
        self, mode: AnalyticsMode, initiated_by: str = "system", force: bool = False
    ) -> tuple[bool, str]:
        """Set the analytics mode with safe transition semantics."""
        return self.set_mode("analytics_mode", mode.value, initiated_by, force)

    def is_mode_transitioning(self, mode_name: str) -> bool:
        """Check if a mode is currently transitioning."""
        state = self.get_mode_state(mode_name)
        return state.is_transitioning if state else False

    def _row_to_mode_state(self, row: sqlite3.Row) -> ModeStateSchema:
        """Convert a database row to a ModeStateSchema."""
        return ModeStateSchema(
            mode_name=row["mode_name"],
            mode_value=row["mode_value"],
            previous_value=row["previous_value"],
            transitioned_at=datetime.fromisoformat(row["transitioned_at"]),
            transitioned_by=row["transitioned_by"],
            is_transitioning=bool(row["is_transitioning"]),
            transition_started_at=(
                datetime.fromisoformat(row["transition_started_at"])
                if row["transition_started_at"]
                else None
            ),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
