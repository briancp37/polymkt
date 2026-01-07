"""Data quality checks for trades, markets, and referential integrity."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import structlog

from polymkt.config import settings

logger = structlog.get_logger()


@dataclass
class UniquenessIssue:
    """Details about a uniqueness violation."""

    column: str
    duplicate_value: str
    occurrence_count: int


@dataclass
class RangeIssue:
    """Details about a value out of expected range."""

    column: str
    value: float
    expected_min: float | None
    expected_max: float | None
    row_identifier: str  # e.g., transaction_hash for trades


@dataclass
class ReferentialIntegrityIssue:
    """Details about a referential integrity violation."""

    source_table: str
    source_column: str
    source_value: str
    target_table: str
    target_column: str


@dataclass
class DataQualityReport:
    """Complete data quality report for an entity or the entire dataset."""

    report_id: str
    entity: str  # "trades", "markets", "all"
    run_type: str  # "bootstrap", "update"
    started_at: datetime
    completed_at: datetime

    # Uniqueness checks
    uniqueness_valid: bool
    uniqueness_issues: list[UniquenessIssue]
    duplicate_count: int

    # Range checks
    range_valid: bool
    range_issues: list[RangeIssue]
    out_of_range_count: int

    # Referential integrity checks
    referential_integrity_valid: bool
    referential_integrity_issues: list[ReferentialIntegrityIssue]
    orphaned_count: int

    # Market-specific checks
    markets_without_closed_time: int

    # Summary
    total_issues: int
    is_valid: bool


class DataQualityChecker:
    """Performs data quality checks on the Parquet/DuckDB data layer."""

    def __init__(
        self,
        parquet_dir: Path | None = None,
        db_path: Path | None = None,
        partitioned: bool = False,
    ) -> None:
        """
        Initialize the data quality checker.

        Args:
            parquet_dir: Path to Parquet files directory
            db_path: Path to SQLite database for persisting reports
            partitioned: Whether trades data is partitioned
        """
        self.parquet_dir = parquet_dir or settings.parquet_dir
        self.db_path = db_path or settings.metadata_db_path
        self.partitioned = partitioned
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create report storage table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_reports (
                    report_id TEXT PRIMARY KEY,
                    entity TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    uniqueness_valid INTEGER NOT NULL,
                    uniqueness_issues TEXT NOT NULL,
                    duplicate_count INTEGER NOT NULL,
                    range_valid INTEGER NOT NULL,
                    range_issues TEXT NOT NULL,
                    out_of_range_count INTEGER NOT NULL,
                    referential_integrity_valid INTEGER NOT NULL,
                    referential_integrity_issues TEXT NOT NULL,
                    orphaned_count INTEGER NOT NULL,
                    markets_without_closed_time INTEGER NOT NULL,
                    total_issues INTEGER NOT NULL,
                    is_valid INTEGER NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_trades_source(self) -> str:
        """Get the DuckDB source expression for trades."""
        if self.partitioned:
            return f"read_parquet('{self.parquet_dir}/trades/**/*.parquet', hive_partitioning=true)"
        return f"read_parquet('{self.parquet_dir}/trades.parquet')"

    def check_uniqueness(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_source: str,
        column: str,
        sample_limit: int = 10,
    ) -> tuple[bool, list[UniquenessIssue], int]:
        """
        Check uniqueness of a column in a table.

        Args:
            conn: DuckDB connection
            table_source: SQL source expression for the table
            column: Column to check for uniqueness
            sample_limit: Maximum number of duplicate examples to return

        Returns:
            Tuple of (is_valid, issues, total_duplicate_count)
        """
        # Find duplicates
        query = f"""
            SELECT {column}, COUNT(*) as cnt
            FROM {table_source}
            WHERE {column} IS NOT NULL
            GROUP BY {column}
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT {sample_limit}
        """
        result = conn.execute(query).fetchall()

        # Get total duplicate count
        count_query = f"""
            SELECT COUNT(*) FROM (
                SELECT {column}
                FROM {table_source}
                WHERE {column} IS NOT NULL
                GROUP BY {column}
                HAVING COUNT(*) > 1
            ) dups
        """
        count_result = conn.execute(count_query).fetchone()
        duplicate_count: int = count_result[0] if count_result else 0

        issues = [
            UniquenessIssue(column=column, duplicate_value=str(row[0]), occurrence_count=row[1])
            for row in result
        ]

        return (duplicate_count == 0, issues, duplicate_count)

    def check_trade_uniqueness(
        self, conn: duckdb.DuckDBPyConnection
    ) -> tuple[bool, list[UniquenessIssue], int]:
        """
        Check transaction_hash uniqueness for trades.

        Returns:
            Tuple of (is_valid, issues, duplicate_count)
        """
        trades_source = self._get_trades_source()
        return self.check_uniqueness(conn, trades_source, "transaction_hash")

    def check_price_range(
        self,
        conn: duckdb.DuckDBPyConnection,
        min_price: float = 0.0,
        max_price: float = 1.0,
        sample_limit: int = 10,
    ) -> tuple[bool, list[RangeIssue], int]:
        """
        Check that trade prices are within expected bounds.

        Args:
            conn: DuckDB connection
            min_price: Minimum expected price
            max_price: Maximum expected price
            sample_limit: Maximum number of outlier examples to return

        Returns:
            Tuple of (is_valid, issues, out_of_range_count)
        """
        trades_source = self._get_trades_source()

        # Find out-of-range prices
        query = f"""
            SELECT price, transaction_hash
            FROM {trades_source}
            WHERE price IS NOT NULL AND (price < {min_price} OR price > {max_price})
            LIMIT {sample_limit}
        """
        result = conn.execute(query).fetchall()

        # Get total count
        count_query = f"""
            SELECT COUNT(*)
            FROM {trades_source}
            WHERE price IS NOT NULL AND (price < {min_price} OR price > {max_price})
        """
        count_result = conn.execute(count_query).fetchone()
        out_of_range_count: int = count_result[0] if count_result else 0

        issues = [
            RangeIssue(
                column="price",
                value=row[0],
                expected_min=min_price,
                expected_max=max_price,
                row_identifier=row[1],
            )
            for row in result
        ]

        return (out_of_range_count == 0, issues, out_of_range_count)

    def check_usd_amount_range(
        self,
        conn: duckdb.DuckDBPyConnection,
        min_amount: float = 0.0,
        max_amount: float | None = None,
        sample_limit: int = 10,
    ) -> tuple[bool, list[RangeIssue], int]:
        """
        Check that usd_amount values are non-negative and within reasonable bounds.

        Args:
            conn: DuckDB connection
            min_amount: Minimum expected amount (typically 0)
            max_amount: Maximum expected amount (None for no upper limit)
            sample_limit: Maximum number of outlier examples to return

        Returns:
            Tuple of (is_valid, issues, out_of_range_count)
        """
        trades_source = self._get_trades_source()

        # Build condition
        if max_amount is not None:
            condition = f"usd_amount < {min_amount} OR usd_amount > {max_amount}"
        else:
            condition = f"usd_amount < {min_amount}"

        # Find out-of-range values
        query = f"""
            SELECT usd_amount, transaction_hash
            FROM {trades_source}
            WHERE usd_amount IS NOT NULL AND ({condition})
            LIMIT {sample_limit}
        """
        result = conn.execute(query).fetchall()

        # Get total count
        count_query = f"""
            SELECT COUNT(*)
            FROM {trades_source}
            WHERE usd_amount IS NOT NULL AND ({condition})
        """
        count_result = conn.execute(count_query).fetchone()
        out_of_range_count: int = count_result[0] if count_result else 0

        issues = [
            RangeIssue(
                column="usd_amount",
                value=row[0],
                expected_min=min_amount,
                expected_max=max_amount,
                row_identifier=row[1],
            )
            for row in result
        ]

        return (out_of_range_count == 0, issues, out_of_range_count)

    def check_trades_referential_integrity(
        self, conn: duckdb.DuckDBPyConnection, sample_limit: int = 10
    ) -> tuple[bool, list[ReferentialIntegrityIssue], int]:
        """
        Check that all trades.market_id values exist in markets.id.

        Returns:
            Tuple of (is_valid, issues, orphaned_count)
        """
        trades_source = self._get_trades_source()
        markets_source = f"read_parquet('{self.parquet_dir}/markets.parquet')"

        # Find orphaned market_ids
        query = f"""
            SELECT DISTINCT t.market_id
            FROM {trades_source} t
            LEFT JOIN {markets_source} m ON t.market_id = m.id
            WHERE m.id IS NULL AND t.market_id IS NOT NULL
            LIMIT {sample_limit}
        """
        result = conn.execute(query).fetchall()

        # Get total count of orphaned market_ids
        count_query = f"""
            SELECT COUNT(DISTINCT t.market_id)
            FROM {trades_source} t
            LEFT JOIN {markets_source} m ON t.market_id = m.id
            WHERE m.id IS NULL AND t.market_id IS NOT NULL
        """
        count_result = conn.execute(count_query).fetchone()
        orphaned_count: int = count_result[0] if count_result else 0

        issues = [
            ReferentialIntegrityIssue(
                source_table="trades",
                source_column="market_id",
                source_value=row[0],
                target_table="markets",
                target_column="id",
            )
            for row in result
        ]

        return (orphaned_count == 0, issues, orphaned_count)

    def check_markets_closed_time(
        self, conn: duckdb.DuckDBPyConnection
    ) -> int:
        """
        Check how many markets are missing closed_time (needed for expiry-based backtests).

        Returns:
            Count of markets without closed_time
        """
        markets_source = f"read_parquet('{self.parquet_dir}/markets.parquet')"

        # First check if closed_time column exists by creating a temp view and describing it
        try:
            conn.execute(f"CREATE OR REPLACE TEMP VIEW _temp_markets AS SELECT * FROM {markets_source}")
            schema_result = conn.execute("DESCRIBE _temp_markets").fetchall()
            columns = [row[0] for row in schema_result]
            if "closed_time" not in columns:
                # Column doesn't exist, can't check
                return 0
        except Exception:
            return 0

        query = f"""
            SELECT COUNT(*)
            FROM {markets_source}
            WHERE closed_time IS NULL
        """
        result = conn.execute(query).fetchone()
        return int(result[0]) if result else 0

    def run_full_check(
        self,
        run_type: str = "bootstrap",
        report_id: str | None = None,
    ) -> DataQualityReport:
        """
        Run all data quality checks and produce a comprehensive report.

        Args:
            run_type: Type of pipeline run ("bootstrap" or "update")
            report_id: Optional report ID (auto-generated if not provided)

        Returns:
            Complete DataQualityReport
        """
        import uuid

        if report_id is None:
            report_id = str(uuid.uuid4())

        started_at = datetime.now(timezone.utc)

        # Connect to DuckDB
        conn = duckdb.connect(":memory:")

        # Initialize aggregated results
        all_uniqueness_issues: list[UniquenessIssue] = []
        all_range_issues: list[RangeIssue] = []
        all_referential_issues: list[ReferentialIntegrityIssue] = []

        total_duplicates = 0
        total_out_of_range = 0
        total_orphaned = 0
        markets_without_closed_time = 0

        uniqueness_valid = True
        range_valid = True
        referential_valid = True

        try:
            # Check trades exist
            trades_source = self._get_trades_source()
            try:
                conn.execute(f"SELECT 1 FROM {trades_source} LIMIT 1").fetchone()
                trades_exist = True
            except Exception:
                trades_exist = False

            # Check markets exist
            markets_source = f"read_parquet('{self.parquet_dir}/markets.parquet')"
            try:
                conn.execute(f"SELECT 1 FROM {markets_source} LIMIT 1").fetchone()
                markets_exist = True
            except Exception:
                markets_exist = False

            if trades_exist:
                # 1. Check transaction_hash uniqueness
                uniq_valid, uniq_issues, uniq_count = self.check_trade_uniqueness(conn)
                uniqueness_valid = uniqueness_valid and uniq_valid
                all_uniqueness_issues.extend(uniq_issues)
                total_duplicates += uniq_count

                # 2. Check price range
                price_valid, price_issues, price_count = self.check_price_range(conn)
                range_valid = range_valid and price_valid
                all_range_issues.extend(price_issues)
                total_out_of_range += price_count

                # 3. Check usd_amount range (non-negative)
                usd_valid, usd_issues, usd_count = self.check_usd_amount_range(conn)
                range_valid = range_valid and usd_valid
                all_range_issues.extend(usd_issues)
                total_out_of_range += usd_count

            if trades_exist and markets_exist:
                # 4. Check referential integrity
                ref_valid, ref_issues, ref_count = self.check_trades_referential_integrity(conn)
                referential_valid = referential_valid and ref_valid
                all_referential_issues.extend(ref_issues)
                total_orphaned += ref_count

            if markets_exist:
                # 5. Check markets have closed_time for backtests
                markets_without_closed_time = self.check_markets_closed_time(conn)

        finally:
            conn.close()

        completed_at = datetime.now(timezone.utc)

        total_issues = (
            total_duplicates
            + total_out_of_range
            + total_orphaned
            # markets_without_closed_time is a warning, not an issue
        )

        is_valid = uniqueness_valid and range_valid and referential_valid

        report = DataQualityReport(
            report_id=report_id,
            entity="all",
            run_type=run_type,
            started_at=started_at,
            completed_at=completed_at,
            uniqueness_valid=uniqueness_valid,
            uniqueness_issues=all_uniqueness_issues,
            duplicate_count=total_duplicates,
            range_valid=range_valid,
            range_issues=all_range_issues,
            out_of_range_count=total_out_of_range,
            referential_integrity_valid=referential_valid,
            referential_integrity_issues=all_referential_issues,
            orphaned_count=total_orphaned,
            markets_without_closed_time=markets_without_closed_time,
            total_issues=total_issues,
            is_valid=is_valid,
        )

        # Persist report
        self._save_report(report)

        logger.info(
            "data_quality_check_completed",
            report_id=report_id,
            is_valid=is_valid,
            total_issues=total_issues,
            duplicates=total_duplicates,
            out_of_range=total_out_of_range,
            orphaned=total_orphaned,
            markets_without_closed_time=markets_without_closed_time,
        )

        return report

    def _save_report(self, report: DataQualityReport) -> None:
        """Save a report to the database."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Serialize issues to JSON
            uniqueness_issues_json = json.dumps(
                [
                    {
                        "column": i.column,
                        "duplicate_value": i.duplicate_value,
                        "occurrence_count": i.occurrence_count,
                    }
                    for i in report.uniqueness_issues
                ]
            )
            range_issues_json = json.dumps(
                [
                    {
                        "column": i.column,
                        "value": i.value,
                        "expected_min": i.expected_min,
                        "expected_max": i.expected_max,
                        "row_identifier": i.row_identifier,
                    }
                    for i in report.range_issues
                ]
            )
            referential_issues_json = json.dumps(
                [
                    {
                        "source_table": i.source_table,
                        "source_column": i.source_column,
                        "source_value": i.source_value,
                        "target_table": i.target_table,
                        "target_column": i.target_column,
                    }
                    for i in report.referential_integrity_issues
                ]
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO data_quality_reports (
                    report_id, entity, run_type, started_at, completed_at,
                    uniqueness_valid, uniqueness_issues, duplicate_count,
                    range_valid, range_issues, out_of_range_count,
                    referential_integrity_valid, referential_integrity_issues, orphaned_count,
                    markets_without_closed_time, total_issues, is_valid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.entity,
                    report.run_type,
                    report.started_at.isoformat(),
                    report.completed_at.isoformat(),
                    1 if report.uniqueness_valid else 0,
                    uniqueness_issues_json,
                    report.duplicate_count,
                    1 if report.range_valid else 0,
                    range_issues_json,
                    report.out_of_range_count,
                    1 if report.referential_integrity_valid else 0,
                    referential_issues_json,
                    report.orphaned_count,
                    report.markets_without_closed_time,
                    report.total_issues,
                    1 if report.is_valid else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_report(self, report_id: str) -> DataQualityReport | None:
        """Get a report by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM data_quality_reports WHERE report_id = ?",
                (report_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_report(row)
        finally:
            conn.close()

    def list_reports(self, limit: int = 50) -> list[DataQualityReport]:
        """List recent data quality reports."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT * FROM data_quality_reports
                ORDER BY completed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [self._row_to_report(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _row_to_report(self, row: sqlite3.Row) -> DataQualityReport:
        """Convert a database row to a DataQualityReport."""
        uniqueness_issues_data = json.loads(row["uniqueness_issues"])
        range_issues_data = json.loads(row["range_issues"])
        referential_issues_data = json.loads(row["referential_integrity_issues"])

        return DataQualityReport(
            report_id=row["report_id"],
            entity=row["entity"],
            run_type=row["run_type"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]),
            uniqueness_valid=bool(row["uniqueness_valid"]),
            uniqueness_issues=[
                UniquenessIssue(
                    column=i["column"],
                    duplicate_value=i["duplicate_value"],
                    occurrence_count=i["occurrence_count"],
                )
                for i in uniqueness_issues_data
            ],
            duplicate_count=row["duplicate_count"],
            range_valid=bool(row["range_valid"]),
            range_issues=[
                RangeIssue(
                    column=i["column"],
                    value=i["value"],
                    expected_min=i["expected_min"],
                    expected_max=i["expected_max"],
                    row_identifier=i["row_identifier"],
                )
                for i in range_issues_data
            ],
            out_of_range_count=row["out_of_range_count"],
            referential_integrity_valid=bool(row["referential_integrity_valid"]),
            referential_integrity_issues=[
                ReferentialIntegrityIssue(
                    source_table=i["source_table"],
                    source_column=i["source_column"],
                    source_value=i["source_value"],
                    target_table=i["target_table"],
                    target_column=i["target_column"],
                )
                for i in referential_issues_data
            ],
            orphaned_count=row["orphaned_count"],
            markets_without_closed_time=row["markets_without_closed_time"],
            total_issues=row["total_issues"],
            is_valid=bool(row["is_valid"]),
        )
