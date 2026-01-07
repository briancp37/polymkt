"""Tests for data quality checks feature."""

import json
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app
from polymkt.storage.data_quality import (
    DataQualityChecker,
    DataQualityReport,
    RangeIssue,
    ReferentialIntegrityIssue,
    UniquenessIssue,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def parquet_dir(temp_dir):
    """Create a parquet directory structure."""
    parquet_path = temp_dir / "parquet"
    parquet_path.mkdir(parents=True, exist_ok=True)
    return parquet_path


@pytest.fixture
def db_path(temp_dir):
    """Create a database path."""
    return temp_dir / "metadata" / "metadata.db"


def create_trades_parquet(parquet_dir: Path, trades_data: list[dict]):
    """Helper to create a trades Parquet file."""
    if not trades_data:
        trades_data = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "nonusdc_side": "yes",
                "maker_direction": "buy",
                "taker_direction": "sell",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            }
        ]

    df = pd.DataFrame(trades_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_dir / "trades.parquet")


def create_markets_parquet(parquet_dir: Path, markets_data: list[dict]):
    """Helper to create a markets Parquet file."""
    if not markets_data:
        markets_data = [
            {
                "id": "market1",
                "question": "Test market?",
                "created_at": datetime.now(timezone.utc),
                "closed_time": datetime.now(timezone.utc) + timedelta(days=30),
            }
        ]

    df = pd.DataFrame(markets_data)
    # Ensure closed_time column exists with proper type even when None
    if "closed_time" in df.columns:
        df["closed_time"] = pd.to_datetime(df["closed_time"], utc=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_dir / "markets.parquet")


class TestDataQualityCheckerCreation:
    """Tests for DataQualityChecker initialization."""

    def test_init_creates_report_table(self, parquet_dir, db_path):
        """Test that initialization creates the reports table."""
        checker = DataQualityChecker(
            parquet_dir=parquet_dir,
            db_path=db_path,
        )

        # Table should exist
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='data_quality_reports'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_creates_parent_dirs(self, parquet_dir, temp_dir):
        """Test that initialization creates parent directories for db_path."""
        nested_db_path = temp_dir / "nested" / "path" / "metadata.db"

        checker = DataQualityChecker(
            parquet_dir=parquet_dir,
            db_path=nested_db_path,
        )

        assert nested_db_path.parent.exists()


class TestUniquenessCheck:
    """Tests for transaction_hash uniqueness validation."""

    def test_uniqueness_valid_with_unique_hashes(self, parquet_dir, db_path):
        """Test that unique transaction hashes pass validation."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": f"0xhash{i}",
            }
            for i in range(10)
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, [{"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)}])

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.uniqueness_valid is True
        assert report.duplicate_count == 0
        assert len(report.uniqueness_issues) == 0

    def test_uniqueness_invalid_with_duplicates(self, parquet_dir, db_path):
        """Test that duplicate transaction hashes are detected."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xduplicate_hash",
            }
            for _ in range(3)
        ]
        # Add a unique hash
        trades.append({
            "timestamp": datetime.now(timezone.utc),
            "market_id": "market1",
            "maker": "0xmaker",
            "taker": "0xtaker",
            "price": 0.5,
            "usd_amount": 100.0,
            "token_amount": 200.0,
            "transaction_hash": "0xunique_hash",
        })

        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, [{"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)}])

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.uniqueness_valid is False
        assert report.duplicate_count == 1  # 1 unique value with duplicates
        assert len(report.uniqueness_issues) >= 1
        assert report.uniqueness_issues[0].duplicate_value == "0xduplicate_hash"
        assert report.uniqueness_issues[0].occurrence_count == 3


class TestRangeCheck:
    """Tests for price and amount range validation."""

    def test_price_valid_in_range(self, parquet_dir, db_path):
        """Test that prices between 0 and 1 pass validation."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": price,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": f"0xhash{i}",
            }
            for i, price in enumerate([0.0, 0.1, 0.5, 0.9, 1.0])
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, [{"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)}])

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.range_valid is True
        assert report.out_of_range_count == 0

    def test_price_invalid_out_of_range(self, parquet_dir, db_path):
        """Test that prices outside 0-1 are detected."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": -0.1,  # Invalid
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 1.5,  # Invalid
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash2",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,  # Valid
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash3",
            },
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, [{"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)}])

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.range_valid is False
        assert report.out_of_range_count >= 2
        # At least the price issues should be present
        price_issues = [i for i in report.range_issues if i.column == "price"]
        assert len(price_issues) >= 2

    def test_usd_amount_invalid_negative(self, parquet_dir, db_path):
        """Test that negative usd_amount is detected."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": -50.0,  # Invalid
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, [{"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)}])

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.range_valid is False
        usd_issues = [i for i in report.range_issues if i.column == "usd_amount"]
        assert len(usd_issues) >= 1


class TestReferentialIntegrityCheck:
    """Tests for trades.market_id -> markets.id referential integrity."""

    def test_referential_integrity_valid(self, parquet_dir, db_path):
        """Test that valid market_id references pass validation."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market2",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.6,
                "usd_amount": 150.0,
                "token_amount": 250.0,
                "transaction_hash": "0xhash2",
            },
        ]
        markets = [
            {"id": "market1", "question": "Test 1?", "created_at": datetime.now(timezone.utc)},
            {"id": "market2", "question": "Test 2?", "created_at": datetime.now(timezone.utc)},
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.referential_integrity_valid is True
        assert report.orphaned_count == 0

    def test_referential_integrity_invalid_orphaned_market_id(self, parquet_dir, db_path):
        """Test that orphaned market_id references are detected."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",  # Exists
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "nonexistent_market",  # Does not exist
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.6,
                "usd_amount": 150.0,
                "token_amount": 250.0,
                "transaction_hash": "0xhash2",
            },
        ]
        markets = [
            {"id": "market1", "question": "Test 1?", "created_at": datetime.now(timezone.utc)},
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.referential_integrity_valid is False
        assert report.orphaned_count == 1
        assert len(report.referential_integrity_issues) >= 1
        assert report.referential_integrity_issues[0].source_value == "nonexistent_market"


class TestMarketsClosedTimeCheck:
    """Tests for markets.closed_time validation."""

    def test_markets_with_closed_time(self, parquet_dir, db_path):
        """Test that markets with closed_time don't trigger warnings."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
        ]
        markets = [
            {
                "id": "market1",
                "question": "Test?",
                "created_at": datetime.now(timezone.utc),
                "closed_time": datetime.now(timezone.utc) + timedelta(days=30),
            },
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.markets_without_closed_time == 0

    def test_markets_without_closed_time(self, parquet_dir, db_path):
        """Test that markets without closed_time are counted."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
        ]
        markets = [
            {
                "id": "market1",
                "question": "Test?",
                "created_at": datetime.now(timezone.utc),
                "closed_time": None,  # No closed_time
            },
            {
                "id": "market2",
                "question": "Test 2?",
                "created_at": datetime.now(timezone.utc),
                "closed_time": datetime.now(timezone.utc) + timedelta(days=30),
            },
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.markets_without_closed_time == 1


class TestFullReport:
    """Tests for the complete data quality report."""

    def test_run_full_check_all_valid(self, parquet_dir, db_path):
        """Test that a clean dataset produces a valid report."""
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": f"0xhash{i}",
            }
            for i in range(5)
        ]
        markets = [
            {
                "id": "market1",
                "question": "Test?",
                "created_at": datetime.now(timezone.utc),
                "closed_time": datetime.now(timezone.utc) + timedelta(days=30),
            },
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check(run_type="bootstrap")

        assert report.is_valid is True
        assert report.uniqueness_valid is True
        assert report.range_valid is True
        assert report.referential_integrity_valid is True
        assert report.total_issues == 0
        assert report.run_type == "bootstrap"
        assert report.entity == "all"

    def test_run_full_check_persists_report(self, parquet_dir, db_path):
        """Test that the report is persisted to the database."""
        create_trades_parquet(parquet_dir, None)
        create_markets_parquet(parquet_dir, None)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        # Retrieve the report
        retrieved = checker.get_report(report.report_id)
        assert retrieved is not None
        assert retrieved.report_id == report.report_id
        assert retrieved.is_valid == report.is_valid

    def test_list_reports_returns_recent(self, parquet_dir, db_path):
        """Test that list_reports returns reports in order."""
        create_trades_parquet(parquet_dir, None)
        create_markets_parquet(parquet_dir, None)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)

        # Create multiple reports
        reports = []
        for i in range(3):
            report = checker.run_full_check(run_type=f"run{i}")
            reports.append(report)

        # List reports
        listed = checker.list_reports(limit=10)
        assert len(listed) == 3
        # Most recent first
        assert listed[0].report_id == reports[2].report_id

    def test_report_with_multiple_issues(self, parquet_dir, db_path):
        """Test a report with multiple types of issues."""
        trades = [
            # Duplicate hash
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xdup",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "orphan_market",  # Orphan
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 1.5,  # Out of range
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xdup",  # Duplicate
            },
        ]
        markets = [
            {
                "id": "market1",
                "question": "Test?",
                "created_at": datetime.now(timezone.utc),
                "closed_time": None,  # Missing
            },
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        assert report.is_valid is False
        assert report.uniqueness_valid is False
        assert report.range_valid is False
        assert report.referential_integrity_valid is False
        assert report.total_issues > 0
        assert report.markets_without_closed_time == 1


class TestDataQualityAPI:
    """Tests for the data quality API endpoints."""

    @pytest.fixture
    def client(self, parquet_dir, db_path, monkeypatch):
        """Create a test client with mocked settings."""
        monkeypatch.setattr("polymkt.config.settings.parquet_dir", parquet_dir)
        monkeypatch.setattr("polymkt.config.settings.metadata_db_path", db_path)
        monkeypatch.setattr("polymkt.config.settings.parquet_partitioning_enabled", False)

        # Create valid data
        create_trades_parquet(parquet_dir, None)
        create_markets_parquet(parquet_dir, None)

        return TestClient(app)

    def test_run_data_quality_check_endpoint(self, client):
        """Test POST /api/data-quality/check endpoint."""
        response = client.post("/api/data-quality/check", json={"run_type": "bootstrap"})

        assert response.status_code == 200
        data = response.json()
        assert "report_id" in data
        assert "is_valid" in data
        assert data["run_type"] == "bootstrap"

    def test_list_reports_endpoint(self, client):
        """Test GET /api/data-quality/reports endpoint."""
        # First create a report
        client.post("/api/data-quality/check", json={"run_type": "test"})

        response = client.get("/api/data-quality/reports?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "reports" in data
        assert "count" in data
        assert len(data["reports"]) >= 1

    def test_get_report_endpoint(self, client):
        """Test GET /api/data-quality/reports/{report_id} endpoint."""
        # First create a report
        create_response = client.post("/api/data-quality/check")
        report_id = create_response.json()["report_id"]

        response = client.get(f"/api/data-quality/reports/{report_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["report_id"] == report_id

    def test_get_report_not_found(self, client):
        """Test GET /api/data-quality/reports/{report_id} with nonexistent ID."""
        response = client.get("/api/data-quality/reports/nonexistent-uuid")

        assert response.status_code == 404

    def test_check_endpoint_returns_issues_sample(self, parquet_dir, db_path, monkeypatch):
        """Test that the API returns issue samples."""
        monkeypatch.setattr("polymkt.config.settings.parquet_dir", parquet_dir)
        monkeypatch.setattr("polymkt.config.settings.metadata_db_path", db_path)
        monkeypatch.setattr("polymkt.config.settings.parquet_partitioning_enabled", False)

        # Create data with issues
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 1.5,  # Out of range
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
        ]
        markets = [
            {"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)},
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        client = TestClient(app)
        response = client.post("/api/data-quality/check")

        assert response.status_code == 200
        data = response.json()
        assert data["range_valid"] is False
        assert len(data["range_issues"]) >= 1


class TestDataQualityReportPersistence:
    """Tests for report persistence and retrieval."""

    def test_report_serialization_roundtrip(self, parquet_dir, db_path):
        """Test that reports survive serialization/deserialization."""
        # Create data with various issues
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xdup",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xdup",
            },
        ]
        markets = [
            {"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)},
        ]
        create_trades_parquet(parquet_dir, trades)
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        original = checker.run_full_check()

        # Retrieve and compare
        retrieved = checker.get_report(original.report_id)
        assert retrieved is not None
        assert retrieved.report_id == original.report_id
        assert retrieved.uniqueness_valid == original.uniqueness_valid
        assert len(retrieved.uniqueness_issues) == len(original.uniqueness_issues)
        assert retrieved.duplicate_count == original.duplicate_count

    def test_get_nonexistent_report(self, parquet_dir, db_path):
        """Test that getting a nonexistent report returns None."""
        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)

        report = checker.get_report("nonexistent-uuid")
        assert report is None


class TestDataQualityWithMissingData:
    """Tests for edge cases with missing data files."""

    def test_no_trades_parquet(self, parquet_dir, db_path):
        """Test check when trades.parquet doesn't exist."""
        # Only create markets
        markets = [
            {"id": "market1", "question": "Test?", "created_at": datetime.now(timezone.utc)},
        ]
        create_markets_parquet(parquet_dir, markets)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        # Should still produce a report without errors
        assert report.report_id is not None
        assert report.uniqueness_valid is True  # No trades to check
        assert report.range_valid is True

    def test_no_markets_parquet(self, parquet_dir, db_path):
        """Test check when markets.parquet doesn't exist."""
        # Only create trades
        trades = [
            {
                "timestamp": datetime.now(timezone.utc),
                "market_id": "market1",
                "maker": "0xmaker",
                "taker": "0xtaker",
                "price": 0.5,
                "usd_amount": 100.0,
                "token_amount": 200.0,
                "transaction_hash": "0xhash1",
            },
        ]
        create_trades_parquet(parquet_dir, trades)

        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        # Should still produce a report
        assert report.report_id is not None
        # Referential integrity can't be checked without markets
        assert report.referential_integrity_valid is True  # Skipped

    def test_empty_parquet_directory(self, parquet_dir, db_path):
        """Test check with no Parquet files."""
        checker = DataQualityChecker(parquet_dir=parquet_dir, db_path=db_path)
        report = checker.run_full_check()

        # Should still produce a valid report
        assert report.report_id is not None
        assert report.is_valid is True  # No data = no issues
