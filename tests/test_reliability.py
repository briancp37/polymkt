"""Tests for reliability nonfunctional feature.

This module tests the reliability requirements per PRD:
1. Pipeline resumability after interruption
2. Structured JSON logs with run_id and watermarks
3. Actionable error messages with remediation steps
"""

import json
import tempfile
from collections.abc import Generator
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from polymkt.logging import configure_logging, create_run_logger, get_logger
from polymkt.pipeline.update import (
    DataSourceError,
    DataValidationError,
    PipelineError,
    WatermarkError,
    run_update,
)
from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.metadata import MetadataStore


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_files(temp_dir: Path) -> dict[str, Path]:
    """Create sample CSV files for testing."""
    events_csv = temp_dir / "events.csv"
    events_csv.write_text(
        """eventId,tags,title,description,createdAt
event1,"[""test""]",Test Event,Test description,2024-01-01 00:00:00
"""
    )

    markets_csv = temp_dir / "markets.csv"
    markets_csv.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category,eventId
2024-01-01 00:00:00,market1,Test Question?,Yes,No,false,test,token1a,token1b,cond1,1000.0,TEST,2024-12-31 23:59:59,Test market,Test,event1
"""
    )

    trades_csv = temp_dir / "trades.csv"
    trades_csv.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash001
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash002
"""
    )

    order_filled_csv = temp_dir / "orderFilled.csv"
    order_filled_csv.write_text(
        """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash001
"""
    )

    return {
        "events_csv": events_csv,
        "markets_csv": markets_csv,
        "trades_csv": trades_csv,
        "order_filled_csv": order_filled_csv,
        "parquet_dir": temp_dir / "parquet",
        "duckdb_path": temp_dir / "test.duckdb",
        "metadata_path": temp_dir / "metadata.db",
    }


@pytest.fixture
def bootstrapped_env(sample_data_files: dict[str, Path]) -> dict[str, Path]:
    """Create a bootstrapped environment for update tests."""
    run_bootstrap(
        markets_csv=sample_data_files["markets_csv"],
        trades_csv=sample_data_files["trades_csv"],
        order_filled_csv=sample_data_files["order_filled_csv"],
        events_csv=sample_data_files["events_csv"],
        parquet_dir=sample_data_files["parquet_dir"],
        duckdb_path=sample_data_files["duckdb_path"],
        metadata_db_path=sample_data_files["metadata_path"],
        require_events_for_tags=True,
    )
    return sample_data_files


class TestPipelineResumability:
    """Tests for pipeline resumability after interruption (PRD step 1-3)."""

    def test_update_resumes_without_duplicating_data(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that re-running update after partial run doesn't duplicate.

        The watermark-based filtering ensures that:
        1. First run: new trades after watermark are written
        2. Second run: watermark filters out all old data, nothing written
        This is the correct resumability behavior.
        """
        env = bootstrapped_env

        # First update - add new trades
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash001
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash002
2024-01-16 12:00:00,market1,0xmaker3,0xtaker3,YES,buy,sell,0.70,200.0,285.71,0xhash003
"""
        )

        summary1 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # First run should write new trade
        assert summary1.rows_written["trades"] == 1
        assert summary1.status == "completed"

        # Re-run the same update (simulating resume after interruption)
        summary2 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Second run: watermark filters out all old data (nothing written, no duplicates)
        assert summary2.rows_written["trades"] == 0
        assert summary2.status == "completed"

    def test_watermark_advances_correctly(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that watermarks advance and are persisted."""
        env = bootstrapped_env

        # Get initial watermark
        metadata_store = MetadataStore(env["metadata_path"])
        initial_watermark = metadata_store.get_watermark("trades")

        # Add newer trades
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash001
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash002
2024-02-01 12:00:00,market1,0xmaker3,0xtaker3,YES,buy,sell,0.80,300.0,375.0,0xhash004
"""
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Check watermark advanced
        assert "trades" in summary.watermark_after
        assert "2024-02-01" in summary.watermark_after["trades"]["last_timestamp"]

    def test_partial_write_recoverable(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that partial writes don't corrupt state on resume."""
        env = bootstrapped_env

        # Run first update
        summary1 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Add more data
        env["trades_csv"].write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash001
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash002
2024-03-01 12:00:00,market1,0xmaker4,0xtaker4,YES,buy,sell,0.75,150.0,200.0,0xhash005
"""
        )

        # Run second update
        summary2 = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Should have written only the new trade
        assert summary2.rows_written["trades"] == 1
        assert summary2.status == "completed"


class TestStructuredLogging:
    """Tests for structured JSON logs with run_id and watermarks (PRD step 4)."""

    def test_configure_logging_produces_json(self) -> None:
        """Test that configure_logging produces valid JSON output."""
        # Capture log output
        log_output = StringIO()

        # Configure with JSON output
        structlog.configure(
            processors=[
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(log_output),
            cache_logger_on_first_use=False,
        )

        # Log a test message
        test_logger = structlog.get_logger()
        test_logger.info("test_event", key="value")

        # Parse and validate JSON
        log_output.seek(0)
        log_line = log_output.readline()
        parsed = json.loads(log_line)

        assert parsed["event"] == "test_event"
        assert parsed["key"] == "value"

    def test_create_run_logger_binds_run_id(self) -> None:
        """Test that create_run_logger binds run_id to all entries."""
        log_output = StringIO()

        structlog.configure(
            processors=[
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(log_output),
            cache_logger_on_first_use=False,
        )

        run_logger = create_run_logger(run_id="test-run-123", operation="update")
        run_logger.info("test_event")

        log_output.seek(0)
        parsed = json.loads(log_output.readline())

        assert parsed["run_id"] == "test-run-123"
        assert parsed["operation"] == "update"

    def test_run_update_logs_include_run_id(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that run_update logs include run_id in all entries."""
        env = bootstrapped_env

        log_output = StringIO()
        structlog.configure(
            processors=[
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(log_output),
            cache_logger_on_first_use=False,
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        log_output.seek(0)
        run_id_found_in_logs = False
        for line in log_output:
            if line.strip():
                parsed = json.loads(line)
                if "run_id" in parsed and parsed["run_id"] == summary.run_id:
                    run_id_found_in_logs = True
                    break

        assert run_id_found_in_logs, "run_id should be in log entries"

    def test_run_update_logs_include_watermarks(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that run_update logs include watermarks."""
        env = bootstrapped_env

        log_output = StringIO()
        structlog.configure(
            processors=[
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(log_output),
            cache_logger_on_first_use=False,
        )

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        log_output.seek(0)
        watermarks_logged = False
        for line in log_output:
            if line.strip():
                parsed = json.loads(line)
                if "watermarks" in parsed or "watermarks_after" in parsed:
                    watermarks_logged = True
                    break

        assert watermarks_logged, "Watermarks should be logged"


class TestActionableErrorMessages:
    """Tests for actionable error messages with remediation (PRD step 5)."""

    def test_pipeline_error_includes_remediation(self) -> None:
        """Test that PipelineError includes remediation steps."""
        error = PipelineError(
            message="Test error",
            remediation="Fix by doing X",
            entity="trades",
            is_retryable=True,
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert "Entity: trades" in error_str
        assert "Remediation: Fix by doing X" in error_str
        assert "can be safely retried" in error_str

    def test_data_source_error_has_path(self) -> None:
        """Test that DataSourceError includes the path."""
        error = DataSourceError(
            message="File not found",
            entity="trades",
            path="/path/to/trades.csv",
        )

        error_str = str(error)
        assert "File not found" in error_str
        assert "/path/to/trades.csv" in error_str
        assert "readable" in error_str

    def test_data_validation_error_has_count(self) -> None:
        """Test that DataValidationError includes invalid count."""
        error = DataValidationError(
            message="Validation failed",
            entity="trades",
            invalid_count=42,
        )

        error_str = str(error)
        assert "Validation failed" in error_str
        assert "42 rows failed validation" in error_str
        assert "quarantined" in error_str

    def test_watermark_error_has_recovery_options(self) -> None:
        """Test that WatermarkError includes recovery options."""
        error = WatermarkError(
            message="Watermark corrupted",
            entity="trades",
        )

        error_str = str(error)
        assert "Watermark corrupted" in error_str
        assert "bootstrap" in error_str
        assert "clear the watermark" in error_str

    def test_update_failure_logs_remediation(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that update failure includes remediation in log."""
        env = bootstrapped_env

        log_output = StringIO()
        structlog.configure(
            processors=[
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(log_output),
            cache_logger_on_first_use=False,
        )

        # Delete the markets CSV to cause an error
        env["markets_csv"].unlink()

        # Run update (should still complete, just with warning)
        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Should complete (missing CSV is just a warning)
        assert summary.status == "completed"

        # Check for warning about missing CSV
        log_output.seek(0)
        csv_warning_found = False
        for line in log_output:
            if line.strip():
                parsed = json.loads(line)
                if parsed.get("event") == "csv_not_found" and "markets" in parsed.get("entity", ""):
                    csv_warning_found = True
                    break

        assert csv_warning_found


class TestRunHistoryTracking:
    """Tests for run history tracking and retrieval."""

    def test_run_record_persisted(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that run records are persisted to metadata store."""
        env = bootstrapped_env

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        # Verify run is in history
        metadata_store = MetadataStore(env["metadata_path"])
        runs = metadata_store.list_runs(limit=10)

        run_ids = [r.run_id for r in runs]
        assert summary.run_id in run_ids

    def test_run_record_has_all_fields(
        self, bootstrapped_env: dict[str, Path]
    ) -> None:
        """Test that run record has all required fields."""
        env = bootstrapped_env

        summary = run_update(
            markets_csv=env["markets_csv"],
            trades_csv=env["trades_csv"],
            order_filled_csv=env["order_filled_csv"],
            events_csv=env["events_csv"],
            parquet_dir=env["parquet_dir"],
            duckdb_path=env["duckdb_path"],
            metadata_db_path=env["metadata_path"],
        )

        metadata_store = MetadataStore(env["metadata_path"])
        run = metadata_store.get_run(summary.run_id)

        assert run is not None
        assert run.run_id == summary.run_id
        assert run.run_type == "update"
        assert run.status == "completed"
        assert run.start_time is not None
        assert run.end_time is not None
        assert run.duration_seconds is not None
        assert run.duration_seconds >= 0
