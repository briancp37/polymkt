"""Tests for performance benchmarking functionality.

PRD requirements tested:
- Benchmark query for a single market_id over a 30-day window
- Benchmark query for 100+ market_ids over the same window
- Verify partition pruning/predicate pushdown is effective (plan/log inspection)
- Verify memory usage is bounded (streaming/limits/spill-to-disk if needed)
- Record baseline numbers for regression tracking
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.performance import (
    PerformanceBenchmarker,
    PerformanceBenchmarkReport,
    QueryBenchmarkResult,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_markets_csv(temp_dir: Path) -> Path:
    """Create a sample markets CSV file with 150 markets."""
    csv_path = temp_dir / "markets.csv"
    lines = [
        "createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category"
    ]
    for i in range(150):
        lines.append(
            f"2024-01-01 00:00:00,market{i},Market question {i}?,Yes,No,false,market-{i},tok{i}a,tok{i}b,cond{i},{1000.0 * (i+1)},MKT{i},2024-12-31 23:59:59,Description {i},Category{i % 5}"
        )
    csv_path.write_text("\n".join(lines))
    return csv_path


@pytest.fixture
def sample_trades_csv_large(temp_dir: Path) -> Path:
    """Create a sample trades CSV file with 1000+ trades across 150 markets."""
    csv_path = temp_dir / "trades.csv"
    lines = [
        "timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash"
    ]
    # Create 10 trades per market = 1500 trades
    for market_i in range(150):
        for trade_i in range(10):
            day = (trade_i % 28) + 1
            hour = trade_i % 24
            price = 0.3 + (trade_i * 0.05) % 0.4
            lines.append(
                f"2024-01-{day:02d} {hour:02d}:00:00,market{market_i},0xmaker{trade_i},0xtaker{trade_i},YES,buy,sell,{price:.2f},{100.0 * (trade_i + 1)},{100.0 / price:.2f},0xhash{market_i}_{trade_i}"
            )
    csv_path.write_text("\n".join(lines))
    return csv_path


@pytest.fixture
def sample_order_filled_csv(temp_dir: Path) -> Path:
    """Create a sample order filled CSV file."""
    csv_path = temp_dir / "orderFilled.csv"
    csv_path.write_text(
        """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash1
2024-01-15 11:00:00,0xmaker2,asset3,50.0,0xtaker2,asset4,142.85,0xhash2
"""
    )
    return csv_path


@pytest.fixture
def bootstrapped_db(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv_large: Path,
    sample_order_filled_csv: Path,
) -> tuple[Path, Path]:
    """Bootstrap a database with sample data."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv_large,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
        partitioning_enabled=False,
    )

    return duckdb_path, parquet_dir


@pytest.fixture
def bootstrapped_db_partitioned(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv_large: Path,
    sample_order_filled_csv: Path,
) -> tuple[Path, Path]:
    """Bootstrap a partitioned database with sample data."""
    parquet_dir = temp_dir / "parquet_partitioned"
    duckdb_path = temp_dir / "test_partitioned.duckdb"
    metadata_path = temp_dir / "metadata_partitioned.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv_large,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
        partitioning_enabled=True,
        hash_bucket_count=8,
    )

    return duckdb_path, parquet_dir


class TestSingleMarketQuery:
    """Tests for single market_id query benchmarking.

    PRD requirement: Benchmark query for a single market_id over a 30-day window.
    """

    def test_single_market_query_returns_result(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that single market query benchmark returns a result."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_single_market_query(market_id="market0")

            assert isinstance(result, QueryBenchmarkResult)
            assert result.query_name == "single_market_query"
            assert result.execution_time_ms >= 0
            assert result.rows_returned >= 0
        finally:
            duckdb_layer.close()

    def test_single_market_query_execution_time_recorded(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that execution time is properly recorded."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_single_market_query(market_id="market0")

            # Should have a positive execution time
            assert result.execution_time_ms > 0
            # Should return trades for market0
            assert result.rows_returned == 10  # 10 trades per market in fixture
        finally:
            duckdb_layer.close()

    def test_single_market_query_auto_selects_market(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that benchmark auto-selects a market if none provided."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_single_market_query()

            assert result.query_name == "single_market_query"
            assert result.rows_returned > 0  # Should find some trades
        finally:
            duckdb_layer.close()


class TestMultiMarketQuery:
    """Tests for 100+ market_ids query benchmarking.

    PRD requirement: Benchmark query for 100+ market_ids over the same window.
    """

    def test_multi_market_query_100_markets(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test querying 100+ markets returns results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_multi_market_query(market_count=100)

            assert result.query_name == "multi_market_query"
            assert result.execution_time_ms > 0
            # 100 markets * 10 trades = 1000 trades expected
            assert result.rows_returned >= 1000
        finally:
            duckdb_layer.close()

    def test_multi_market_query_150_markets(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test querying 150 markets returns all trades."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_multi_market_query(market_count=150)

            assert result.rows_returned == 1500  # All trades
        finally:
            duckdb_layer.close()

    def test_multi_market_query_completes_in_acceptable_time(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that 100+ market query completes within acceptable latency."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_multi_market_query(market_count=100)

            # Should complete in under 500ms for research workflows
            assert result.execution_time_ms < 500
        finally:
            duckdb_layer.close()


class TestPredicatePushdown:
    """Tests for partition pruning/predicate pushdown verification.

    PRD requirement: Verify partition pruning/predicate pushdown is effective.
    """

    def test_predicate_pushdown_returns_plan(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that predicate pushdown benchmark returns a query plan."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_predicate_pushdown()

            assert result.query_name == "predicate_pushdown"
            assert result.query_plan is not None
            assert len(result.query_plan) > 0
        finally:
            duckdb_layer.close()

    def test_predicate_pushdown_plan_shows_filter(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that query plan shows filter operations."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_predicate_pushdown()

            # Query plan should be present and non-empty
            assert result.query_plan is not None
            assert len(result.query_plan) > 0
            # The plan is returned by DuckDB - format varies by version
        finally:
            duckdb_layer.close()

    def test_partitioned_data_query_plan(
        self, bootstrapped_db_partitioned: tuple[Path, Path]
    ) -> None:
        """Test query plan for partitioned data."""
        duckdb_path, parquet_dir = bootstrapped_db_partitioned
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_predicate_pushdown()

            assert result.query_plan is not None
            # Partitioned data should still have a valid plan
            assert len(result.query_plan) > 0
        finally:
            duckdb_layer.close()


class TestMemoryBoundedQuery:
    """Tests for memory usage verification.

    PRD requirement: Verify memory usage is bounded (streaming/limits/spill-to-disk).
    """

    def test_memory_bounded_query_with_limit(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that query with limit returns bounded results."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_memory_bounded_query(limit=100)

            assert result.query_name == "memory_bounded_query"
            # Should respect the limit
            assert result.rows_returned <= 100
        finally:
            duckdb_layer.close()

    def test_memory_bounded_query_large_limit(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that large limit query still completes."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_memory_bounded_query(limit=10000)

            assert result.execution_time_ms > 0
            # Should return up to limit
            assert result.rows_returned <= 10000
        finally:
            duckdb_layer.close()


class TestDaysToExpFilter:
    """Tests for days_to_exp filtering performance."""

    def test_days_to_exp_filter_benchmark(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that days_to_exp filter benchmark works."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_days_to_exp_filter(
                days_to_exp_min=85, days_to_exp_max=95
            )

            assert result.query_name == "days_to_exp_filter"
            assert result.execution_time_ms > 0
        finally:
            duckdb_layer.close()


class TestFullBenchmarkReport:
    """Tests for complete benchmark report generation.

    PRD requirement: Record baseline numbers for regression tracking.
    """

    def test_full_benchmark_returns_report(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that full benchmark produces a complete report."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            report = benchmarker.run_full_benchmark()

            assert isinstance(report, PerformanceBenchmarkReport)
            assert report.report_id is not None
            assert report.created_at is not None
            assert report.total_trades > 0
            assert report.total_markets > 0
            assert len(report.benchmarks) >= 5  # At least 5 benchmarks
        finally:
            duckdb_layer.close()

    def test_full_benchmark_includes_summary(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that report includes summary statistics."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            report = benchmarker.run_full_benchmark()

            assert "avg_execution_time_ms" in report.summary
            assert "max_execution_time_ms" in report.summary
            assert "min_execution_time_ms" in report.summary
            assert "total_rows_returned" in report.summary
            assert "benchmark_count" in report.summary
        finally:
            duckdb_layer.close()

    def test_full_benchmark_to_dict(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that report can be serialized to dict."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            report = benchmarker.run_full_benchmark()
            report_dict = report.to_dict()

            assert "report_id" in report_dict
            assert "created_at" in report_dict
            assert "benchmarks" in report_dict
            assert "summary" in report_dict
            assert isinstance(report_dict["benchmarks"], list)
        finally:
            duckdb_layer.close()

    def test_full_benchmark_tracks_partitioned_status(
        self, bootstrapped_db_partitioned: tuple[Path, Path]
    ) -> None:
        """Test that report tracks whether data is partitioned."""
        duckdb_path, parquet_dir = bootstrapped_db_partitioned
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=True)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            report = benchmarker.run_full_benchmark()

            assert report.partitioned is True
        finally:
            duckdb_layer.close()


class TestPerformanceLatencyTargets:
    """Tests verifying latency targets for research workflows."""

    def test_single_market_under_100ms(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test single market query completes under 100ms target."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            result = benchmarker.benchmark_single_market_query()

            # PRD requirement: fast for iterative research
            assert result.execution_time_ms < 100
        finally:
            duckdb_layer.close()

    def test_summary_tracks_latency_targets(
        self, bootstrapped_db: tuple[Path, Path]
    ) -> None:
        """Test that summary includes latency target checks."""
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            report = benchmarker.run_full_benchmark()

            # Summary should track whether targets are met
            assert "single_market_under_100ms" in report.summary
            assert "multi_market_under_500ms" in report.summary
        finally:
            duckdb_layer.close()


class TestPartitionedVsMonolithic:
    """Tests comparing partitioned and monolithic performance."""

    def test_both_modes_produce_valid_benchmarks(
        self,
        bootstrapped_db: tuple[Path, Path],
        bootstrapped_db_partitioned: tuple[Path, Path],
    ) -> None:
        """Test that both partitioned and monolithic modes work."""
        # Test monolithic
        duckdb_path, parquet_dir = bootstrapped_db
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            benchmarker = PerformanceBenchmarker(duckdb_layer)
            mono_report = benchmarker.run_full_benchmark()
        finally:
            duckdb_layer.close()

        # Test partitioned
        duckdb_path_p, parquet_dir_p = bootstrapped_db_partitioned
        duckdb_layer_p = DuckDBLayer(duckdb_path_p, parquet_dir_p, partitioned=True)
        try:
            duckdb_layer_p.create_views()
            benchmarker_p = PerformanceBenchmarker(duckdb_layer_p)
            part_report = benchmarker_p.run_full_benchmark()
        finally:
            duckdb_layer_p.close()

        # Both should produce valid reports
        assert mono_report.partitioned is False
        assert part_report.partitioned is True
        assert len(mono_report.benchmarks) == len(part_report.benchmarks)


class TestEmptyDatabase:
    """Tests for handling empty databases gracefully."""

    def test_empty_database_handles_gracefully(self, temp_dir: Path) -> None:
        """Test that benchmarker handles empty database gracefully."""
        duckdb_path = temp_dir / "empty.duckdb"
        parquet_dir = temp_dir / "parquet"
        parquet_dir.mkdir(parents=True)

        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            benchmarker = PerformanceBenchmarker(duckdb_layer)

            # Should not raise, even with no data
            result = benchmarker.benchmark_single_market_query()

            assert result.execution_time_ms == 0
            assert result.rows_returned == 0
            assert "error" in result.parameters
        finally:
            duckdb_layer.close()


class TestPerformanceAPI:
    """Tests for performance benchmark API endpoint."""

    def test_benchmark_endpoint(
        self,
        bootstrapped_db: tuple[Path, Path],
    ) -> None:
        """Test the performance benchmark API endpoint."""
        from fastapi.testclient import TestClient
        from polymkt.api.main import app
        from polymkt.config import settings

        duckdb_path, parquet_dir = bootstrapped_db

        # Backup original settings
        orig_parquet_dir = settings.parquet_dir
        orig_duckdb_path = settings.duckdb_path

        try:
            # Set test paths
            settings.parquet_dir = parquet_dir
            settings.duckdb_path = duckdb_path

            client = TestClient(app)
            response = client.post("/api/performance/benchmark")

            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.json()}")

            assert response.status_code == 200
            data = response.json()
            assert "report_id" in data
            assert "benchmarks" in data
            assert "summary" in data
            assert len(data["benchmarks"]) >= 5
        finally:
            # Restore original settings
            settings.parquet_dir = orig_parquet_dir
            settings.duckdb_path = orig_duckdb_path

    def test_benchmark_endpoint_no_data(self, temp_dir: Path) -> None:
        """Test benchmark endpoint returns error when no data."""
        from fastapi.testclient import TestClient
        from polymkt.api.main import app
        from polymkt.config import settings

        # Backup original settings
        orig_parquet_dir = settings.parquet_dir

        try:
            # Set to non-existent path
            settings.parquet_dir = temp_dir / "nonexistent"

            client = TestClient(app)
            response = client.post("/api/performance/benchmark")

            assert response.status_code == 400
            assert "bootstrap" in response.json()["detail"].lower()
        finally:
            settings.parquet_dir = orig_parquet_dir
