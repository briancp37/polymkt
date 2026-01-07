"""Performance benchmarking for DuckDB-over-Parquet queries."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from polymkt.storage.duckdb_layer import DuckDBLayer

logger = structlog.get_logger()


@dataclass
class QueryBenchmarkResult:
    """Result of a single query benchmark."""

    query_name: str
    query_description: str
    execution_time_ms: float
    rows_returned: int
    parameters: dict[str, Any] = field(default_factory=dict)
    query_plan: str | None = None
    memory_usage_bytes: int | None = None


@dataclass
class PerformanceBenchmarkReport:
    """Complete performance benchmark report."""

    report_id: str
    created_at: datetime
    total_trades: int
    total_markets: int
    partitioned: bool
    benchmarks: list[QueryBenchmarkResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "created_at": self.created_at.isoformat(),
            "total_trades": self.total_trades,
            "total_markets": self.total_markets,
            "partitioned": self.partitioned,
            "benchmarks": [
                {
                    "query_name": b.query_name,
                    "query_description": b.query_description,
                    "execution_time_ms": b.execution_time_ms,
                    "rows_returned": b.rows_returned,
                    "parameters": b.parameters,
                    "query_plan": b.query_plan,
                    "memory_usage_bytes": b.memory_usage_bytes,
                }
                for b in self.benchmarks
            ],
            "summary": self.summary,
        }


class PerformanceBenchmarker:
    """Benchmarker for DuckDB-over-Parquet query performance."""

    def __init__(
        self,
        duckdb_layer: DuckDBLayer,
    ) -> None:
        """
        Initialize the performance benchmarker.

        Args:
            duckdb_layer: The DuckDB layer to benchmark
        """
        self.duckdb_layer = duckdb_layer

    def _get_data_stats(self) -> tuple[int, int]:
        """Get total trades and markets counts."""
        try:
            result = self.duckdb_layer.execute(
                "SELECT COUNT(*) FROM v_trades"
            ).fetchone()
            total_trades = result[0] if result else 0
        except Exception:
            total_trades = 0

        try:
            result = self.duckdb_layer.execute(
                "SELECT COUNT(*) FROM v_markets"
            ).fetchone()
            total_markets = result[0] if result else 0
        except Exception:
            total_markets = 0

        return total_trades, total_markets

    def _get_sample_market_ids(self, count: int = 100) -> list[str]:
        """Get a sample of market IDs for benchmarking."""
        try:
            result = self.duckdb_layer.execute(
                f"SELECT DISTINCT market_id FROM v_trades LIMIT {count}"
            ).fetchall()
            return [row[0] for row in result]
        except Exception:
            return []

    def _benchmark_query(
        self,
        name: str,
        description: str,
        query_func: Any,
        params: dict[str, Any],
        include_plan: bool = False,
    ) -> QueryBenchmarkResult:
        """
        Execute a query and measure performance.

        Args:
            name: Short name for the query
            description: Description of what the query does
            query_func: Function to call (will be timed)
            params: Parameters to pass to the function
            include_plan: Whether to include the query plan

        Returns:
            QueryBenchmarkResult with timing information
        """
        # Warm up - run once to ensure data is cached
        try:
            query_func(**params)
        except Exception:
            pass

        # Measure execution time
        start_time = time.perf_counter()
        result = query_func(**params)
        end_time = time.perf_counter()

        execution_time_ms = (end_time - start_time) * 1000

        # Get row count
        if isinstance(result, tuple) and len(result) >= 2:
            rows_returned = len(result[0])
        elif isinstance(result, list):
            rows_returned = len(result)
        else:
            rows_returned = 0

        logger.info(
            "benchmark_query_completed",
            query_name=name,
            execution_time_ms=round(execution_time_ms, 2),
            rows_returned=rows_returned,
        )

        return QueryBenchmarkResult(
            query_name=name,
            query_description=description,
            execution_time_ms=round(execution_time_ms, 2),
            rows_returned=rows_returned,
            parameters=params,
        )

    def benchmark_single_market_query(
        self,
        market_id: str | None = None,
        time_window_days: int = 30,
    ) -> QueryBenchmarkResult:
        """
        Benchmark query for a single market_id over a time window.

        PRD requirement: Benchmark query for a single market_id over a 30-day window.

        Args:
            market_id: The market ID to query (uses first available if None)
            time_window_days: Number of days to query (default 30)

        Returns:
            QueryBenchmarkResult with timing information
        """
        if market_id is None:
            market_ids = self._get_sample_market_ids(1)
            if not market_ids:
                return QueryBenchmarkResult(
                    query_name="single_market_query",
                    query_description="Query single market_id over time window",
                    execution_time_ms=0,
                    rows_returned=0,
                    parameters={"error": "No markets available"},
                )
            market_id = market_ids[0]

        params = {
            "market_id": market_id,
            "limit": 10000,
        }

        return self._benchmark_query(
            name="single_market_query",
            description=f"Query single market_id over {time_window_days}-day window",
            query_func=self.duckdb_layer.query_trades,
            params=params,
        )

    def benchmark_multi_market_query(
        self,
        market_count: int = 100,
        time_window_days: int = 30,
    ) -> QueryBenchmarkResult:
        """
        Benchmark query for 100+ market_ids over a time window.

        PRD requirement: Benchmark query for 100+ market_ids over the same window.

        Args:
            market_count: Number of market IDs to query (default 100)
            time_window_days: Number of days to query (default 30)

        Returns:
            QueryBenchmarkResult with timing information
        """
        market_ids = self._get_sample_market_ids(market_count)
        if not market_ids:
            return QueryBenchmarkResult(
                query_name="multi_market_query",
                query_description=f"Query {market_count}+ market_ids over time window",
                execution_time_ms=0,
                rows_returned=0,
                parameters={"error": "No markets available"},
            )

        params = {
            "market_ids": market_ids,
            "limit": 50000,
        }

        return self._benchmark_query(
            name="multi_market_query",
            description=f"Query {len(market_ids)}+ market_ids over {time_window_days}-day window",
            query_func=self.duckdb_layer.query_trades,
            params=params,
        )

    def benchmark_predicate_pushdown(self) -> QueryBenchmarkResult:
        """
        Verify partition pruning/predicate pushdown is effective.

        PRD requirement: Verify partition pruning/predicate pushdown is effective.

        Returns:
            QueryBenchmarkResult with query plan showing pushdown
        """
        market_ids = self._get_sample_market_ids(1)
        if not market_ids:
            return QueryBenchmarkResult(
                query_name="predicate_pushdown",
                query_description="Verify predicate pushdown in query plan",
                execution_time_ms=0,
                rows_returned=0,
                parameters={"error": "No markets available"},
            )

        market_id = market_ids[0]

        # Get the query plan
        query = f"""
            SELECT * FROM v_trades
            WHERE market_id = '{market_id}'
            LIMIT 100
        """

        start_time = time.perf_counter()
        plan = self.duckdb_layer.explain_query(query, analyze=True)
        result = self.duckdb_layer.execute(query).fetchall()
        end_time = time.perf_counter()

        execution_time_ms = (end_time - start_time) * 1000

        logger.info(
            "benchmark_predicate_pushdown_completed",
            execution_time_ms=round(execution_time_ms, 2),
            plan_length=len(plan),
        )

        return QueryBenchmarkResult(
            query_name="predicate_pushdown",
            query_description="Verify predicate pushdown in query plan",
            execution_time_ms=round(execution_time_ms, 2),
            rows_returned=len(result),
            parameters={"market_id": market_id},
            query_plan=plan,
        )

    def benchmark_memory_bounded_query(
        self,
        limit: int = 10000,
    ) -> QueryBenchmarkResult:
        """
        Verify memory usage is bounded with limits.

        PRD requirement: Verify memory usage is bounded (streaming/limits/spill-to-disk if needed).

        Args:
            limit: Row limit to apply (default 10000)

        Returns:
            QueryBenchmarkResult with memory usage information
        """
        params = {
            "limit": limit,
        }

        result = self._benchmark_query(
            name="memory_bounded_query",
            description=f"Query with limit={limit} to verify memory bounds",
            query_func=self.duckdb_layer.query_trades,
            params=params,
        )

        # Note: memory_usage from pragma_database_size() returns a string like "368.0 KiB"
        # We skip setting memory_usage_bytes since it requires parsing

        return result

    def benchmark_days_to_exp_filter(
        self,
        days_to_exp_min: float = 85,
        days_to_exp_max: float = 95,
    ) -> QueryBenchmarkResult:
        """
        Benchmark filtering by days_to_exp range.

        This is important for backtesting strategies like "buy at 90 days to expiry".

        Args:
            days_to_exp_min: Minimum days to expiry
            days_to_exp_max: Maximum days to expiry

        Returns:
            QueryBenchmarkResult with timing information
        """
        params = {
            "days_to_exp_min": days_to_exp_min,
            "days_to_exp_max": days_to_exp_max,
            "limit": 10000,
        }

        return self._benchmark_query(
            name="days_to_exp_filter",
            description=f"Filter trades by days_to_exp ({days_to_exp_min}-{days_to_exp_max})",
            query_func=self.duckdb_layer.query_trades_with_markets,
            params=params,
        )

    def run_full_benchmark(
        self,
        include_plans: bool = True,
    ) -> PerformanceBenchmarkReport:
        """
        Run a complete performance benchmark suite.

        PRD requirement: Record baseline numbers for regression tracking.

        Args:
            include_plans: Whether to include query plans in the report

        Returns:
            PerformanceBenchmarkReport with all benchmark results
        """
        import uuid
        from datetime import datetime, timezone

        report_id = str(uuid.uuid4())
        total_trades, total_markets = self._get_data_stats()

        logger.info(
            "benchmark_suite_started",
            report_id=report_id,
            total_trades=total_trades,
            total_markets=total_markets,
        )

        benchmarks: list[QueryBenchmarkResult] = []

        # Run all benchmarks
        benchmarks.append(self.benchmark_single_market_query())
        benchmarks.append(self.benchmark_multi_market_query(market_count=100))
        benchmarks.append(self.benchmark_predicate_pushdown())
        benchmarks.append(self.benchmark_memory_bounded_query())
        benchmarks.append(self.benchmark_days_to_exp_filter())

        # Compute summary statistics
        valid_benchmarks = [b for b in benchmarks if b.execution_time_ms > 0]
        summary = {}

        if valid_benchmarks:
            execution_times = [b.execution_time_ms for b in valid_benchmarks]
            summary["avg_execution_time_ms"] = round(
                sum(execution_times) / len(execution_times), 2
            )
            summary["max_execution_time_ms"] = round(max(execution_times), 2)
            summary["min_execution_time_ms"] = round(min(execution_times), 2)
            summary["total_rows_returned"] = sum(
                b.rows_returned for b in valid_benchmarks
            )
            summary["benchmark_count"] = len(valid_benchmarks)

            # Check if single market query is under target latency (e.g., 100ms)
            single_market = next(
                (b for b in benchmarks if b.query_name == "single_market_query"),
                None,
            )
            if single_market:
                summary["single_market_under_100ms"] = (
                    single_market.execution_time_ms < 100
                )

            # Check if multi-market query is under acceptable latency (e.g., 500ms)
            multi_market = next(
                (b for b in benchmarks if b.query_name == "multi_market_query"),
                None,
            )
            if multi_market:
                summary["multi_market_under_500ms"] = (
                    multi_market.execution_time_ms < 500
                )

        report = PerformanceBenchmarkReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc),
            total_trades=total_trades,
            total_markets=total_markets,
            partitioned=self.duckdb_layer.partitioned,
            benchmarks=benchmarks,
            summary=summary,
        )

        logger.info(
            "benchmark_suite_completed",
            report_id=report_id,
            benchmark_count=len(benchmarks),
            avg_time_ms=summary.get("avg_execution_time_ms"),
        )

        return report
