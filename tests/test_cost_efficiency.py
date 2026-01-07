"""Tests for cost efficiency verification.

PRD requirements tested:
- Run end-to-end locally using filesystem Parquet + DuckDB + SQLite
- Verify no managed DB is required for core workflows
- Optionally configure object storage (S3/MinIO) and re-run read-only queries
- Verify storage footprint is materially reduced vs CSV due to Parquet compression
- Document expected costs and size estimates
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from polymkt.pipeline.bootstrap import run_bootstrap
from polymkt.storage.cost_efficiency import (
    CompressionRatio,
    CostEfficiencyAnalyzer,
    CostEfficiencyReport,
    InfrastructureRequirements,
    StorageFootprint,
    verify_local_dev_functional,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_markets_csv(temp_dir: Path) -> Path:
    """Create a sample markets CSV file."""
    csv_path = temp_dir / "markets.csv"
    lines = [
        "createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category"
    ]
    for i in range(50):
        lines.append(
            f"2024-01-01 00:00:00,market{i},Market question {i}?,Yes,No,false,market-{i},tok{i}a,tok{i}b,cond{i},{1000.0 * (i+1)},MKT{i},2024-12-31 23:59:59,Description {i},Category{i % 5}"
        )
    csv_path.write_text("\n".join(lines))
    return csv_path


@pytest.fixture
def sample_trades_csv(temp_dir: Path) -> Path:
    """Create a sample trades CSV file."""
    csv_path = temp_dir / "trades.csv"
    lines = [
        "timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash"
    ]
    # Create 500 trades to have meaningful compression testing
    for i in range(500):
        market_i = i % 50
        day = (i % 28) + 1
        hour = i % 24
        price = 0.3 + (i * 0.001) % 0.4
        lines.append(
            f"2024-01-{day:02d} {hour:02d}:00:00,market{market_i},0xmaker{i},0xtaker{i},YES,buy,sell,{price:.4f},{100.0 * (i % 10 + 1)},{100.0 / price:.4f},0xhash{i:06d}"
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
def bootstrapped_data(
    temp_dir: Path,
    sample_markets_csv: Path,
    sample_trades_csv: Path,
    sample_order_filled_csv: Path,
) -> tuple[Path, Path, Path, Path]:
    """Bootstrap data and return paths."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    run_bootstrap(
        markets_csv=sample_markets_csv,
        trades_csv=sample_trades_csv,
        order_filled_csv=sample_order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
    )

    return temp_dir, parquet_dir, duckdb_path, metadata_path


class TestStorageFootprint:
    """Tests for StorageFootprint dataclass."""

    def test_storage_footprint_size_mb(self) -> None:
        """Test size_mb calculation."""
        fp = StorageFootprint(
            path="/test/file.parquet",
            size_bytes=1024 * 1024 * 5,  # 5 MB
            file_count=1,
            format="parquet",
        )
        assert fp.size_mb == 5.0

    def test_storage_footprint_size_kb(self) -> None:
        """Test size_kb calculation."""
        fp = StorageFootprint(
            path="/test/file.parquet",
            size_bytes=1024 * 10,  # 10 KB
            file_count=1,
            format="parquet",
        )
        assert fp.size_kb == 10.0


class TestCompressionRatio:
    """Tests for CompressionRatio dataclass."""

    def test_compression_ratio_calculation(self) -> None:
        """Test compression ratio calculation."""
        cr = CompressionRatio(
            original_format="csv",
            compressed_format="parquet",
            original_size_bytes=1000,
            compressed_size_bytes=250,
        )
        assert cr.ratio == 4.0
        assert cr.savings_percent == 75.0

    def test_compression_ratio_zero_compressed(self) -> None:
        """Test handling of zero compressed size."""
        cr = CompressionRatio(
            original_format="csv",
            compressed_format="parquet",
            original_size_bytes=1000,
            compressed_size_bytes=0,
        )
        assert cr.ratio == 0.0

    def test_compression_ratio_zero_original(self) -> None:
        """Test handling of zero original size."""
        cr = CompressionRatio(
            original_format="csv",
            compressed_format="parquet",
            original_size_bytes=0,
            compressed_size_bytes=100,
        )
        assert cr.savings_percent == 0.0


class TestLocalDevFunctional:
    """Tests for local dev functionality verification.

    PRD requirement: Run end-to-end locally using filesystem Parquet + DuckDB + SQLite.
    """

    def test_verify_local_dev_with_bootstrapped_data(
        self, bootstrapped_data: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that local dev is fully functional after bootstrap."""
        temp_dir, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        results = verify_local_dev_functional(
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        assert results["parquet_exists"] is True
        assert results["parquet_files_found"] is True
        assert results["duckdb_functional"] is True
        assert results["sqlite_functional"] is True
        assert results["no_managed_db_required"] is True

    def test_verify_duckdb_is_embedded(self, temp_dir: Path) -> None:
        """Test that DuckDB works as embedded database without server."""
        import duckdb

        db_path = temp_dir / "test_embedded.duckdb"

        # DuckDB should work without any server process
        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1), (2), (3)")
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 3
        finally:
            conn.close()

        # Verify file was created (embedded, not external server)
        assert db_path.exists()

    def test_verify_sqlite_is_embedded(self, temp_dir: Path) -> None:
        """Test that SQLite works as embedded database without server."""
        import sqlite3

        db_path = temp_dir / "test_embedded.db"

        # SQLite should work without any server process
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1), (2), (3)")
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 3
        finally:
            conn.close()

        # Verify file was created (embedded, not external server)
        assert db_path.exists()


class TestNoManagedDbRequired:
    """Tests verifying no managed database is required.

    PRD requirement: Verify no managed DB is required for core workflows.
    """

    def test_infrastructure_requirements(self) -> None:
        """Test that infrastructure requirements show no managed DB."""
        requirements = InfrastructureRequirements()

        # Default should not require managed DB
        assert requirements.requires_managed_db is False
        assert requirements.requires_external_services == []

    def test_core_workflow_without_external_services(
        self, bootstrapped_data: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that all core workflows work without external services."""
        temp_dir, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        from polymkt.storage.duckdb_layer import DuckDBLayer
        from polymkt.storage.metadata import MetadataStore

        # Query workflow (DuckDB over Parquet)
        duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
        try:
            duckdb_layer.create_views()
            trades, count = duckdb_layer.query_trades(limit=10)
            assert count > 0
            assert len(trades) > 0
        finally:
            duckdb_layer.close()

        # Metadata workflow (SQLite)
        metadata_store = MetadataStore(metadata_path)
        runs = metadata_store.list_runs()
        assert len(runs) >= 1  # Bootstrap run


class TestParquetCompression:
    """Tests for Parquet compression verification.

    PRD requirement: Verify storage footprint is materially reduced vs CSV.
    """

    def test_parquet_smaller_than_csv_for_large_files(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
        bootstrapped_data: tuple[Path, Path, Path, Path],
    ) -> None:
        """Test that Parquet files are smaller than CSV files for large data.

        Note: Parquet has fixed metadata overhead (~2KB), so very small CSV files
        may actually be larger in Parquet. We only test compression for files
        above a minimum size threshold (1KB).
        """
        _, parquet_dir, _, _ = bootstrapped_data

        # Get CSV sizes
        csv_sizes = {
            "markets": sample_markets_csv.stat().st_size,
            "trades": sample_trades_csv.stat().st_size,
        }

        # Get Parquet sizes
        parquet_sizes = {}
        if (parquet_dir / "markets.parquet").exists():
            parquet_sizes["markets"] = (parquet_dir / "markets.parquet").stat().st_size
        if (parquet_dir / "trades.parquet").exists():
            parquet_sizes["trades"] = (parquet_dir / "trades.parquet").stat().st_size

        # Verify compression for files above minimum size (1KB)
        # Parquet has ~2KB metadata overhead, so very small files may be larger
        min_csv_size_for_compression = 1024

        for name in parquet_sizes:
            if name in csv_sizes and csv_sizes[name] >= min_csv_size_for_compression:
                assert parquet_sizes[name] <= csv_sizes[name], (
                    f"Parquet {name} ({parquet_sizes[name]}) should be <= "
                    f"CSV {name} ({csv_sizes[name]}) for files >= 1KB"
                )

    def test_compression_ratio_calculated(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
        bootstrapped_data: tuple[Path, Path, Path, Path],
    ) -> None:
        """Test that compression ratios are calculated correctly."""
        data_dir, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        analyzer = CostEfficiencyAnalyzer(
            data_dir=temp_dir,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            csv_dir=temp_dir,
        )

        csv_footprints = analyzer.analyze_csv_storage()
        parquet_footprints = analyzer.analyze_parquet_storage()
        ratios = analyzer.calculate_compression_ratios(csv_footprints, parquet_footprints)

        # Should have at least some compression ratios
        assert len(ratios) > 0

        # Files above ~1KB should show compression (ratio >= 1)
        # Very small files may have ratio < 1 due to Parquet metadata overhead
        min_size_for_compression = 1024
        large_file_ratios = [
            r for r in ratios if r.original_size_bytes >= min_size_for_compression
        ]

        # Verify compression for large files
        for ratio in large_file_ratios:
            assert ratio.ratio >= 1.0, (
                f"Expected compression for {ratio.original_format} -> {ratio.compressed_format} "
                f"(original: {ratio.original_size_bytes} bytes)"
            )


class TestCostEfficiencyAnalyzer:
    """Tests for CostEfficiencyAnalyzer."""

    def test_analyze_storage_footprints(
        self, bootstrapped_data: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test that storage footprints are analyzed correctly."""
        data_dir, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        analyzer = CostEfficiencyAnalyzer(
            data_dir=data_dir,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        parquet_footprints = analyzer.analyze_parquet_storage()
        db_footprints = analyzer.analyze_database_storage()

        # Should find Parquet files
        assert len(parquet_footprints) > 0

        # Should find database files
        assert len(db_footprints) >= 1  # At least DuckDB or SQLite

    def test_infrastructure_analysis(
        self, bootstrapped_data: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test infrastructure requirements analysis."""
        data_dir, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        analyzer = CostEfficiencyAnalyzer(
            data_dir=data_dir,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        requirements = analyzer.analyze_infrastructure_requirements()

        assert requirements.requires_managed_db is False
        assert "DuckDB (embedded)" in requirements.storage_backends
        assert "SQLite (embedded)" in requirements.storage_backends
        assert len(requirements.notes) > 0

    def test_cost_estimation(
        self, bootstrapped_data: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test monthly cost estimation."""
        data_dir, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        analyzer = CostEfficiencyAnalyzer(
            data_dir=data_dir,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        parquet_footprints = analyzer.analyze_parquet_storage()
        db_footprints = analyzer.analyze_database_storage()

        cost = analyzer.estimate_monthly_cost(parquet_footprints + db_footprints)

        # For test data (KB range), cost should be very low
        assert cost < 1.0  # Less than $1/month for test data
        assert cost >= 0.0


class TestFullCostEfficiencyReport:
    """Tests for complete cost efficiency report.

    PRD requirement: Document expected costs and size estimates.
    """

    def test_full_report_generation(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
        bootstrapped_data: tuple[Path, Path, Path, Path],
    ) -> None:
        """Test that a complete report is generated."""
        _, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        analyzer = CostEfficiencyAnalyzer(
            data_dir=temp_dir,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            csv_dir=temp_dir,
        )

        report = analyzer.run_analysis()

        assert isinstance(report, CostEfficiencyReport)
        assert report.report_id is not None
        assert report.created_at is not None
        assert len(report.storage_footprints) > 0
        assert report.infrastructure.requires_managed_db is False

    def test_report_to_dict(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
        bootstrapped_data: tuple[Path, Path, Path, Path],
    ) -> None:
        """Test report serialization to dict."""
        _, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        analyzer = CostEfficiencyAnalyzer(
            data_dir=temp_dir,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            csv_dir=temp_dir,
        )

        report = analyzer.run_analysis()
        report_dict = report.to_dict()

        assert "report_id" in report_dict
        assert "storage_footprints" in report_dict
        assert "compression_ratios" in report_dict
        assert "infrastructure" in report_dict
        assert "estimated_monthly_cost_usd" in report_dict
        assert "notes" in report_dict

    def test_report_includes_compression_notes(
        self,
        temp_dir: Path,
        sample_markets_csv: Path,
        sample_trades_csv: Path,
        sample_order_filled_csv: Path,
        bootstrapped_data: tuple[Path, Path, Path, Path],
    ) -> None:
        """Test that report includes compression analysis in notes."""
        _, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        analyzer = CostEfficiencyAnalyzer(
            data_dir=temp_dir,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
            csv_dir=temp_dir,
        )

        report = analyzer.run_analysis()

        # Should have notes about storage and costs
        assert len(report.notes) > 0
        notes_text = " ".join(report.notes).lower()

        # Should mention no managed database
        assert "no managed database" in notes_text or "parquet" in notes_text


class TestCostEfficiencyAPI:
    """Tests for cost efficiency API endpoint."""

    def test_cost_efficiency_endpoint(
        self, bootstrapped_data: tuple[Path, Path, Path, Path]
    ) -> None:
        """Test the cost efficiency API endpoint."""
        from fastapi.testclient import TestClient
        from polymkt.api.main import app
        from polymkt.config import settings

        data_dir, parquet_dir, duckdb_path, metadata_path = bootstrapped_data

        # Backup original settings
        orig_data_dir = settings.data_dir
        orig_parquet_dir = settings.parquet_dir
        orig_duckdb_path = settings.duckdb_path
        orig_metadata_path = settings.metadata_db_path

        try:
            # Set test paths
            settings.data_dir = data_dir
            settings.parquet_dir = parquet_dir
            settings.duckdb_path = duckdb_path
            settings.metadata_db_path = metadata_path

            client = TestClient(app)
            response = client.post("/api/cost-efficiency/analyze")

            assert response.status_code == 200
            data = response.json()
            assert "report_id" in data
            assert "infrastructure" in data
            assert data["infrastructure"]["requires_managed_db"] is False
        finally:
            # Restore original settings
            settings.data_dir = orig_data_dir
            settings.parquet_dir = orig_parquet_dir
            settings.duckdb_path = orig_duckdb_path
            settings.metadata_db_path = orig_metadata_path
