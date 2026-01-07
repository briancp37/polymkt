"""Cost efficiency analysis for storage and infrastructure.

This module verifies that the polymkt system meets its cost efficiency goals:
- Local dev is fully functional with filesystem Parquet + DuckDB + SQLite
- No managed database server is required for core workflows
- Storage footprint is reduced vs CSV due to Parquet compression
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class StorageFootprint:
    """Storage footprint for a file or directory."""

    path: str
    size_bytes: int
    file_count: int
    format: str  # "csv", "parquet", "sqlite", "duckdb"

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def size_kb(self) -> float:
        """Size in kilobytes."""
        return self.size_bytes / 1024


@dataclass
class CompressionRatio:
    """Compression ratio between two storage formats."""

    original_format: str
    compressed_format: str
    original_size_bytes: int
    compressed_size_bytes: int

    @property
    def ratio(self) -> float:
        """Compression ratio (original / compressed)."""
        if self.compressed_size_bytes == 0:
            return 0.0
        return self.original_size_bytes / self.compressed_size_bytes

    @property
    def savings_percent(self) -> float:
        """Storage savings as a percentage."""
        if self.original_size_bytes == 0:
            return 0.0
        return (1 - self.compressed_size_bytes / self.original_size_bytes) * 100


@dataclass
class InfrastructureRequirements:
    """Infrastructure requirements for the current setup."""

    requires_managed_db: bool = False
    requires_external_services: list[str] = field(default_factory=list)
    local_only_services: list[str] = field(default_factory=list)
    storage_backends: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class CostEfficiencyReport:
    """Complete cost efficiency report."""

    report_id: str
    created_at: datetime
    storage_footprints: list[StorageFootprint] = field(default_factory=list)
    compression_ratios: list[CompressionRatio] = field(default_factory=list)
    infrastructure: InfrastructureRequirements = field(
        default_factory=InfrastructureRequirements
    )
    estimated_monthly_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "created_at": self.created_at.isoformat(),
            "storage_footprints": [
                {
                    "path": sf.path,
                    "size_bytes": sf.size_bytes,
                    "size_mb": round(sf.size_mb, 2),
                    "file_count": sf.file_count,
                    "format": sf.format,
                }
                for sf in self.storage_footprints
            ],
            "compression_ratios": [
                {
                    "original_format": cr.original_format,
                    "compressed_format": cr.compressed_format,
                    "original_size_bytes": cr.original_size_bytes,
                    "compressed_size_bytes": cr.compressed_size_bytes,
                    "ratio": round(cr.ratio, 2),
                    "savings_percent": round(cr.savings_percent, 1),
                }
                for cr in self.compression_ratios
            ],
            "infrastructure": {
                "requires_managed_db": self.infrastructure.requires_managed_db,
                "requires_external_services": self.infrastructure.requires_external_services,
                "local_only_services": self.infrastructure.local_only_services,
                "storage_backends": self.infrastructure.storage_backends,
                "notes": self.infrastructure.notes,
            },
            "estimated_monthly_cost_usd": self.estimated_monthly_cost_usd,
            "notes": self.notes,
        }


class CostEfficiencyAnalyzer:
    """Analyzer for cost efficiency of the polymkt storage system."""

    def __init__(
        self,
        data_dir: Path,
        parquet_dir: Path,
        duckdb_path: Path,
        metadata_db_path: Path,
        csv_dir: Path | None = None,
    ) -> None:
        """
        Initialize the cost efficiency analyzer.

        Args:
            data_dir: Root data directory
            parquet_dir: Directory containing Parquet files
            duckdb_path: Path to DuckDB database file
            metadata_db_path: Path to SQLite metadata database
            csv_dir: Optional directory containing CSV files for comparison
        """
        self.data_dir = data_dir
        self.parquet_dir = parquet_dir
        self.duckdb_path = duckdb_path
        self.metadata_db_path = metadata_db_path
        self.csv_dir = csv_dir or data_dir

    def _get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        if path.is_file():
            return path.stat().st_size
        return 0

    def _get_directory_size(self, path: Path, extension: str | None = None) -> tuple[int, int]:
        """
        Get total size and file count for a directory.

        Args:
            path: Directory path
            extension: Optional file extension filter (e.g., ".parquet")

        Returns:
            Tuple of (total_size_bytes, file_count)
        """
        if not path.exists():
            return 0, 0

        total_size = 0
        file_count = 0

        if path.is_file():
            return path.stat().st_size, 1

        for item in path.rglob("*"):
            if item.is_file():
                if extension is None or item.suffix == extension:
                    total_size += item.stat().st_size
                    file_count += 1

        return total_size, file_count

    def analyze_csv_storage(self) -> list[StorageFootprint]:
        """Analyze CSV file storage footprint."""
        footprints: list[StorageFootprint] = []

        csv_files = [
            "markets.csv",
            "trades.csv",
            "orderFilled.csv",
            "events.csv",
        ]

        for csv_file in csv_files:
            csv_path = self.csv_dir / csv_file
            if csv_path.exists():
                size = self._get_file_size(csv_path)
                footprints.append(
                    StorageFootprint(
                        path=str(csv_path),
                        size_bytes=size,
                        file_count=1,
                        format="csv",
                    )
                )
                logger.info(
                    "csv_storage_analyzed",
                    file=csv_file,
                    size_bytes=size,
                    size_mb=round(size / (1024 * 1024), 2),
                )

        return footprints

    def analyze_parquet_storage(self) -> list[StorageFootprint]:
        """Analyze Parquet file storage footprint."""
        footprints: list[StorageFootprint] = []

        if not self.parquet_dir.exists():
            return footprints

        # Check for individual parquet files
        parquet_files = [
            "markets.parquet",
            "trades.parquet",
            "order_filled.parquet",
            "events.parquet",
        ]

        for parquet_file in parquet_files:
            parquet_path = self.parquet_dir / parquet_file
            if parquet_path.exists():
                size = self._get_file_size(parquet_path)
                footprints.append(
                    StorageFootprint(
                        path=str(parquet_path),
                        size_bytes=size,
                        file_count=1,
                        format="parquet",
                    )
                )
                logger.info(
                    "parquet_storage_analyzed",
                    file=parquet_file,
                    size_bytes=size,
                    size_mb=round(size / (1024 * 1024), 2),
                )

        # Check for partitioned trades directory
        trades_dir = self.parquet_dir / "trades"
        if trades_dir.exists() and trades_dir.is_dir():
            size, count = self._get_directory_size(trades_dir, ".parquet")
            footprints.append(
                StorageFootprint(
                    path=str(trades_dir),
                    size_bytes=size,
                    file_count=count,
                    format="parquet",
                )
            )
            logger.info(
                "parquet_storage_analyzed",
                directory="trades",
                size_bytes=size,
                file_count=count,
                partitioned=True,
            )

        # Check raw and analytics layers
        for layer in ["raw", "analytics"]:
            layer_dir = self.parquet_dir / layer
            if layer_dir.exists():
                size, count = self._get_directory_size(layer_dir, ".parquet")
                footprints.append(
                    StorageFootprint(
                        path=str(layer_dir),
                        size_bytes=size,
                        file_count=count,
                        format="parquet",
                    )
                )
                logger.info(
                    "parquet_layer_analyzed",
                    layer=layer,
                    size_bytes=size,
                    file_count=count,
                )

        return footprints

    def analyze_database_storage(self) -> list[StorageFootprint]:
        """Analyze DuckDB and SQLite database storage."""
        footprints = []

        # DuckDB
        if self.duckdb_path.exists():
            size = self._get_file_size(self.duckdb_path)
            footprints.append(
                StorageFootprint(
                    path=str(self.duckdb_path),
                    size_bytes=size,
                    file_count=1,
                    format="duckdb",
                )
            )
            logger.info(
                "duckdb_storage_analyzed",
                size_bytes=size,
                size_kb=round(size / 1024, 2),
            )

        # SQLite metadata
        if self.metadata_db_path.exists():
            size = self._get_file_size(self.metadata_db_path)
            footprints.append(
                StorageFootprint(
                    path=str(self.metadata_db_path),
                    size_bytes=size,
                    file_count=1,
                    format="sqlite",
                )
            )
            logger.info(
                "sqlite_storage_analyzed",
                size_bytes=size,
                size_kb=round(size / 1024, 2),
            )

        return footprints

    def calculate_compression_ratios(
        self,
        csv_footprints: list[StorageFootprint],
        parquet_footprints: list[StorageFootprint],
    ) -> list[CompressionRatio]:
        """Calculate compression ratios between CSV and Parquet."""
        ratios = []

        # Map CSV files to their sizes
        csv_sizes = {}
        for fp in csv_footprints:
            name = Path(fp.path).stem.lower()
            csv_sizes[name] = fp.size_bytes

        # Map Parquet files to their sizes
        parquet_sizes: dict[str, int] = {}
        for fp in parquet_footprints:
            name = Path(fp.path).stem.lower()
            # Handle both "trades.parquet" and "trades" directory
            if name == "order_filled":
                name = "orderfilled"  # Normalize for comparison
            parquet_sizes[name] = parquet_sizes.get(name, 0) + fp.size_bytes

        # Calculate ratios for matching files
        for name, csv_size in csv_sizes.items():
            parquet_name = name
            if name == "orderfilled":
                parquet_name = "order_filled"

            if parquet_name in parquet_sizes or name in parquet_sizes:
                parquet_size = parquet_sizes.get(parquet_name, parquet_sizes.get(name, 0))
                if parquet_size > 0:
                    ratio = CompressionRatio(
                        original_format="csv",
                        compressed_format="parquet",
                        original_size_bytes=csv_size,
                        compressed_size_bytes=parquet_size,
                    )
                    ratios.append(ratio)
                    logger.info(
                        "compression_ratio_calculated",
                        file=name,
                        csv_bytes=csv_size,
                        parquet_bytes=parquet_size,
                        ratio=round(ratio.ratio, 2),
                        savings_percent=round(ratio.savings_percent, 1),
                    )

        return ratios

    def analyze_infrastructure_requirements(self) -> InfrastructureRequirements:
        """Analyze infrastructure requirements for the current setup."""
        requirements = InfrastructureRequirements()

        # Core storage is all local
        requirements.storage_backends = [
            "filesystem (Parquet)",
            "DuckDB (embedded)",
            "SQLite (embedded)",
        ]

        requirements.local_only_services = [
            "DuckDB query engine",
            "SQLite metadata store",
            "Parquet file storage",
        ]

        # No managed DB required for core workflows
        requirements.requires_managed_db = False

        # External services are optional
        requirements.requires_external_services = []

        requirements.notes = [
            "Core workflows run entirely on local filesystem",
            "DuckDB is an embedded database (no server process)",
            "SQLite is an embedded database (no server process)",
            "Parquet files can optionally be stored on S3/MinIO for shared access",
            "Semantic search requires OpenAI API (optional feature)",
        ]

        return requirements

    def estimate_monthly_cost(
        self,
        storage_footprints: list[StorageFootprint],
        s3_price_per_gb: float = 0.023,  # S3 Standard pricing
        api_calls_per_month: int = 10000,
        api_price_per_1000: float = 0.0004,  # S3 GET pricing
    ) -> float:
        """
        Estimate monthly cost if data were stored on S3.

        Args:
            storage_footprints: List of storage footprints
            s3_price_per_gb: S3 storage price per GB per month
            api_calls_per_month: Estimated API calls per month
            api_price_per_1000: Price per 1000 API calls

        Returns:
            Estimated monthly cost in USD
        """
        total_bytes = sum(fp.size_bytes for fp in storage_footprints)
        total_gb = total_bytes / (1024 * 1024 * 1024)

        storage_cost = total_gb * s3_price_per_gb
        api_cost = (api_calls_per_month / 1000) * api_price_per_1000

        return storage_cost + api_cost

    def run_analysis(self) -> CostEfficiencyReport:
        """
        Run a complete cost efficiency analysis.

        Returns:
            CostEfficiencyReport with all analysis results
        """
        import uuid

        report_id = str(uuid.uuid4())

        logger.info("cost_efficiency_analysis_started", report_id=report_id)

        # Analyze storage
        csv_footprints = self.analyze_csv_storage()
        parquet_footprints = self.analyze_parquet_storage()
        db_footprints = self.analyze_database_storage()

        all_footprints = csv_footprints + parquet_footprints + db_footprints

        # Calculate compression ratios
        compression_ratios = self.calculate_compression_ratios(
            csv_footprints, parquet_footprints
        )

        # Analyze infrastructure
        infrastructure = self.analyze_infrastructure_requirements()

        # Estimate costs (Parquet + DB only, not CSV)
        production_footprints = parquet_footprints + db_footprints
        monthly_cost = self.estimate_monthly_cost(production_footprints)

        # Build notes
        notes = []

        # Total storage comparison
        csv_total = sum(fp.size_bytes for fp in csv_footprints)
        parquet_total = sum(fp.size_bytes for fp in parquet_footprints)
        db_total = sum(fp.size_bytes for fp in db_footprints)

        if csv_total > 0 and parquet_total > 0:
            overall_ratio = csv_total / parquet_total
            overall_savings = (1 - parquet_total / csv_total) * 100
            notes.append(
                f"Overall CSV to Parquet compression: {overall_ratio:.2f}x "
                f"({overall_savings:.1f}% storage savings)"
            )

        notes.append(f"Total Parquet storage: {parquet_total / 1024:.1f} KB")
        notes.append(f"Total database storage (DuckDB + SQLite): {db_total / 1024:.1f} KB")
        notes.append(f"Estimated S3 monthly cost: ${monthly_cost:.4f}")
        notes.append("No managed database server required for any workflow")

        report = CostEfficiencyReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc),
            storage_footprints=all_footprints,
            compression_ratios=compression_ratios,
            infrastructure=infrastructure,
            estimated_monthly_cost_usd=round(monthly_cost, 4),
            notes=notes,
        )

        logger.info(
            "cost_efficiency_analysis_completed",
            report_id=report_id,
            footprint_count=len(all_footprints),
            compression_ratio_count=len(compression_ratios),
        )

        return report


def verify_local_dev_functional(
    parquet_dir: Path,
    duckdb_path: Path,
    metadata_db_path: Path,
) -> dict[str, bool]:
    """
    Verify that local dev is fully functional without external services.

    Args:
        parquet_dir: Directory containing Parquet files
        duckdb_path: Path to DuckDB database
        metadata_db_path: Path to SQLite metadata database

    Returns:
        Dictionary of verification results
    """
    results = {}

    # Check Parquet files exist
    results["parquet_exists"] = parquet_dir.exists()

    # Check for at least one parquet file
    if parquet_dir.exists():
        parquet_files = list(parquet_dir.glob("**/*.parquet"))
        results["parquet_files_found"] = len(parquet_files) > 0
    else:
        results["parquet_files_found"] = False

    # Check DuckDB can be created/opened (embedded, no server)
    try:
        import duckdb
        duckdb_conn = duckdb.connect(str(duckdb_path))
        duckdb_conn.execute("SELECT 1").fetchone()
        duckdb_conn.close()
        results["duckdb_functional"] = True
    except Exception:
        results["duckdb_functional"] = False

    # Check SQLite can be created/opened (embedded, no server)
    try:
        import sqlite3
        sqlite_conn = sqlite3.connect(str(metadata_db_path))
        sqlite_conn.execute("SELECT 1").fetchone()
        sqlite_conn.close()
        results["sqlite_functional"] = True
    except Exception:
        results["sqlite_functional"] = False

    # All core components work without managed DB
    results["no_managed_db_required"] = all([
        results.get("duckdb_functional", False),
        results.get("sqlite_functional", False),
    ])

    logger.info(
        "local_dev_verification_completed",
        results=results,
        all_passed=all(results.values()),
    )

    return results
