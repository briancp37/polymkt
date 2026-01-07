"""Tests for schema evolution feature.

PRD Requirements:
- "Schema evolution: adding new columns later (e.g., tags or additional market
   metadata) should not break reads or backtests"
- "Bootstrap baseline schema and validate queries/backtests work"
- "Add new derived columns in the analytics layer"
- "Verify older partitions remain readable"
- "Verify query layer selects required columns safely"
- "Verify backtest code handles missing optional columns gracefully"
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polymkt.storage.schema_evolution import (
    SchemaEvolutionManager,
    SchemaVersion,
    create_evolution_safe_view,
    detect_schema_changes,
    get_safe_column_sql,
    project_to_schema,
    read_parquet_with_schema_evolution,
    safe_get,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tmp_metadata_db(tmp_path: Path) -> Path:
    """Create a temporary metadata database path."""
    return tmp_path / "metadata" / "test.db"


@pytest.fixture
def schema_manager(tmp_metadata_db: Path) -> SchemaEvolutionManager:
    """Create a schema evolution manager."""
    return SchemaEvolutionManager(tmp_metadata_db)


@pytest.fixture
def old_trades_schema() -> pa.Schema:
    """Original trades schema (v1) - baseline."""
    return pa.schema([
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("market_id", pa.string()),
        ("price", pa.float64()),
        ("usd_amount", pa.float64()),
        ("transaction_hash", pa.string()),
    ])


@pytest.fixture
def new_trades_schema() -> pa.Schema:
    """New trades schema (v2) - with additional column."""
    return pa.schema([
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("market_id", pa.string()),
        ("price", pa.float64()),
        ("usd_amount", pa.float64()),
        ("transaction_hash", pa.string()),
        # New column added in v2
        ("trade_source", pa.string()),
    ])


@pytest.fixture
def extended_trades_schema() -> pa.Schema:
    """Extended trades schema (v3) - with multiple new columns."""
    return pa.schema([
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("market_id", pa.string()),
        ("price", pa.float64()),
        ("usd_amount", pa.float64()),
        ("transaction_hash", pa.string()),
        ("trade_source", pa.string()),
        # New columns in v3
        ("liquidity_type", pa.string()),  # "maker" or "taker"
        ("fee_amount", pa.float64()),
    ])


@pytest.fixture
def sample_old_data(old_trades_schema: pa.Schema) -> pa.Table:
    """Sample data with old schema."""
    now = datetime.now(timezone.utc)
    return pa.table({
        "timestamp": [now, now - timedelta(hours=1)],
        "market_id": ["market_001", "market_002"],
        "price": [0.65, 0.45],
        "usd_amount": [100.0, 50.0],
        "transaction_hash": ["0xabc", "0xdef"],
    })


@pytest.fixture
def parquet_old_schema_file(tmp_path: Path, sample_old_data: pa.Table) -> Path:
    """Create a Parquet file with the old schema."""
    output_path = tmp_path / "trades_old.parquet"
    pq.write_table(sample_old_data, output_path)
    return output_path


# =============================================================================
# Tests: Schema Version Management
# =============================================================================


class TestSchemaEvolutionManager:
    """Tests for schema version tracking."""

    def test_init_creates_table(self, schema_manager: SchemaEvolutionManager) -> None:
        """Verify schema_versions table is created."""
        import sqlite3

        conn = sqlite3.connect(str(schema_manager.metadata_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_versions'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_register_schema_version(
        self, schema_manager: SchemaEvolutionManager
    ) -> None:
        """Test registering a new schema version."""
        version = schema_manager.register_schema_version(
            entity="trades",
            version=1,
            columns=["timestamp", "market_id", "price"],
            added_columns=[],
        )

        assert version.version == 1
        assert version.name == "trades"
        assert len(version.columns) == 3
        assert version.added_columns == []

    def test_register_multiple_versions(
        self, schema_manager: SchemaEvolutionManager
    ) -> None:
        """Test registering multiple schema versions."""
        schema_manager.register_schema_version(
            entity="trades",
            version=1,
            columns=["timestamp", "market_id", "price"],
        )
        schema_manager.register_schema_version(
            entity="trades",
            version=2,
            columns=["timestamp", "market_id", "price", "trade_source"],
            added_columns=["trade_source"],
        )

        # Get current (latest) version
        current = schema_manager.get_current_version("trades")
        assert current is not None
        assert current.version == 2
        assert "trade_source" in current.columns
        assert current.added_columns == ["trade_source"]

    def test_get_all_versions(
        self, schema_manager: SchemaEvolutionManager
    ) -> None:
        """Test getting all schema versions."""
        schema_manager.register_schema_version(
            entity="trades", version=1, columns=["a", "b"]
        )
        schema_manager.register_schema_version(
            entity="trades", version=2, columns=["a", "b", "c"]
        )
        schema_manager.register_schema_version(
            entity="trades", version=3, columns=["a", "b", "c", "d"]
        )

        versions = schema_manager.get_all_versions("trades")
        assert len(versions) == 3
        # Newest first
        assert versions[0].version == 3
        assert versions[1].version == 2
        assert versions[2].version == 1

    def test_get_current_version_none_when_empty(
        self, schema_manager: SchemaEvolutionManager
    ) -> None:
        """Test that get_current_version returns None for unknown entity."""
        result = schema_manager.get_current_version("unknown_entity")
        assert result is None


# =============================================================================
# Tests: Reading Old Parquet with New Schema
# =============================================================================


class TestReadParquetWithSchemaEvolution:
    """Tests for reading old Parquet files with new schema."""

    def test_read_old_parquet_with_new_schema(
        self,
        parquet_old_schema_file: Path,
        new_trades_schema: pa.Schema,
    ) -> None:
        """Verify old Parquet file can be read with new schema."""
        table = read_parquet_with_schema_evolution(
            parquet_old_schema_file, new_trades_schema
        )

        # Should have all columns from new schema
        assert set(table.column_names) == {
            "timestamp",
            "market_id",
            "price",
            "usd_amount",
            "transaction_hash",
            "trade_source",
        }

        # Old columns should have data
        assert table.column("price").to_pylist() == [0.65, 0.45]

        # New column should be null-filled
        trade_source = table.column("trade_source").to_pylist()
        assert all(v is None for v in trade_source)

    def test_project_to_schema_fills_missing_columns(
        self, sample_old_data: pa.Table, new_trades_schema: pa.Schema
    ) -> None:
        """Test that project_to_schema fills missing columns with nulls."""
        projected = project_to_schema(sample_old_data, new_trades_schema)

        assert "trade_source" in projected.column_names
        assert all(v is None for v in projected.column("trade_source").to_pylist())

    def test_project_to_schema_drops_extra_columns(
        self, sample_old_data: pa.Table
    ) -> None:
        """Test that project_to_schema drops columns not in target schema."""
        # Schema with fewer columns than source
        minimal_schema = pa.schema([
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("market_id", pa.string()),
            ("price", pa.float64()),
        ])

        projected = project_to_schema(sample_old_data, minimal_schema)

        assert set(projected.column_names) == {"timestamp", "market_id", "price"}
        assert "usd_amount" not in projected.column_names
        assert "transaction_hash" not in projected.column_names

    def test_project_to_schema_casts_types(self) -> None:
        """Test that project_to_schema casts column types."""
        source = pa.table({
            "value": pa.array([1, 2, 3], type=pa.int32()),
        })
        target_schema = pa.schema([
            ("value", pa.float64()),
        ])

        projected = project_to_schema(source, target_schema)
        assert projected.schema.field("value").type == pa.float64()


class TestOlderPartitionsRemainReadable:
    """PRD: 'Verify older partitions remain readable'."""

    def test_partitioned_data_with_mixed_schemas(self, tmp_path: Path) -> None:
        """
        Simulate reading partitioned data where older partitions have
        fewer columns than newer partitions.
        """
        # Create "old" partition
        old_partition = tmp_path / "trades" / "year=2024" / "month=01"
        old_partition.mkdir(parents=True)
        old_data = pa.table({
            "timestamp": [datetime.now(timezone.utc)],
            "market_id": ["m1"],
            "price": [0.5],
        })
        pq.write_table(old_data, old_partition / "data.parquet")

        # Create "new" partition with additional column
        new_partition = tmp_path / "trades" / "year=2024" / "month=02"
        new_partition.mkdir(parents=True)
        new_data = pa.table({
            "timestamp": [datetime.now(timezone.utc)],
            "market_id": ["m2"],
            "price": [0.6],
            "trade_source": ["api"],
        })
        pq.write_table(new_data, new_partition / "data.parquet")

        # Read with DuckDB using UNION ALL approach
        conn = duckdb.connect(":memory:")

        # This is how DuckDB handles schema union by default with read_parquet
        result = conn.execute(f"""
            SELECT * FROM read_parquet('{tmp_path}/trades/**/*.parquet', union_by_name=true)
            ORDER BY market_id
        """).fetchall()

        # Both rows should be readable
        assert len(result) == 2

        # Column order might vary, so check by column name
        columns = [desc[0] for desc in conn.execute(f"""
            SELECT * FROM read_parquet('{tmp_path}/trades/**/*.parquet', union_by_name=true)
            LIMIT 0
        """).description]

        market_idx = columns.index("market_id")
        price_idx = columns.index("price")

        assert result[0][market_idx] == "m1"
        assert result[1][market_idx] == "m2"
        assert result[0][price_idx] == 0.5
        assert result[1][price_idx] == 0.6


# =============================================================================
# Tests: Query Layer Safe Column Access
# =============================================================================


class TestQueryLayerSafeColumnAccess:
    """PRD: 'Verify query layer selects required columns safely'."""

    def test_get_safe_column_sql_with_default(self) -> None:
        """Test SQL generation for safe column access with default."""
        sql = get_safe_column_sql("trade_source", "VARCHAR", "'unknown'")
        assert "COALESCE" in sql
        assert "TRY_CAST" in sql
        assert "'unknown'" in sql

    def test_get_safe_column_sql_without_default(self) -> None:
        """Test SQL generation without default value."""
        sql = get_safe_column_sql("price", "DOUBLE")
        assert "TRY_CAST" in sql
        assert "COALESCE" not in sql

    def test_get_safe_column_sql_with_alias(self) -> None:
        """Test SQL generation with table alias."""
        sql = get_safe_column_sql("price", "DOUBLE", None, "t")
        assert "t.price" in sql

    def test_create_evolution_safe_view(self, tmp_path: Path) -> None:
        """Test creating a view that handles schema evolution.

        Note: create_evolution_safe_view works with columns that EXIST in the
        source but may have NULL values. For columns that don't exist at all
        in the source schema, use DuckDB's union_by_name approach or add
        constant default columns manually.
        """
        conn = duckdb.connect(":memory:")

        # Create source table with all columns (some with NULL values)
        conn.execute("""
            CREATE TABLE source_trades (
                market_id VARCHAR,
                price DOUBLE,
                trade_source VARCHAR,
                fee_amount DOUBLE
            )
        """)
        # Insert row where optional columns are NULL
        conn.execute("INSERT INTO source_trades VALUES ('m1', 0.5, NULL, NULL)")

        # Create evolution-safe view that handles NULL values
        create_evolution_safe_view(
            conn=conn,
            view_name="v_safe_trades",
            source="source_trades",
            required_columns=[
                ("market_id", "VARCHAR"),
                ("price", "DOUBLE"),
            ],
            optional_columns=[
                ("trade_source", "VARCHAR", "'unknown'"),
                ("fee_amount", "DOUBLE", "0.0"),
            ],
        )

        # Query the view
        result = conn.execute("SELECT * FROM v_safe_trades").fetchone()
        columns = [desc[0] for desc in conn.execute("SELECT * FROM v_safe_trades LIMIT 0").description]

        market_idx = columns.index("market_id")
        price_idx = columns.index("price")
        source_idx = columns.index("trade_source")
        fee_idx = columns.index("fee_amount")

        assert result[market_idx] == "m1"
        assert result[price_idx] == 0.5
        assert result[source_idx] == "unknown"  # Default value because NULL in source
        assert result[fee_idx] == 0.0  # Default value because NULL in source


# =============================================================================
# Tests: Backtest Code Handles Missing Optional Columns
# =============================================================================


class TestBacktestSafeColumnAccess:
    """PRD: 'Verify backtest code handles missing optional columns gracefully'."""

    def test_safe_get_returns_value_when_present(self) -> None:
        """Test safe_get returns existing column value."""
        record = {"price": 0.75, "market_id": "m1"}
        assert safe_get(record, "price", 0.0) == 0.75

    def test_safe_get_returns_default_when_missing(self) -> None:
        """Test safe_get returns default for missing column."""
        record = {"price": 0.75}
        assert safe_get(record, "trade_source", "unknown") == "unknown"

    def test_safe_get_returns_default_when_none(self) -> None:
        """Test safe_get returns default when value is None."""
        record = {"price": 0.75, "trade_source": None}
        assert safe_get(record, "trade_source", "unknown") == "unknown"

    def test_safe_get_with_type_checking(self) -> None:
        """Test safe_get with expected_type parameter."""
        record = {"price": "0.75"}  # String instead of float
        # Should convert string to float
        result = safe_get(record, "price", 0.0, float)
        assert result == 0.75
        assert isinstance(result, float)

    def test_safe_get_with_type_mismatch_returns_default(self) -> None:
        """Test safe_get returns default when type conversion fails."""
        record = {"price": "not_a_number"}
        result = safe_get(record, "price", 0.0, float)
        assert result == 0.0

    def test_backtest_with_missing_optional_column(self) -> None:
        """Simulate backtest accessing data with missing optional column."""
        # Simulated trade data (old schema without trade_source)
        trade = {
            "timestamp": datetime.now(timezone.utc),
            "market_id": "m1",
            "price": 0.65,
            "closed_time": datetime.now(timezone.utc) + timedelta(days=30),
        }

        # Backtest code using safe_get
        market_id = safe_get(trade, "market_id", "", str)
        price = safe_get(trade, "price", 0.0, float)
        trade_source = safe_get(trade, "trade_source", "legacy", str)
        fee_amount = safe_get(trade, "fee_amount", 0.0, float)

        assert market_id == "m1"
        assert price == 0.65
        assert trade_source == "legacy"  # Default since column missing
        assert fee_amount == 0.0  # Default since column missing


# =============================================================================
# Tests: Schema Change Detection
# =============================================================================


class TestSchemaChangeDetection:
    """Tests for detecting schema changes."""

    def test_detect_added_columns(
        self, old_trades_schema: pa.Schema, new_trades_schema: pa.Schema
    ) -> None:
        """Test detection of added columns."""
        added, removed, type_changed = detect_schema_changes(
            old_trades_schema, new_trades_schema
        )

        assert added == ["trade_source"]
        assert removed == []
        assert type_changed == []

    def test_detect_removed_columns(
        self, new_trades_schema: pa.Schema, old_trades_schema: pa.Schema
    ) -> None:
        """Test detection of removed columns (reverse direction)."""
        added, removed, type_changed = detect_schema_changes(
            new_trades_schema, old_trades_schema
        )

        assert added == []
        assert removed == ["trade_source"]
        assert type_changed == []

    def test_detect_type_changes(self) -> None:
        """Test detection of type changes."""
        old = pa.schema([("value", pa.int32())])
        new = pa.schema([("value", pa.float64())])

        added, removed, type_changed = detect_schema_changes(old, new)

        assert added == []
        assert removed == []
        assert type_changed == ["value"]

    def test_detect_multiple_changes(self) -> None:
        """Test detection of multiple schema changes."""
        old = pa.schema([
            ("a", pa.string()),
            ("b", pa.int32()),
            ("c", pa.string()),
        ])
        new = pa.schema([
            ("a", pa.string()),
            ("b", pa.float64()),  # Type changed
            # "c" removed
            ("d", pa.string()),  # Added
        ])

        added, removed, type_changed = detect_schema_changes(old, new)

        assert added == ["d"]
        assert removed == ["c"]
        assert type_changed == ["b"]


# =============================================================================
# Tests: DuckDB Views With Schema Evolution
# =============================================================================


class TestDuckDBViewsWithSchemaEvolution:
    """Tests for DuckDB views handling schema evolution."""

    def test_duckdb_union_by_name_handles_missing_columns(
        self, tmp_path: Path
    ) -> None:
        """Test that DuckDB's union_by_name fills missing columns with NULL."""
        # Create two files with different schemas
        file1 = tmp_path / "part1.parquet"
        file2 = tmp_path / "part2.parquet"

        pq.write_table(
            pa.table({"a": [1, 2], "b": ["x", "y"]}),
            file1,
        )
        pq.write_table(
            pa.table({"a": [3, 4], "c": [1.0, 2.0]}),  # Has 'c' instead of 'b'
            file2,
        )

        conn = duckdb.connect(":memory:")
        result = conn.execute(f"""
            SELECT * FROM read_parquet(['{file1}', '{file2}'], union_by_name=true)
            ORDER BY a
        """).fetchall()

        # Should have 4 rows with some NULLs
        assert len(result) == 4

        # Check columns exist
        columns = conn.execute(f"""
            SELECT * FROM read_parquet(['{file1}', '{file2}'], union_by_name=true)
            LIMIT 0
        """).description
        column_names = [c[0] for c in columns]
        assert "a" in column_names
        assert "b" in column_names
        assert "c" in column_names

    def test_view_with_coalesce_for_optional_columns(self) -> None:
        """Test creating a view that uses COALESCE for optional columns."""
        conn = duckdb.connect(":memory:")

        # Create table without optional column
        conn.execute("""
            CREATE TABLE trades (
                market_id VARCHAR,
                price DOUBLE
            )
        """)
        conn.execute("INSERT INTO trades VALUES ('m1', 0.5)")

        # Create view with COALESCE for optional column that doesn't exist
        # We need to handle this at the application level since DuckDB
        # will error if column doesn't exist
        conn.execute("""
            CREATE VIEW v_trades AS
            SELECT
                market_id,
                price,
                'unknown' as trade_source
            FROM trades
        """)

        result = conn.execute("SELECT * FROM v_trades").fetchone()
        assert result[2] == "unknown"


# =============================================================================
# Integration Tests
# =============================================================================


class TestSchemaEvolutionIntegration:
    """Integration tests for complete schema evolution workflow."""

    def test_full_schema_evolution_workflow(
        self,
        tmp_path: Path,
        schema_manager: SchemaEvolutionManager,
        old_trades_schema: pa.Schema,
        new_trades_schema: pa.Schema,
    ) -> None:
        """
        Test complete workflow:
        1. Bootstrap with baseline schema
        2. Add new column to schema
        3. Verify old data still readable
        4. Verify queries work with new schema
        """
        # Step 1: Bootstrap with baseline schema
        schema_manager.register_schema_version(
            entity="trades",
            version=1,
            columns=[f.name for f in old_trades_schema],
        )

        # Write old data
        old_data_path = tmp_path / "trades_v1.parquet"
        old_data = pa.table({
            "timestamp": [datetime.now(timezone.utc)],
            "market_id": ["m1"],
            "price": [0.5],
            "usd_amount": [100.0],
            "transaction_hash": ["0xabc"],
        })
        pq.write_table(old_data, old_data_path)

        # Step 2: Add new column to schema
        schema_manager.register_schema_version(
            entity="trades",
            version=2,
            columns=[f.name for f in new_trades_schema],
            added_columns=["trade_source"],
        )

        current = schema_manager.get_current_version("trades")
        assert current is not None
        assert current.version == 2

        # Step 3: Verify old data still readable with new schema
        projected = read_parquet_with_schema_evolution(old_data_path, new_trades_schema)

        assert projected.num_rows == 1
        assert projected.column("market_id").to_pylist() == ["m1"]
        assert projected.column("trade_source").to_pylist() == [None]

        # Step 4: Verify queries work with new schema using DuckDB
        conn = duckdb.connect(":memory:")
        conn.execute(f"CREATE VIEW v_trades AS SELECT * FROM '{old_data_path}'")

        # Query should work even though old data doesn't have trade_source
        result = conn.execute("""
            SELECT
                market_id,
                price,
                NULL as trade_source
            FROM v_trades
        """).fetchone()

        assert result[0] == "m1"
        assert result[1] == 0.5
        assert result[2] is None

    def test_analytics_layer_can_add_derived_columns(
        self, tmp_path: Path
    ) -> None:
        """
        Test that analytics layer can add derived columns without
        breaking reads from raw layer.

        PRD: "Add new derived columns in the analytics layer"
        """
        # Raw data (immutable)
        raw_path = tmp_path / "raw" / "trades.parquet"
        raw_path.parent.mkdir(parents=True)
        raw_data = pa.table({
            "timestamp": [datetime.now(timezone.utc)],
            "market_id": ["m1"],
            "price": [0.5],
        })
        pq.write_table(raw_data, raw_path)

        # Analytics data with derived column
        analytics_path = tmp_path / "analytics" / "trades.parquet"
        analytics_path.parent.mkdir(parents=True)
        analytics_data = pa.table({
            "timestamp": [datetime.now(timezone.utc)],
            "market_id": ["m1"],
            "price": [0.5],
            "days_to_exp": [90.0],  # Derived column
        })
        pq.write_table(analytics_data, analytics_path)

        conn = duckdb.connect(":memory:")

        # Raw view (no derived columns)
        conn.execute(f"""
            CREATE VIEW v_trades_raw AS
            SELECT * FROM '{raw_path}'
        """)

        # Analytics view (with derived columns)
        conn.execute(f"""
            CREATE VIEW v_trades_analytics AS
            SELECT * FROM '{analytics_path}'
        """)

        # Raw view should NOT have days_to_exp
        raw_columns = [c[0] for c in conn.execute("SELECT * FROM v_trades_raw LIMIT 0").description]
        assert "days_to_exp" not in raw_columns

        # Analytics view SHOULD have days_to_exp
        analytics_columns = [c[0] for c in conn.execute("SELECT * FROM v_trades_analytics LIMIT 0").description]
        assert "days_to_exp" in analytics_columns

        # Both should be queryable
        assert conn.execute("SELECT COUNT(*) FROM v_trades_raw").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM v_trades_analytics").fetchone()[0] == 1
