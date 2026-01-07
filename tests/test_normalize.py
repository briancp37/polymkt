"""Tests for the field normalization and validation module."""

import tempfile
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from polymkt.pipeline.normalize import (
    ValidationResult,
    normalize_address,
    normalize_numeric,
    normalize_timestamp,
    validate_and_normalize_markets,
    validate_and_normalize_order_filled,
    validate_and_normalize_trades,
)


# =============================================================================
# Unit Tests for normalize_address
# =============================================================================


class TestNormalizeAddress:
    """Tests for normalize_address function."""

    def test_valid_address_lowercase(self) -> None:
        """Test valid lowercase address with 0x prefix."""
        result = normalize_address("0x1234567890abcdef1234567890abcdef12345678")
        assert result == "0x1234567890abcdef1234567890abcdef12345678"

    def test_valid_address_uppercase(self) -> None:
        """Test valid uppercase address is normalized to lowercase."""
        result = normalize_address("0x1234567890ABCDEF1234567890ABCDEF12345678")
        assert result == "0x1234567890abcdef1234567890abcdef12345678"

    def test_valid_address_without_prefix(self) -> None:
        """Test address without 0x prefix gets it added."""
        result = normalize_address("1234567890abcdef1234567890abcdef12345678")
        assert result == "0x1234567890abcdef1234567890abcdef12345678"

    def test_valid_address_with_whitespace(self) -> None:
        """Test address with whitespace is stripped."""
        result = normalize_address("  0x1234567890abcdef1234567890abcdef12345678  ")
        assert result == "0x1234567890abcdef1234567890abcdef12345678"

    def test_invalid_address_too_short(self) -> None:
        """Test address that is too short returns None."""
        result = normalize_address("0x12345")
        assert result is None

    def test_invalid_address_too_long(self) -> None:
        """Test address that is too long returns None."""
        result = normalize_address("0x1234567890abcdef1234567890abcdef1234567890")
        assert result is None

    def test_invalid_address_non_hex(self) -> None:
        """Test address with non-hex characters returns None."""
        result = normalize_address("0x1234567890GHIJKL1234567890abcdef12345678")
        assert result is None

    def test_null_address(self) -> None:
        """Test null address returns None."""
        result = normalize_address(None)
        assert result is None

    def test_empty_string_address(self) -> None:
        """Test empty string returns None."""
        result = normalize_address("")
        assert result is None


# =============================================================================
# Unit Tests for normalize_timestamp
# =============================================================================


class TestNormalizeTimestamp:
    """Tests for normalize_timestamp function."""

    def test_valid_datetime_with_tz(self) -> None:
        """Test datetime with timezone is converted to UTC."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = normalize_timestamp(dt, "test_field")
        assert result == dt
        assert result.tzinfo == timezone.utc

    def test_valid_datetime_naive(self) -> None:
        """Test naive datetime gets UTC timezone added."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = normalize_timestamp(dt, "test_field")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_valid_string_format_1(self) -> None:
        """Test string format 'YYYY-MM-DD HH:MM:SS'."""
        result = normalize_timestamp("2024-01-15 10:30:00", "test_field")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_valid_string_format_iso(self) -> None:
        """Test ISO format 'YYYY-MM-DDTHH:MM:SS'."""
        result = normalize_timestamp("2024-01-15T10:30:00", "test_field")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_valid_string_format_iso_z(self) -> None:
        """Test ISO format with Z suffix."""
        result = normalize_timestamp("2024-01-15T10:30:00Z", "test_field")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_valid_string_format_with_microseconds(self) -> None:
        """Test format with microseconds."""
        result = normalize_timestamp("2024-01-15T10:30:00.123456", "test_field")
        assert result is not None
        assert result.microsecond == 123456

    def test_invalid_string_format(self) -> None:
        """Test invalid string format returns None."""
        result = normalize_timestamp("not-a-date", "test_field")
        assert result is None

    def test_null_timestamp(self) -> None:
        """Test null timestamp returns None."""
        result = normalize_timestamp(None, "test_field")
        assert result is None

    def test_empty_string_timestamp(self) -> None:
        """Test empty string returns None."""
        result = normalize_timestamp("", "test_field")
        assert result is None


# =============================================================================
# Unit Tests for normalize_numeric
# =============================================================================


class TestNormalizeNumeric:
    """Tests for normalize_numeric function."""

    def test_valid_int(self) -> None:
        """Test integer is converted to float."""
        result = normalize_numeric(100, "test_field")
        assert result == 100.0

    def test_valid_float(self) -> None:
        """Test float is returned as-is."""
        result = normalize_numeric(0.65, "test_field")
        assert result == 0.65

    def test_valid_string_number(self) -> None:
        """Test string number is converted."""
        result = normalize_numeric("0.75", "test_field")
        assert result == 0.75

    def test_min_value_valid(self) -> None:
        """Test value above min is accepted."""
        result = normalize_numeric(0.5, "test_field", min_value=0.0)
        assert result == 0.5

    def test_min_value_invalid(self) -> None:
        """Test value below min is rejected."""
        result = normalize_numeric(-0.5, "test_field", min_value=0.0)
        assert result is None

    def test_max_value_valid(self) -> None:
        """Test value below max is accepted."""
        result = normalize_numeric(0.5, "test_field", max_value=1.0)
        assert result == 0.5

    def test_max_value_invalid(self) -> None:
        """Test value above max is rejected."""
        result = normalize_numeric(1.5, "test_field", max_value=1.0)
        assert result is None

    def test_nan_value(self) -> None:
        """Test NaN is rejected."""
        result = normalize_numeric(float("nan"), "test_field")
        assert result is None

    def test_infinity_value(self) -> None:
        """Test infinity is rejected."""
        result = normalize_numeric(float("inf"), "test_field")
        assert result is None

    def test_null_value(self) -> None:
        """Test null returns None."""
        result = normalize_numeric(None, "test_field")
        assert result is None

    def test_invalid_string(self) -> None:
        """Test non-numeric string is rejected."""
        result = normalize_numeric("not-a-number", "test_field")
        assert result is None


# =============================================================================
# Integration Tests for validate_and_normalize_trades
# =============================================================================


@pytest.fixture
def valid_trades_table() -> pa.Table:
    """Create a valid trades table for testing."""
    return pa.table(
        {
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            ],
            "market_id": ["market1", "market1"],
            "maker": ["0xMaker1", "0xMaker2"],
            "taker": ["0xTaker1", "0xTaker2"],
            "nonusdc_side": ["YES", "NO"],
            "maker_direction": ["buy", "sell"],
            "taker_direction": ["sell", "buy"],
            "price": [0.65, 0.35],
            "usd_amount": [100.0, 50.0],
            "token_amount": [153.84, 142.85],
            "transaction_hash": ["0xhash1", "0xhash2"],
        }
    )


@pytest.fixture
def trades_with_edge_cases() -> pa.Table:
    """Create a trades table with edge cases for testing."""
    return pa.table(
        {
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),  # Valid
                None,  # Invalid - missing timestamp
                datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),  # Valid
                datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),  # Invalid - bad price
            ],
            "market_id": ["market1", "market2", "market3", "market4"],
            "maker": [
                "0x1234567890abcdef1234567890abcdef12345678",  # Valid address
                "0xMaker2",  # Will become None (invalid)
                "1234567890ABCDEF1234567890ABCDEF12345678",  # Will be normalized
                "0x1234567890abcdef1234567890abcdef12345678",
            ],
            "taker": [
                "0xabcdef1234567890abcdef1234567890abcdef12",
                "0xTaker2",
                "0xabcdef1234567890abcdef1234567890abcdef12",
                "0xabcdef1234567890abcdef1234567890abcdef12",
            ],
            "nonusdc_side": ["YES", "YES", "NO", "YES"],
            "maker_direction": ["buy", "buy", "sell", "buy"],
            "taker_direction": ["sell", "sell", "buy", "sell"],
            "price": [0.65, 0.50, 0.35, 1.5],  # Last one is invalid (>1)
            "usd_amount": [100.0, 200.0, 50.0, 100.0],
            "token_amount": [153.84, 400.0, 142.85, 66.66],
            "transaction_hash": ["0xhash1", "0xhash2", "0xhash3", "0xhash4"],
        }
    )


class TestValidateAndNormalizeTrades:
    """Tests for validate_and_normalize_trades function."""

    def test_valid_table_passes(self, valid_trades_table: pa.Table) -> None:
        """Test that a valid table passes validation."""
        result = validate_and_normalize_trades(valid_trades_table)

        assert result.rows_processed == 2
        assert result.rows_valid == 2
        assert result.rows_quarantined == 0
        assert len(result.quarantined_rows) == 0
        assert not result.has_errors

    def test_addresses_are_normalized(self, valid_trades_table: pa.Table) -> None:
        """Test that addresses are normalized to lowercase with 0x prefix."""
        result = validate_and_normalize_trades(
            valid_trades_table, normalize_addresses=True
        )

        # The test addresses in the fixture are not valid 40-char hex
        # so they will be normalized to None
        valid_table = result.valid_table
        assert valid_table.num_rows == 2

    def test_edge_cases_quarantined(self, trades_with_edge_cases: pa.Table) -> None:
        """Test that rows with edge cases are quarantined."""
        result = validate_and_normalize_trades(trades_with_edge_cases)

        # Row 1 (index 1): missing timestamp - quarantined
        # Row 3 (index 3): price > 1 - quarantined
        assert result.rows_processed == 4
        assert result.rows_valid == 2
        assert result.rows_quarantined == 2
        assert len(result.quarantined_rows) == 2
        assert result.has_errors

    def test_empty_table(self) -> None:
        """Test that empty table returns empty result."""
        empty_table = pa.table(
            {
                "timestamp": pa.array([], type=pa.timestamp("us", tz="UTC")),
                "market_id": pa.array([], type=pa.string()),
                "maker": pa.array([], type=pa.string()),
                "taker": pa.array([], type=pa.string()),
                "nonusdc_side": pa.array([], type=pa.string()),
                "maker_direction": pa.array([], type=pa.string()),
                "taker_direction": pa.array([], type=pa.string()),
                "price": pa.array([], type=pa.float64()),
                "usd_amount": pa.array([], type=pa.float64()),
                "token_amount": pa.array([], type=pa.float64()),
                "transaction_hash": pa.array([], type=pa.string()),
            }
        )
        result = validate_and_normalize_trades(empty_table)

        assert result.rows_processed == 0
        assert result.rows_valid == 0
        assert result.rows_quarantined == 0


# =============================================================================
# Integration Tests for validate_and_normalize_markets
# =============================================================================


@pytest.fixture
def valid_markets_table() -> pa.Table:
    """Create a valid markets table for testing."""
    return pa.table(
        {
            "id": ["market1", "market2"],
            "question": ["Will it rain?", "Who wins?"],
            "created_at": [
                datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            ],
            "answer1": ["Yes", "A"],
            "answer2": ["No", "B"],
            "neg_risk": [False, True],
            "market_slug": ["will-it-rain", "who-wins"],
            "token1": ["token1a", "token2a"],
            "token2": ["token1b", "token2b"],
            "condition_id": ["cond1", "cond2"],
            "volume": [1000.0, 5000.0],
            "ticker": ["RAIN", "WIN"],
            "closed_time": [
                datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
            ],
            "description": ["Weather prediction", "Election prediction"],
            "category": ["Weather", "Politics"],
        }
    )


@pytest.fixture
def markets_with_edge_cases() -> pa.Table:
    """Create a markets table with edge cases."""
    return pa.table(
        {
            "id": ["market1", None, "market3"],  # Row 2 missing id
            "question": ["Will it rain?", "Who wins?", None],  # Row 3 missing question
            "created_at": [
                datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
            ],
            "answer1": ["Yes", "A", "X"],
            "answer2": ["No", "B", "Y"],
            "neg_risk": [False, True, False],
            "market_slug": ["will-it-rain", "who-wins", "test"],
            "token1": ["token1a", "token2a", "token3a"],
            "token2": ["token1b", "token2b", "token3b"],
            "condition_id": ["cond1", "cond2", "cond3"],
            "volume": [1000.0, 5000.0, 100.0],
            "ticker": ["RAIN", "WIN", "TEST"],
            "closed_time": [
                datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                None,
                datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
            ],
            "description": ["Weather prediction", "Election prediction", "Test"],
            "category": ["Weather", "Politics", "Test"],
        }
    )


class TestValidateAndNormalizeMarkets:
    """Tests for validate_and_normalize_markets function."""

    def test_valid_table_passes(self, valid_markets_table: pa.Table) -> None:
        """Test that a valid table passes validation."""
        result = validate_and_normalize_markets(valid_markets_table)

        assert result.rows_processed == 2
        assert result.rows_valid == 2
        assert result.rows_quarantined == 0
        assert not result.has_errors

    def test_edge_cases_quarantined(self, markets_with_edge_cases: pa.Table) -> None:
        """Test that rows with missing required fields are quarantined."""
        result = validate_and_normalize_markets(markets_with_edge_cases)

        # Row 1 (index 1): missing id - quarantined
        # Row 2 (index 2): missing question - quarantined
        assert result.rows_processed == 3
        assert result.rows_valid == 1
        assert result.rows_quarantined == 2
        assert result.has_errors


# =============================================================================
# Integration Tests for validate_and_normalize_order_filled
# =============================================================================


@pytest.fixture
def valid_order_filled_table() -> pa.Table:
    """Create a valid order filled table for testing."""
    return pa.table(
        {
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            ],
            "maker": ["0xmaker1", "0xmaker2"],
            "maker_asset_id": ["asset1", "asset3"],
            "maker_amount_filled": [100.0, 50.0],
            "taker": ["0xtaker1", "0xtaker2"],
            "taker_asset_id": ["asset2", "asset4"],
            "taker_amount_filled": [153.84, 142.85],
            "transaction_hash": ["0xhash1", "0xhash2"],
        }
    )


class TestValidateAndNormalizeOrderFilled:
    """Tests for validate_and_normalize_order_filled function."""

    def test_valid_table_passes(self, valid_order_filled_table: pa.Table) -> None:
        """Test that a valid table passes validation."""
        result = validate_and_normalize_order_filled(valid_order_filled_table)

        assert result.rows_processed == 2
        assert result.rows_valid == 2
        assert result.rows_quarantined == 0
        assert not result.has_errors


# =============================================================================
# Integration Test with Bootstrap Pipeline
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_bootstrap_with_validation_and_edge_cases(temp_dir: Path) -> None:
    """Test bootstrap pipeline with validation enabled and edge case data."""
    from polymkt.pipeline.bootstrap import run_bootstrap

    # Create CSV with edge cases
    markets_csv = temp_dir / "markets.csv"
    markets_csv.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category
2024-01-01 00:00:00,market1,Will it rain?,Yes,No,false,will-it-rain,token1a,token1b,cond1,1000.0,RAIN,2024-12-31 23:59:59,Weather prediction,Weather
"""
    )

    # Create trades CSV with one invalid row (price > 1)
    trades_csv = temp_dir / "trades.csv"
    trades_csv.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0x1234567890abcdef1234567890abcdef12345678,0xabcdef1234567890abcdef1234567890abcdef12,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0x1234567890abcdef1234567890abcdef12345679,0xabcdef1234567890abcdef1234567890abcdef13,NO,sell,buy,1.5,50.0,142.85,0xhash2
2024-01-15 12:00:00,market1,0x1234567890abcdef1234567890abcdef1234567a,0xabcdef1234567890abcdef1234567890abcdef14,YES,buy,sell,0.75,200.0,266.67,0xhash3
"""
    )

    # Create order filled CSV
    order_filled_csv = temp_dir / "orderFilled.csv"
    order_filled_csv.write_text(
        """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0x1234567890abcdef1234567890abcdef12345678,asset1,100.0,0xabcdef1234567890abcdef1234567890abcdef12,asset2,153.84,0xhash1
"""
    )

    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    summary = run_bootstrap(
        markets_csv=markets_csv,
        trades_csv=trades_csv,
        order_filled_csv=order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
        validate_data=True,
        normalize_addresses=True,
    )

    # Verify summary
    assert summary.status == "completed"
    assert summary.markets_rows == 1
    # One trade should be quarantined (price > 1)
    assert summary.trades_rows == 2
    assert summary.rows_quarantined.get("trades", 0) == 1
    assert summary.order_filled_rows == 1


def test_bootstrap_without_validation(temp_dir: Path) -> None:
    """Test bootstrap pipeline with validation disabled."""
    from polymkt.pipeline.bootstrap import run_bootstrap

    markets_csv = temp_dir / "markets.csv"
    markets_csv.write_text(
        """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category
2024-01-01 00:00:00,market1,Will it rain?,Yes,No,false,will-it-rain,token1a,token1b,cond1,1000.0,RAIN,2024-12-31 23:59:59,Weather prediction,Weather
"""
    )

    trades_csv = temp_dir / "trades.csv"
    trades_csv.write_text(
        """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
"""
    )

    order_filled_csv = temp_dir / "orderFilled.csv"
    order_filled_csv.write_text(
        """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash1
"""
    )

    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"
    metadata_path = temp_dir / "metadata.db"

    summary = run_bootstrap(
        markets_csv=markets_csv,
        trades_csv=trades_csv,
        order_filled_csv=order_filled_csv,
        parquet_dir=parquet_dir,
        duckdb_path=duckdb_path,
        metadata_db_path=metadata_path,
        validate_data=False,
        normalize_addresses=False,
    )

    assert summary.status == "completed"
    assert summary.markets_rows == 1
    assert summary.trades_rows == 1
    assert summary.order_filled_rows == 1
    # No quarantine stats when validation is disabled
    assert summary.rows_quarantined == {}
