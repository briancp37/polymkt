"""Field normalization and validation for data pipeline.

This module provides consistent normalization and type-casting for:
- Timestamps: Normalized to UTC datetime
- Numerics: Proper precision for price/usd_amount/token_amount
- Addresses: Lowercase with 0x prefix
- Invalid rows: Quarantined or rejected with clear error logs
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import structlog

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of validating and normalizing a table."""

    valid_table: pa.Table
    quarantined_rows: list[dict[str, Any]] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    rows_processed: int = 0
    rows_valid: int = 0
    rows_quarantined: int = 0

    @property
    def has_errors(self) -> bool:
        """Check if there were any validation errors."""
        return len(self.error_messages) > 0


@dataclass
class BatchValidationResult:
    """Result of validating a single batch (memory-efficient, no row storage)."""

    valid_batch: pa.RecordBatch
    rows_processed: int = 0
    rows_valid: int = 0
    rows_quarantined: int = 0
    error_sample: str | None = None  # Only store a sample error, not all


# Maximum quarantined rows to store in memory (for sampling)
MAX_QUARANTINE_SAMPLES = 100

# Regex pattern for valid Ethereum addresses
ETH_ADDRESS_PATTERN = re.compile(r"^(0x)?[a-fA-F0-9]{40}$")


def normalize_address(address: str | None) -> str | None:
    """
    Normalize Ethereum address to lowercase with 0x prefix.

    Args:
        address: Raw address string

    Returns:
        Normalized address or None if invalid/null
    """
    if address is None or address == "":
        return None

    # Strip whitespace
    address = address.strip()

    # Check if it matches the pattern
    if not ETH_ADDRESS_PATTERN.match(address):
        return None

    # Ensure lowercase and 0x prefix
    address = address.lower()
    if not address.startswith("0x"):
        address = "0x" + address

    return address


def normalize_timestamp(
    value: Any,
    field_name: str,
) -> datetime | None:
    """
    Normalize a timestamp value to UTC datetime.

    Args:
        value: Raw timestamp value (string, datetime, or timestamp)
        field_name: Name of the field for error reporting

    Returns:
        UTC datetime or None if invalid/null
    """
    if value is None:
        return None

    # If already a datetime
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    # Handle Unix epoch timestamps (int or float)
    if isinstance(value, (int, float)):
        # Detect milliseconds vs seconds
        if value > 1e12:  # Likely milliseconds
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(value, tz=timezone.utc)

    # If string, try to parse
    if isinstance(value, str):
        if not value or value.strip() == "":
            return None

        value = value.strip()

        # Try to parse as Unix epoch (numeric string)
        try:
            epoch = float(value)
            # Detect milliseconds vs seconds
            if epoch > 1e12:  # Likely milliseconds
                return datetime.fromtimestamp(epoch / 1000, tz=timezone.utc)
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except ValueError:
            pass  # Not a numeric string, continue with datetime format parsing

        # Normalize short timezone offsets like +00 to +0000
        # Matches patterns like +00, -05, +0530, etc.
        tz_match = re.search(r'([+-])(\d{2})(?::?(\d{2}))?$', value)
        if tz_match:
            sign, hours, minutes = tz_match.groups()
            minutes = minutes or '00'
            # Normalize to +HHMM format without colon
            normalized_tz = f"{sign}{hours}{minutes}"
            value = value[:tz_match.start()] + normalized_tz

        timestamp_formats = [
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
        ]

        for fmt in timestamp_formats:
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue

        logger.warning(
            "timestamp_parse_failed",
            field=field_name,
            value=value,
        )
        return None

    return None


def normalize_numeric(
    value: Any,
    field_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """
    Normalize a numeric value with optional range validation.

    Args:
        value: Raw numeric value
        field_name: Name of the field for error reporting
        min_value: Optional minimum value (inclusive)
        max_value: Optional maximum value (inclusive)

    Returns:
        Normalized float or None if invalid/null
    """
    if value is None:
        return None

    try:
        result = float(value)

        # Check for NaN or infinity
        if result != result or result == float("inf") or result == float("-inf"):
            logger.warning(
                "numeric_invalid_special",
                field=field_name,
                value=value,
            )
            return None

        # Check range if specified
        if min_value is not None and result < min_value:
            logger.warning(
                "numeric_below_min",
                field=field_name,
                value=result,
                min_value=min_value,
            )
            return None

        if max_value is not None and result > max_value:
            logger.warning(
                "numeric_above_max",
                field=field_name,
                value=result,
                max_value=max_value,
            )
            return None

        return result
    except (TypeError, ValueError) as e:
        logger.warning(
            "numeric_parse_failed",
            field=field_name,
            value=value,
            error=str(e),
        )
        return None


def _normalize_address_column(column: pa.Array) -> pa.Array:
    """Normalize an entire column of addresses."""
    normalized = []
    for value in column.to_pylist():
        normalized.append(normalize_address(value))
    return pa.array(normalized, type=pa.string())


def _check_required_field(
    row: dict[str, Any],
    field_name: str,
    errors: list[str],
) -> bool:
    """Check if a required field is present and non-null."""
    if field_name not in row or row[field_name] is None:
        errors.append(f"Missing required field: {field_name}")
        return False
    return True


def validate_and_normalize_trades(
    table: pa.Table,
    normalize_addresses: bool = True,
) -> ValidationResult:
    """
    Validate and normalize trades data.

    Performs:
    - Timestamp validation (must be valid UTC datetime)
    - Numeric validation (price 0-1, amounts >= 0)
    - Address normalization (lowercase 0x format)
    - Quarantine of invalid rows

    Args:
        table: PyArrow table of trades data
        normalize_addresses: Whether to normalize address fields

    Returns:
        ValidationResult with valid table and quarantined rows
    """
    result = ValidationResult(
        valid_table=table,
        rows_processed=table.num_rows,
    )

    if table.num_rows == 0:
        result.rows_valid = 0
        return result

    # Convert to Python for row-by-row validation
    rows = table.to_pylist()
    valid_rows: list[dict[str, Any]] = []
    column_names = table.column_names

    for row_idx, row in enumerate(rows):
        row_errors: list[str] = []

        # Validate required fields
        required_fields = ["timestamp", "market_id", "price", "transaction_hash"]
        has_required = all(
            _check_required_field(row, f, row_errors) for f in required_fields
        )

        if not has_required:
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        # Normalize and validate timestamp
        ts = normalize_timestamp(row.get("timestamp"), "timestamp")
        if ts is None:
            row_errors.append(f"Invalid timestamp: {row.get('timestamp')}")
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        # Validate price (should be 0-1 for prediction markets)
        price = normalize_numeric(row.get("price"), "price", min_value=0.0, max_value=1.0)
        if price is None:
            row_errors.append(f"Invalid price: {row.get('price')} (expected 0-1)")
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        # Validate amounts (should be >= 0)
        usd_amount = normalize_numeric(row.get("usd_amount"), "usd_amount", min_value=0.0)
        token_amount = normalize_numeric(row.get("token_amount"), "token_amount", min_value=0.0)

        # Build normalized row
        normalized_row = dict(row)
        normalized_row["timestamp"] = ts
        normalized_row["price"] = price
        normalized_row["usd_amount"] = usd_amount
        normalized_row["token_amount"] = token_amount

        # Normalize addresses if enabled
        if normalize_addresses:
            if "maker" in row:
                normalized_row["maker"] = normalize_address(row.get("maker"))
            if "taker" in row:
                normalized_row["taker"] = normalize_address(row.get("taker"))

        valid_rows.append(normalized_row)

    # Build result table from valid rows
    if valid_rows:
        # Ensure we maintain the original column order
        result.valid_table = pa.Table.from_pylist(valid_rows, schema=table.schema)
    else:
        # Return empty table with same schema
        result.valid_table = table.slice(0, 0)

    result.rows_valid = len(valid_rows)
    result.rows_quarantined = len(result.quarantined_rows)

    # Log summary
    logger.info(
        "trades_validation_complete",
        rows_processed=result.rows_processed,
        rows_valid=result.rows_valid,
        rows_quarantined=result.rows_quarantined,
    )

    if result.rows_quarantined > 0:
        logger.warning(
            "trades_rows_quarantined",
            count=result.rows_quarantined,
            sample_errors=result.error_messages[:5],
        )

    return result


def validate_and_normalize_markets(
    table: pa.Table,
) -> ValidationResult:
    """
    Validate and normalize markets data.

    Performs:
    - Timestamp validation for created_at and closed_time
    - Numeric validation for volume
    - Required field validation (id, question)

    Args:
        table: PyArrow table of markets data

    Returns:
        ValidationResult with valid table and quarantined rows
    """
    result = ValidationResult(
        valid_table=table,
        rows_processed=table.num_rows,
    )

    if table.num_rows == 0:
        result.rows_valid = 0
        return result

    rows = table.to_pylist()
    valid_rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        row_errors: list[str] = []

        # Validate required fields
        if not _check_required_field(row, "id", row_errors):
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        if not _check_required_field(row, "question", row_errors):
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        # Normalize timestamps
        created_at = normalize_timestamp(row.get("created_at"), "created_at")
        closed_time = normalize_timestamp(row.get("closed_time"), "closed_time")

        # Validate volume
        volume = normalize_numeric(row.get("volume"), "volume", min_value=0.0)

        # Build normalized row
        normalized_row = dict(row)
        normalized_row["created_at"] = created_at
        normalized_row["closed_time"] = closed_time
        normalized_row["volume"] = volume

        valid_rows.append(normalized_row)

    # Build result table from valid rows
    if valid_rows:
        result.valid_table = pa.Table.from_pylist(valid_rows, schema=table.schema)
    else:
        result.valid_table = table.slice(0, 0)

    result.rows_valid = len(valid_rows)
    result.rows_quarantined = len(result.quarantined_rows)

    logger.info(
        "markets_validation_complete",
        rows_processed=result.rows_processed,
        rows_valid=result.rows_valid,
        rows_quarantined=result.rows_quarantined,
    )

    if result.rows_quarantined > 0:
        logger.warning(
            "markets_rows_quarantined",
            count=result.rows_quarantined,
            sample_errors=result.error_messages[:5],
        )

    return result


def validate_and_normalize_events(
    table: pa.Table,
) -> ValidationResult:
    """
    Validate and normalize events data.

    Performs:
    - Required field validation (event_id)
    - Tags validation (must be a list of strings if present)
    - Timestamp validation for created_at

    Args:
        table: PyArrow table of events data

    Returns:
        ValidationResult with valid table and quarantined rows
    """
    result = ValidationResult(
        valid_table=table,
        rows_processed=table.num_rows,
    )

    if table.num_rows == 0:
        result.rows_valid = 0
        return result

    rows = table.to_pylist()
    valid_rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        row_errors: list[str] = []

        # Validate required field: event_id
        if not _check_required_field(row, "event_id", row_errors):
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        # Normalize timestamp if present
        created_at = normalize_timestamp(row.get("created_at"), "created_at")

        # Validate tags (should be a list of strings or None)
        tags = row.get("tags")
        if tags is not None:
            if not isinstance(tags, list):
                row_errors.append(f"Invalid tags: expected list, got {type(tags).__name__}")
                if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                    result.quarantined_rows.append(
                        {"row_index": row_idx, "row": row, "errors": row_errors}
                    )
                result.error_messages.extend(
                    [f"Row {row_idx}: {e}" for e in row_errors]
                )
                continue
            # Ensure all items are strings
            tags = [str(t) if t is not None else "" for t in tags]

        # Build normalized row
        normalized_row = dict(row)
        normalized_row["created_at"] = created_at
        normalized_row["tags"] = tags if tags is not None else []

        valid_rows.append(normalized_row)

    # Build result table from valid rows
    if valid_rows:
        result.valid_table = pa.Table.from_pylist(valid_rows, schema=table.schema)
    else:
        result.valid_table = table.slice(0, 0)

    result.rows_valid = len(valid_rows)
    result.rows_quarantined = len(result.quarantined_rows)

    logger.info(
        "events_validation_complete",
        rows_processed=result.rows_processed,
        rows_valid=result.rows_valid,
        rows_quarantined=result.rows_quarantined,
    )

    if result.rows_quarantined > 0:
        logger.warning(
            "events_rows_quarantined",
            count=result.rows_quarantined,
            sample_errors=result.error_messages[:5],
        )

    return result


def validate_and_normalize_order_filled(
    table: pa.Table,
    normalize_addresses: bool = True,
) -> ValidationResult:
    """
    Validate and normalize order filled data.

    Performs:
    - Timestamp validation
    - Numeric validation for amounts
    - Address normalization

    Args:
        table: PyArrow table of order filled data
        normalize_addresses: Whether to normalize address fields

    Returns:
        ValidationResult with valid table and quarantined rows
    """
    result = ValidationResult(
        valid_table=table,
        rows_processed=table.num_rows,
    )

    if table.num_rows == 0:
        result.rows_valid = 0
        return result

    rows = table.to_pylist()
    valid_rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        row_errors: list[str] = []

        # Validate required fields
        required_fields = ["timestamp", "transaction_hash"]
        has_required = all(
            _check_required_field(row, f, row_errors) for f in required_fields
        )

        if not has_required:
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        # Normalize timestamp
        ts = normalize_timestamp(row.get("timestamp"), "timestamp")
        if ts is None:
            row_errors.append(f"Invalid timestamp: {row.get('timestamp')}")
            if len(result.quarantined_rows) < MAX_QUARANTINE_SAMPLES:
                result.quarantined_rows.append(
                    {"row_index": row_idx, "row": row, "errors": row_errors}
                )
            result.error_messages.extend(
                [f"Row {row_idx}: {e}" for e in row_errors]
            )
            continue

        # Validate amounts
        maker_amount = normalize_numeric(
            row.get("maker_amount_filled"), "maker_amount_filled", min_value=0.0
        )
        taker_amount = normalize_numeric(
            row.get("taker_amount_filled"), "taker_amount_filled", min_value=0.0
        )

        # Build normalized row
        normalized_row = dict(row)
        normalized_row["timestamp"] = ts
        normalized_row["maker_amount_filled"] = maker_amount
        normalized_row["taker_amount_filled"] = taker_amount

        # Normalize addresses if enabled
        if normalize_addresses:
            if "maker" in row:
                normalized_row["maker"] = normalize_address(row.get("maker"))
            if "taker" in row:
                normalized_row["taker"] = normalize_address(row.get("taker"))

        valid_rows.append(normalized_row)

    # Build result table from valid rows
    if valid_rows:
        result.valid_table = pa.Table.from_pylist(valid_rows, schema=table.schema)
    else:
        result.valid_table = table.slice(0, 0)

    result.rows_valid = len(valid_rows)
    result.rows_quarantined = len(result.quarantined_rows)

    logger.info(
        "order_filled_validation_complete",
        rows_processed=result.rows_processed,
        rows_valid=result.rows_valid,
        rows_quarantined=result.rows_quarantined,
    )

    if result.rows_quarantined > 0:
        logger.warning(
            "order_filled_rows_quarantined",
            count=result.rows_quarantined,
            sample_errors=result.error_messages[:5],
        )

    return result


# =============================================================================
# Vectorized Batch Validation Functions (Memory-Efficient)
# =============================================================================
# These functions use PyArrow compute operations instead of Python loops
# for processing large datasets without memory exhaustion.


def _vectorized_normalize_timestamp_column(
    col: pa.Array,
    field_name: str,
) -> pa.Array:
    """
    Normalize a timestamp column using vectorized operations.

    Handles string timestamps by parsing them in bulk.
    """
    if pa.types.is_timestamp(col.type):
        # Already a timestamp, just ensure UTC
        return col

    if not pa.types.is_string(col.type) and not pa.types.is_large_string(col.type):
        # Not a string, try to cast
        try:
            return col.cast(pa.timestamp("us", tz="UTC"))
        except pa.ArrowInvalid:
            return col

    # String column - parse timestamps
    # Use PyArrow's strptime for bulk parsing
    try:
        # Try ISO format first (most common)
        parsed = pc.strptime(col, format="%Y-%m-%d %H:%M:%S", unit="us")
        return parsed
    except pa.ArrowInvalid:
        pass

    try:
        # Try ISO with T separator
        parsed = pc.strptime(col, format="%Y-%m-%dT%H:%M:%S", unit="us")
        return parsed
    except pa.ArrowInvalid:
        pass

    # Fall back to per-value parsing for complex formats (but still return array)
    logger.warning(
        "timestamp_fallback_parsing",
        field=field_name,
        message="Using fallback timestamp parsing",
    )
    results = []
    for val in col.to_pylist():
        results.append(normalize_timestamp(val, field_name))
    return pa.array(results, type=pa.timestamp("us", tz="UTC"))


def _vectorized_normalize_address_column(col: pa.Array) -> pa.Array:
    """
    Normalize an entire column of Ethereum addresses using vectorized operations.

    Converts to lowercase and ensures 0x prefix.
    """
    if col.null_count == len(col):
        return col  # All nulls, nothing to do

    # Use PyArrow string operations for bulk processing
    # Convert to lowercase
    lowered = pc.utf8_lower(col)

    # Check for 0x prefix and add if missing (vectorized)
    has_prefix = pc.starts_with(lowered, "0x")
    prefix_array = pa.array(["0x"] * len(col), type=pa.string())

    # Where no prefix, prepend "0x"
    result = pc.if_else(
        has_prefix,
        lowered,
        pc.binary_join_element_wise(prefix_array, lowered, ""),
    )

    return result


def validate_trades_batch(
    batch: pa.RecordBatch,
    target_schema: pa.Schema,
    normalize_addresses: bool = True,
) -> BatchValidationResult:
    """
    Validate and normalize a batch of trades using vectorized operations.

    Memory-efficient: processes data in Arrow format without Python conversion.

    Args:
        batch: PyArrow RecordBatch of trades data
        target_schema: Target schema to conform to
        normalize_addresses: Whether to normalize address fields

    Returns:
        BatchValidationResult with valid batch only
    """
    result = BatchValidationResult(
        valid_batch=batch,
        rows_processed=batch.num_rows,
    )

    if batch.num_rows == 0:
        return result

    # Convert to table for easier column operations
    table = pa.Table.from_batches([batch])
    column_names = table.column_names

    # Build validity mask using vectorized operations
    validity_masks = []

    # Check required fields are not null
    required_fields = ["timestamp", "market_id", "price", "transaction_hash"]
    for field in required_fields:
        if field in column_names:
            col = table.column(field)
            not_null = pc.is_valid(col)
            validity_masks.append(not_null)
        else:
            # Field missing entirely - all rows invalid
            result.error_sample = f"Missing required field: {field}"
            result.rows_valid = 0
            result.rows_quarantined = batch.num_rows
            result.valid_batch = batch.slice(0, 0)
            return result

    # Validate price range (0-1)
    if "price" in column_names:
        price_col = table.column("price")
        # Cast to float if needed
        if not pa.types.is_floating(price_col.type):
            try:
                price_col = price_col.cast(pa.float64())
            except pa.ArrowInvalid:
                result.error_sample = "Invalid price values (cannot cast to float)"

        price_valid = pc.and_(
            pc.greater_equal(price_col, 0.0),
            pc.less_equal(price_col, 1.0),
        )
        validity_masks.append(price_valid)

    # Combine all validity masks
    if validity_masks:
        combined_mask = validity_masks[0]
        for mask in validity_masks[1:]:
            combined_mask = pc.and_(combined_mask, mask)
    else:
        combined_mask = pa.array([True] * batch.num_rows, type=pa.bool_())

    # Filter to valid rows
    valid_table = table.filter(combined_mask)

    # Now normalize columns in the valid table
    columns = {}
    for field in target_schema:
        if field.name in valid_table.column_names:
            col = valid_table.column(field.name)

            # Handle timestamp normalization
            if pa.types.is_timestamp(field.type):
                col = _vectorized_normalize_timestamp_column(col, field.name)

            # Handle address normalization
            if normalize_addresses and field.name in ("maker", "taker"):
                col = _vectorized_normalize_address_column(col)

            # Cast to target type
            try:
                col = col.cast(field.type)
            except pa.ArrowInvalid:
                pass  # Keep original type if cast fails

            columns[field.name] = col
        else:
            # Create null column for missing fields
            columns[field.name] = pa.nulls(valid_table.num_rows, type=field.type)

    # Build final table with target schema
    final_table = pa.table(columns, schema=target_schema)

    result.valid_batch = final_table.to_batches()[0] if final_table.num_rows > 0 else batch.slice(0, 0)
    result.rows_valid = final_table.num_rows
    result.rows_quarantined = batch.num_rows - final_table.num_rows

    return result


def validate_markets_batch(
    batch: pa.RecordBatch,
    target_schema: pa.Schema,
) -> BatchValidationResult:
    """
    Validate and normalize a batch of markets using vectorized operations.

    Args:
        batch: PyArrow RecordBatch of markets data
        target_schema: Target schema to conform to

    Returns:
        BatchValidationResult with valid batch only
    """
    result = BatchValidationResult(
        valid_batch=batch,
        rows_processed=batch.num_rows,
    )

    if batch.num_rows == 0:
        return result

    table = pa.Table.from_batches([batch])
    column_names = table.column_names

    # Build validity mask
    validity_masks = []

    # Check required fields
    required_fields = ["id", "question"]
    for field in required_fields:
        if field in column_names:
            col = table.column(field)
            not_null = pc.is_valid(col)
            validity_masks.append(not_null)
        else:
            result.error_sample = f"Missing required field: {field}"
            result.rows_valid = 0
            result.rows_quarantined = batch.num_rows
            result.valid_batch = batch.slice(0, 0)
            return result

    # Combine validity masks
    if validity_masks:
        combined_mask = validity_masks[0]
        for mask in validity_masks[1:]:
            combined_mask = pc.and_(combined_mask, mask)
    else:
        combined_mask = pa.array([True] * batch.num_rows, type=pa.bool_())

    valid_table = table.filter(combined_mask)

    # Normalize columns
    columns = {}
    for field in target_schema:
        if field.name in valid_table.column_names:
            col = valid_table.column(field.name)

            # Handle timestamp normalization
            if pa.types.is_timestamp(field.type):
                col = _vectorized_normalize_timestamp_column(col, field.name)

            try:
                col = col.cast(field.type)
            except pa.ArrowInvalid:
                pass

            columns[field.name] = col
        else:
            columns[field.name] = pa.nulls(valid_table.num_rows, type=field.type)

    final_table = pa.table(columns, schema=target_schema)

    result.valid_batch = final_table.to_batches()[0] if final_table.num_rows > 0 else batch.slice(0, 0)
    result.rows_valid = final_table.num_rows
    result.rows_quarantined = batch.num_rows - final_table.num_rows

    return result


def validate_order_filled_batch(
    batch: pa.RecordBatch,
    target_schema: pa.Schema,
    normalize_addresses: bool = True,
) -> BatchValidationResult:
    """
    Validate and normalize a batch of order_filled using vectorized operations.

    Args:
        batch: PyArrow RecordBatch of order_filled data
        target_schema: Target schema to conform to
        normalize_addresses: Whether to normalize address fields

    Returns:
        BatchValidationResult with valid batch only
    """
    result = BatchValidationResult(
        valid_batch=batch,
        rows_processed=batch.num_rows,
    )

    if batch.num_rows == 0:
        return result

    table = pa.Table.from_batches([batch])
    column_names = table.column_names

    validity_masks = []

    # Check required fields
    required_fields = ["timestamp", "transaction_hash"]
    for field in required_fields:
        if field in column_names:
            col = table.column(field)
            not_null = pc.is_valid(col)
            validity_masks.append(not_null)
        else:
            result.error_sample = f"Missing required field: {field}"
            result.rows_valid = 0
            result.rows_quarantined = batch.num_rows
            result.valid_batch = batch.slice(0, 0)
            return result

    # Combine validity masks
    if validity_masks:
        combined_mask = validity_masks[0]
        for mask in validity_masks[1:]:
            combined_mask = pc.and_(combined_mask, mask)
    else:
        combined_mask = pa.array([True] * batch.num_rows, type=pa.bool_())

    valid_table = table.filter(combined_mask)

    # Normalize columns
    columns = {}
    for field in target_schema:
        if field.name in valid_table.column_names:
            col = valid_table.column(field.name)

            if pa.types.is_timestamp(field.type):
                col = _vectorized_normalize_timestamp_column(col, field.name)

            if normalize_addresses and field.name in ("maker", "taker"):
                col = _vectorized_normalize_address_column(col)

            try:
                col = col.cast(field.type)
            except pa.ArrowInvalid:
                pass

            columns[field.name] = col
        else:
            columns[field.name] = pa.nulls(valid_table.num_rows, type=field.type)

    final_table = pa.table(columns, schema=target_schema)

    result.valid_batch = final_table.to_batches()[0] if final_table.num_rows > 0 else batch.slice(0, 0)
    result.rows_valid = final_table.num_rows
    result.rows_quarantined = batch.num_rows - final_table.num_rows

    return result


def validate_events_batch(
    batch: pa.RecordBatch,
    target_schema: pa.Schema,
) -> BatchValidationResult:
    """
    Validate and normalize a batch of events using vectorized operations.

    Args:
        batch: PyArrow RecordBatch of events data
        target_schema: Target schema to conform to

    Returns:
        BatchValidationResult with valid batch only
    """
    result = BatchValidationResult(
        valid_batch=batch,
        rows_processed=batch.num_rows,
    )

    if batch.num_rows == 0:
        return result

    table = pa.Table.from_batches([batch])
    column_names = table.column_names

    validity_masks = []

    # Check required field: event_id
    if "event_id" in column_names:
        col = table.column("event_id")
        not_null = pc.is_valid(col)
        validity_masks.append(not_null)
    else:
        result.error_sample = "Missing required field: event_id"
        result.rows_valid = 0
        result.rows_quarantined = batch.num_rows
        result.valid_batch = batch.slice(0, 0)
        return result

    # Combine validity masks
    if validity_masks:
        combined_mask = validity_masks[0]
        for mask in validity_masks[1:]:
            combined_mask = pc.and_(combined_mask, mask)
    else:
        combined_mask = pa.array([True] * batch.num_rows, type=pa.bool_())

    valid_table = table.filter(combined_mask)

    # Normalize columns
    columns = {}
    for field in target_schema:
        if field.name in valid_table.column_names:
            col = valid_table.column(field.name)

            if pa.types.is_timestamp(field.type):
                col = _vectorized_normalize_timestamp_column(col, field.name)

            try:
                col = col.cast(field.type)
            except pa.ArrowInvalid:
                pass

            columns[field.name] = col
        else:
            columns[field.name] = pa.nulls(valid_table.num_rows, type=field.type)

    final_table = pa.table(columns, schema=target_schema)

    result.valid_batch = final_table.to_batches()[0] if final_table.num_rows > 0 else batch.slice(0, 0)
    result.rows_valid = final_table.num_rows
    result.rows_quarantined = batch.num_rows - final_table.num_rows

    return result
