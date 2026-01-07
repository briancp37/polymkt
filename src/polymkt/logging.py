"""Structured logging configuration for polymkt.

This module configures structlog for production use with:
- JSON output format for log aggregation
- Consistent context binding (run_id, watermarks)
- Timestamp formatting in ISO 8601
- Log level filtering

Usage:
    from polymkt.logging import configure_logging, get_logger

    # Configure once at application startup
    configure_logging(log_level="INFO", json_output=True)

    # Get a logger with context
    logger = get_logger().bind(run_id="abc-123")
    logger.info("processing_started", entity="trades")
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

import structlog


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO 8601 timestamp to log entries."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_service_name(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add service name to log entries."""
    event_dict["service"] = "polymkt"
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
) -> None:
    """
    Configure structlog for production use.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: Whether to output JSON (True) or human-readable (False)
    """
    # Configure standard library logging to capture logs from other libraries
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Build processor chain
    processors: list[Any] = [
        structlog.stdlib.add_log_level,
        add_timestamp,
        add_service_name,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.BoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger()  # type: ignore[no-any-return]


def create_run_logger(run_id: str, operation: str) -> structlog.BoundLogger:
    """
    Create a logger with run_id and operation bound to all entries.

    Args:
        run_id: Unique identifier for this pipeline run
        operation: Operation type (e.g., "bootstrap", "update")

    Returns:
        A logger with context bound
    """
    return get_logger().bind(run_id=run_id, operation=operation)


# Configure logging on module import with defaults
# Can be reconfigured by calling configure_logging() again
configure_logging()
