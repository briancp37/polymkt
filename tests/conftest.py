"""Pytest configuration and shared fixtures."""

import pytest

# Configure structlog to not output during tests
import structlog

structlog.configure(
    processors=[
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
