"""Ingestion service for managing data pipeline execution modes.

Supports three ingestion modes:
- off: Disables all ingestion except manual backfill
- batched: Runs update_to_now() on a 5-minute cadence
- live: Runs a continuous loop with short intervals for low-latency alerts
"""

import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from polymkt.config import settings
from polymkt.models.schemas import IngestMode, UpdateSummary
from polymkt.pipeline.update import run_update
from polymkt.storage.metadata import MetadataStore

logger = structlog.get_logger()


class IngestionStatus(str, Enum):
    """Current status of the ingestion service."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class IngestionService:
    """
    Background service for running data ingestion based on INGEST_MODE.

    The service respects the INGEST_MODE setting from the metadata store:
    - off: Service stops itself
    - batched: Runs updates every 5 minutes
    - live: Runs updates every 10-30 seconds for low-latency alerts

    Thread-safe singleton pattern ensures only one instance runs.
    """

    _instance: "IngestionService | None" = None
    _lock = threading.Lock()
    _initialized: bool = False

    # Interval configuration (seconds)
    BATCHED_INTERVAL = 300  # 5 minutes
    LIVE_INTERVAL = 15  # 15 seconds for responsive alerts
    MODE_CHECK_INTERVAL = 10  # Check mode changes every 10 seconds

    def __new__(cls) -> "IngestionService":
        """Singleton pattern - ensures only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the ingestion service."""
        if self._initialized:
            return

        self._status = IngestionStatus.STOPPED
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_run_time: datetime | None = None
        self._last_run_result: UpdateSummary | None = None
        self._run_count = 0
        self._error_count = 0
        self._last_error: str | None = None
        self._current_mode: IngestMode = IngestMode.OFF
        self._initialized = True

    @property
    def status(self) -> IngestionStatus:
        """Get the current service status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if the service is currently running."""
        return self._status == IngestionStatus.RUNNING

    @property
    def stats(self) -> dict[str, Any]:
        """Get current service statistics."""
        return {
            "status": self._status.value,
            "current_mode": self._current_mode.value,
            "last_run_time": self._last_run_time.isoformat() if self._last_run_time else None,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "last_run_result": (
                {
                    "rows_read": self._last_run_result.rows_read,
                    "rows_written": self._last_run_result.rows_written,
                    "duration_seconds": self._last_run_result.duration_seconds,
                }
                if self._last_run_result
                else None
            ),
        }

    def start(self) -> tuple[bool, str]:
        """
        Start the ingestion service.

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._status == IngestionStatus.RUNNING:
                return True, "Ingestion service is already running"

            if self._status == IngestionStatus.STARTING:
                return False, "Ingestion service is already starting"

            # Check current mode
            metadata_store = MetadataStore(settings.metadata_db_path)
            mode = metadata_store.get_ingest_mode()

            if mode == IngestMode.OFF:
                return False, "Cannot start: INGEST_MODE is off. Set to batched or live first."

            self._status = IngestionStatus.STARTING
            self._stop_event.clear()
            self._current_mode = mode

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

            logger.info(
                "ingestion_service_started",
                mode=mode.value,
            )
            return True, f"Ingestion service started in {mode.value} mode"

    def stop(self, timeout: float = 30.0) -> tuple[bool, str]:
        """
        Stop the ingestion service gracefully.

        Args:
            timeout: Maximum seconds to wait for thread to stop

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if self._status == IngestionStatus.STOPPED:
                return True, "Ingestion service is already stopped"

            if self._status == IngestionStatus.STOPPING:
                return False, "Ingestion service is already stopping"

            self._status = IngestionStatus.STOPPING
            self._stop_event.set()

        # Wait for thread outside the lock
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

            if self._thread.is_alive():
                logger.warning("ingestion_service_stop_timeout")
                return False, "Ingestion service stop timed out"

        with self._lock:
            self._status = IngestionStatus.STOPPED
            self._thread = None

        logger.info("ingestion_service_stopped")
        return True, "Ingestion service stopped"

    def _run_loop(self) -> None:
        """Main loop that runs updates based on current mode."""
        self._status = IngestionStatus.RUNNING

        while not self._stop_event.is_set():
            try:
                # Check current mode from metadata store
                metadata_store = MetadataStore(settings.metadata_db_path)
                mode = metadata_store.get_ingest_mode()
                self._current_mode = mode

                if mode == IngestMode.OFF:
                    logger.info("ingestion_mode_off_stopping")
                    break

                # Determine interval based on mode
                interval = (
                    self.LIVE_INTERVAL
                    if mode == IngestMode.LIVE
                    else self.BATCHED_INTERVAL
                )

                # Run update
                self._execute_update()

                # Wait for interval or stop signal
                self._stop_event.wait(timeout=interval)

            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                self._status = IngestionStatus.ERROR
                logger.error("ingestion_loop_error", error=str(e))
                # Wait before retry on error
                self._stop_event.wait(timeout=60)
                if not self._stop_event.is_set():
                    self._status = IngestionStatus.RUNNING

        self._status = IngestionStatus.STOPPED

    def _execute_update(self) -> None:
        """Execute a single update run."""
        try:
            logger.info(
                "ingestion_update_starting",
                mode=self._current_mode.value,
            )

            result = run_update()

            self._last_run_time = datetime.now(timezone.utc)
            self._last_run_result = result
            self._run_count += 1

            logger.info(
                "ingestion_update_complete",
                mode=self._current_mode.value,
                rows_written=result.rows_written,
                duration_seconds=result.duration_seconds,
            )

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(
                "ingestion_update_error",
                mode=self._current_mode.value,
                error=str(e),
            )
            raise

    def run_once(self) -> UpdateSummary:
        """
        Execute a single update run manually (for off mode or testing).

        This is the manual backfill entry point that works regardless of INGEST_MODE.

        Returns:
            UpdateSummary from the pipeline run
        """
        logger.info("ingestion_manual_run_starting")
        result = run_update()
        self._last_run_time = datetime.now(timezone.utc)
        self._last_run_result = result
        self._run_count += 1
        logger.info(
            "ingestion_manual_run_complete",
            rows_written=result.rows_written,
            duration_seconds=result.duration_seconds,
        )
        return result


# Module-level singleton instance
_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """Get the singleton ingestion service instance."""
    global _service
    if _service is None:
        _service = IngestionService()
    return _service
