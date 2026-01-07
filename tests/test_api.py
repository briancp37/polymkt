"""Tests for the FastAPI endpoints."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_list_runs_empty(client: TestClient) -> None:
    """Test listing runs when none exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("polymkt.api.main.settings") as mock_settings:
            mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"
            response = client.get("/api/runs")
            assert response.status_code == 200
            data = response.json()
            assert data["runs"] == []


def test_get_run_not_found(client: TestClient) -> None:
    """Test getting a non-existent run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("polymkt.api.main.settings") as mock_settings:
            mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"
            response = client.get("/api/runs/nonexistent-id")
            assert response.status_code == 404


def test_query_trades_no_data(client: TestClient) -> None:
    """Test querying trades before bootstrap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("polymkt.api.main.settings") as mock_settings:
            mock_settings.parquet_dir = Path(tmpdir) / "parquet"
            response = client.post(
                "/api/query/trades",
                json={"market_id": "test"},
            )
            assert response.status_code == 400
            assert "Run bootstrap first" in response.json()["detail"]


def test_get_watermarks_empty(client: TestClient) -> None:
    """Test getting watermarks when none exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("polymkt.api.main.settings") as mock_settings:
            mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"
            response = client.get("/api/watermarks")
            assert response.status_code == 200
            assert response.json() == {}
