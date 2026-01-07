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


class TestQueryTradesAPI:
    """Tests for the /api/query/trades endpoint with multiple market_ids."""

    @pytest.fixture
    def setup_bootstrap(self, client: TestClient) -> tuple[Path, Path, Path]:
        """Set up bootstrap data for API testing."""
        import tempfile

        tmpdir = tempfile.mkdtemp()
        tmp_path = Path(tmpdir)

        # Create sample CSVs
        markets_csv = tmp_path / "markets.csv"
        markets_csv.write_text(
            """createdAt,id,question,answer1,answer2,neg_risk,market_slug,token1,token2,condition_id,volume,ticker,closedTime,description,category
2024-01-01 00:00:00,market1,Will A win?,Yes,No,false,market-a,tok1a,tok1b,cond1,1000.0,MKT1,2024-12-31 23:59:59,Market A,Sports
2024-01-01 00:00:00,market2,Will B win?,Yes,No,false,market-b,tok2a,tok2b,cond2,2000.0,MKT2,2024-12-31 23:59:59,Market B,Sports
2024-01-01 00:00:00,market3,Will C win?,Yes,No,false,market-c,tok3a,tok3b,cond3,3000.0,MKT3,2024-12-31 23:59:59,Market C,Politics
"""
        )

        trades_csv = tmp_path / "trades.csv"
        trades_csv.write_text(
            """timestamp,market_id,maker,taker,nonusdc_side,maker_direction,taker_direction,price,usd_amount,token_amount,transactionHash
2024-01-15 10:00:00,market1,0xmaker1,0xtaker1,YES,buy,sell,0.65,100.0,153.84,0xhash1
2024-01-15 11:00:00,market1,0xmaker2,0xtaker2,NO,sell,buy,0.35,50.0,142.85,0xhash2
2024-01-16 10:00:00,market2,0xmaker3,0xtaker3,YES,buy,sell,0.50,150.0,300.0,0xhash3
2024-01-16 11:00:00,market2,0xmaker4,0xtaker4,YES,buy,sell,0.55,200.0,363.63,0xhash4
2024-01-17 10:00:00,market3,0xmaker5,0xtaker5,YES,buy,sell,0.70,250.0,357.14,0xhash5
"""
        )

        order_filled_csv = tmp_path / "orderFilled.csv"
        order_filled_csv.write_text(
            """timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash
2024-01-15 10:00:00,0xmaker1,asset1,100.0,0xtaker1,asset2,153.84,0xhash1
"""
        )

        parquet_dir = tmp_path / "parquet"
        duckdb_path = tmp_path / "test.duckdb"
        metadata_path = tmp_path / "metadata.db"

        # Run bootstrap
        from polymkt.pipeline.bootstrap import run_bootstrap

        run_bootstrap(
            markets_csv=markets_csv,
            trades_csv=trades_csv,
            order_filled_csv=order_filled_csv,
            parquet_dir=parquet_dir,
            duckdb_path=duckdb_path,
            metadata_db_path=metadata_path,
        )

        return parquet_dir, duckdb_path, metadata_path

    def test_query_multiple_market_ids(
        self, client: TestClient, setup_bootstrap: tuple[Path, Path, Path]
    ) -> None:
        """Test querying with multiple market_ids via API."""
        parquet_dir, duckdb_path, metadata_path = setup_bootstrap

        with patch("polymkt.api.main.settings") as mock_settings:
            mock_settings.parquet_dir = parquet_dir
            mock_settings.duckdb_path = duckdb_path
            mock_settings.metadata_db_path = metadata_path

            response = client.post(
                "/api/query/trades",
                json={"market_ids": ["market1", "market2"]},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 4  # 2 from market1 + 2 from market2
            assert data["count"] == 4
            assert data["has_more"] is False
            for trade in data["trades"]:
                assert trade["market_id"] in ["market1", "market2"]

    def test_query_multiple_market_ids_with_pagination(
        self, client: TestClient, setup_bootstrap: tuple[Path, Path, Path]
    ) -> None:
        """Test pagination with multiple market_ids."""
        parquet_dir, duckdb_path, metadata_path = setup_bootstrap

        with patch("polymkt.api.main.settings") as mock_settings:
            mock_settings.parquet_dir = parquet_dir
            mock_settings.duckdb_path = duckdb_path
            mock_settings.metadata_db_path = metadata_path

            response = client.post(
                "/api/query/trades",
                json={
                    "market_ids": ["market1", "market2", "market3"],
                    "limit": 2,
                    "offset": 0,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 5  # All trades
            assert data["count"] == 2
            assert data["has_more"] is True

    def test_query_single_market_id_precedence(
        self, client: TestClient, setup_bootstrap: tuple[Path, Path, Path]
    ) -> None:
        """Test that market_id takes precedence over market_ids."""
        parquet_dir, duckdb_path, metadata_path = setup_bootstrap

        with patch("polymkt.api.main.settings") as mock_settings:
            mock_settings.parquet_dir = parquet_dir
            mock_settings.duckdb_path = duckdb_path
            mock_settings.metadata_db_path = metadata_path

            response = client.post(
                "/api/query/trades",
                json={
                    "market_id": "market1",
                    "market_ids": ["market2", "market3"],
                },
            )

            assert response.status_code == 200
            data = response.json()
            # Should return only market1 trades
            assert data["total_count"] == 2
            for trade in data["trades"]:
                assert trade["market_id"] == "market1"
