"""Tests for the Backtests persistence feature."""

import tempfile
import time
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app
from polymkt.models.schemas import (
    BacktestCreateRequest,
    BacktestMetrics,
    BacktestTradeRecord,
    BacktestUpdateRequest,
    StrategyConfig,
)
from polymkt.storage.backtests import BacktestNotFoundError, BacktestStore


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_metadata.db"


@pytest.fixture
def backtest_store(temp_db_path: Path) -> BacktestStore:
    """Create a BacktestStore for testing."""
    return BacktestStore(temp_db_path)


@pytest.fixture
def sample_strategy_config() -> StrategyConfig:
    """Create a sample strategy configuration."""
    return StrategyConfig(
        name="buy_favorite",
        entry_days_to_exp=90.0,
        exit_rule="expiry",
        favorite_rule="max_yes_price",
        fee_rate=0.0,
        slippage_rate=0.0,
        position_size=1.0,
    )


# =============================================================================
# BacktestStore unit tests
# =============================================================================


class TestBacktestStoreCreation:
    """Tests for BacktestStore initialization and table creation."""

    def test_init_creates_table(self, temp_db_path: Path) -> None:
        """Test that initializing the store creates the backtests table."""
        store = BacktestStore(temp_db_path)
        assert temp_db_path.exists()

        # Verify table exists by trying to list backtests
        summaries, total = store.list_backtests()
        assert summaries == []
        assert total == 0

    def test_init_creates_parent_dirs(self) -> None:
        """Test that initializing the store creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dirs" / "metadata.db"
            BacktestStore(db_path)
            assert db_path.exists()


class TestBacktestCreate:
    """Tests for creating backtests."""

    def test_create_backtest_basic(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test creating a basic backtest."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        backtest = backtest_store.create_backtest(request)

        assert backtest.id is not None
        assert len(backtest.id) == 36  # UUID format
        assert backtest.dataset_id == "dataset-123"
        assert backtest.strategy_config.name == "buy_favorite"
        assert backtest.strategy_config.entry_days_to_exp == 90.0
        assert backtest.status == "pending"
        assert backtest.metrics is None
        assert backtest.trades is None
        assert backtest.equity_curve is None
        assert backtest.created_at is not None
        assert backtest.updated_at is not None
        assert backtest.completed_at is None

    def test_create_backtest_with_fees_and_slippage(
        self, backtest_store: BacktestStore
    ) -> None:
        """Test creating a backtest with fees and slippage."""
        config = StrategyConfig(
            name="buy_favorite",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            fee_rate=0.01,
            slippage_rate=0.005,
            position_size=0.5,
        )
        request = BacktestCreateRequest(
            dataset_id="dataset-456",
            strategy_config=config,
        )
        backtest = backtest_store.create_backtest(request)

        assert backtest.strategy_config.fee_rate == 0.01
        assert backtest.strategy_config.slippage_rate == 0.005
        assert backtest.strategy_config.position_size == 0.5

    def test_create_backtest_with_extra_params(
        self, backtest_store: BacktestStore
    ) -> None:
        """Test creating a backtest with extra strategy parameters."""
        config = StrategyConfig(
            name="custom_strategy",
            extra_params={"custom_threshold": 0.75, "lookback_days": 30},
        )
        request = BacktestCreateRequest(
            dataset_id="dataset-789",
            strategy_config=config,
        )
        backtest = backtest_store.create_backtest(request)

        assert backtest.strategy_config.extra_params is not None
        assert backtest.strategy_config.extra_params["custom_threshold"] == 0.75
        assert backtest.strategy_config.extra_params["lookback_days"] == 30

    def test_create_multiple_backtests(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test creating multiple backtests."""
        for i in range(5):
            request = BacktestCreateRequest(
                dataset_id=f"dataset-{i}",
                strategy_config=sample_strategy_config,
            )
            backtest_store.create_backtest(request)

        summaries, total = backtest_store.list_backtests()
        assert total == 5
        assert len(summaries) == 5


class TestBacktestGet:
    """Tests for getting backtests by ID."""

    def test_get_backtest_by_id(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test getting a backtest by ID."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        retrieved = backtest_store.get_backtest(created.id)
        assert retrieved.id == created.id
        assert retrieved.dataset_id == created.dataset_id
        assert retrieved.strategy_config.name == created.strategy_config.name

    def test_get_nonexistent_backtest_raises(self, backtest_store: BacktestStore) -> None:
        """Test that getting a nonexistent backtest raises an error."""
        with pytest.raises(BacktestNotFoundError):
            backtest_store.get_backtest("nonexistent-uuid")

    def test_get_backtest_with_results(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test that a completed backtest can be retrieved with all results."""
        # Create backtest
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        # Update with results
        metrics = BacktestMetrics(
            total_return=15.5,
            total_pnl=1550.0,
            win_rate=0.65,
            trade_count=20,
            winning_trades=13,
            losing_trades=7,
        )
        trades = [
            BacktestTradeRecord(
                market_id="market-1",
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                entry_price=0.45,
                exit_time=datetime(2024, 4, 1, tzinfo=timezone.utc),
                exit_price=1.0,
                position_size=100.0,
                pnl=55.0,
            )
        ]
        equity_curve = [
            {"timestamp": "2024-01-01", "equity": 10000},
            {"timestamp": "2024-04-01", "equity": 11550},
        ]

        update_request = BacktestUpdateRequest(
            status="completed",
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
        )
        backtest_store.update_backtest(created.id, update_request)

        # Retrieve and verify
        retrieved = backtest_store.get_backtest(created.id)
        assert retrieved.status == "completed"
        assert retrieved.metrics is not None
        assert retrieved.metrics.total_return == 15.5
        assert retrieved.trades is not None
        assert len(retrieved.trades) == 1
        assert retrieved.equity_curve is not None
        assert len(retrieved.equity_curve) == 2
        assert retrieved.completed_at is not None


class TestBacktestUpdate:
    """Tests for updating backtests."""

    def test_update_backtest_status(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test updating just the backtest status."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        update_request = BacktestUpdateRequest(status="running")
        updated = backtest_store.update_backtest(created.id, update_request)

        assert updated.status == "running"
        assert updated.updated_at > created.updated_at
        assert updated.completed_at is None  # Not completed yet

    def test_update_backtest_with_metrics(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test updating backtest with metrics."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        metrics = BacktestMetrics(
            total_return=25.0,
            total_pnl=2500.0,
            win_rate=0.7,
            trade_count=50,
            winning_trades=35,
            losing_trades=15,
            max_drawdown=8.5,
            sharpe_ratio=1.8,
        )

        update_request = BacktestUpdateRequest(status="completed", metrics=metrics)
        updated = backtest_store.update_backtest(created.id, update_request)

        assert updated.metrics is not None
        assert updated.metrics.total_return == 25.0
        assert updated.metrics.max_drawdown == 8.5
        assert updated.metrics.sharpe_ratio == 1.8

    def test_update_backtest_with_trades(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test updating backtest with trade records."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        trades = [
            BacktestTradeRecord(
                market_id="market-1",
                election_group_id="group-1",
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                entry_price=0.5,
                exit_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
                exit_price=1.0,
                position_size=100.0,
                pnl=50.0,
                fees_paid=1.0,
                slippage_cost=0.5,
            ),
            BacktestTradeRecord(
                market_id="market-2",
                entry_time=datetime(2024, 2, 1, tzinfo=timezone.utc),
                entry_price=0.3,
                exit_time=datetime(2024, 4, 1, tzinfo=timezone.utc),
                exit_price=0.0,
                position_size=100.0,
                pnl=-30.0,
            ),
        ]

        update_request = BacktestUpdateRequest(trades=trades)
        updated = backtest_store.update_backtest(created.id, update_request)

        assert updated.trades is not None
        assert len(updated.trades) == 2
        assert updated.trades[0].market_id == "market-1"
        assert updated.trades[0].pnl == 50.0
        assert updated.trades[1].pnl == -30.0

    def test_update_backtest_sets_completed_at(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test that completing a backtest sets completed_at timestamp."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)
        assert created.completed_at is None

        # Update to running (should not set completed_at)
        update_request = BacktestUpdateRequest(status="running")
        updated = backtest_store.update_backtest(created.id, update_request)
        assert updated.completed_at is None

        # Update to completed (should set completed_at)
        update_request = BacktestUpdateRequest(status="completed")
        updated = backtest_store.update_backtest(created.id, update_request)
        assert updated.completed_at is not None

    def test_update_backtest_with_error(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test updating a failed backtest with error message."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        update_request = BacktestUpdateRequest(
            status="failed",
            error_message="No markets matched the strategy criteria",
        )
        updated = backtest_store.update_backtest(created.id, update_request)

        assert updated.status == "failed"
        assert updated.error_message == "No markets matched the strategy criteria"

    def test_update_nonexistent_backtest_raises(
        self, backtest_store: BacktestStore
    ) -> None:
        """Test that updating a nonexistent backtest raises an error."""
        with pytest.raises(BacktestNotFoundError):
            backtest_store.update_backtest(
                "nonexistent-uuid",
                BacktestUpdateRequest(status="completed"),
            )

    def test_update_preserves_created_at(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test that updates preserve the original created_at timestamp."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)
        original_created_at = created.created_at

        # Wait a bit and update
        time.sleep(0.01)
        update_request = BacktestUpdateRequest(status="completed")
        updated = backtest_store.update_backtest(created.id, update_request)

        assert updated.created_at == original_created_at
        assert updated.updated_at > original_created_at


class TestBacktestDelete:
    """Tests for deleting backtests."""

    def test_delete_backtest(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test deleting a backtest."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        backtest_store.delete_backtest(created.id)

        with pytest.raises(BacktestNotFoundError):
            backtest_store.get_backtest(created.id)

    def test_delete_nonexistent_backtest_raises(self, backtest_store: BacktestStore) -> None:
        """Test that deleting a nonexistent backtest raises an error."""
        with pytest.raises(BacktestNotFoundError):
            backtest_store.delete_backtest("nonexistent-uuid")

    def test_delete_updates_count(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test that deleting a backtest updates the count."""
        # Create 3 backtests
        backtest_ids = []
        for i in range(3):
            request = BacktestCreateRequest(
                dataset_id=f"dataset-{i}",
                strategy_config=sample_strategy_config,
            )
            backtest = backtest_store.create_backtest(request)
            backtest_ids.append(backtest.id)

        _, total = backtest_store.list_backtests()
        assert total == 3

        # Delete one
        backtest_store.delete_backtest(backtest_ids[0])

        _, total = backtest_store.list_backtests()
        assert total == 2


class TestBacktestList:
    """Tests for listing backtests."""

    def test_list_empty(self, backtest_store: BacktestStore) -> None:
        """Test listing with no backtests."""
        summaries, total = backtest_store.list_backtests()
        assert summaries == []
        assert total == 0

    def test_list_returns_summaries(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test that list returns summary format."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        backtest_store.create_backtest(request)

        summaries, total = backtest_store.list_backtests()
        assert total == 1
        assert len(summaries) == 1

        summary = summaries[0]
        assert summary.dataset_id == "dataset-123"
        assert summary.strategy_name == "buy_favorite"
        assert summary.status == "pending"
        assert summary.total_return is None  # Not completed
        assert summary.trade_count is None

    def test_list_with_completed_backtest(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test that list shows metrics for completed backtests."""
        request = BacktestCreateRequest(
            dataset_id="dataset-123",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)

        # Complete the backtest with metrics
        metrics = BacktestMetrics(total_return=25.5, trade_count=30)
        update_request = BacktestUpdateRequest(status="completed", metrics=metrics)
        backtest_store.update_backtest(created.id, update_request)

        summaries, _ = backtest_store.list_backtests()
        assert summaries[0].total_return == 25.5
        assert summaries[0].trade_count == 30
        assert summaries[0].completed_at is not None

    def test_list_pagination_limit(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test pagination with limit."""
        for i in range(10):
            request = BacktestCreateRequest(
                dataset_id=f"dataset-{i}",
                strategy_config=sample_strategy_config,
            )
            backtest_store.create_backtest(request)

        summaries, total = backtest_store.list_backtests(limit=5)
        assert total == 10
        assert len(summaries) == 5

    def test_list_pagination_offset(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test pagination with offset."""
        for i in range(10):
            request = BacktestCreateRequest(
                dataset_id=f"dataset-{i}",
                strategy_config=sample_strategy_config,
            )
            backtest_store.create_backtest(request)

        summaries, total = backtest_store.list_backtests(limit=5, offset=5)
        assert total == 10
        assert len(summaries) == 5

    def test_list_ordered_by_created_at(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test that list is ordered by created_at descending."""
        dataset_ids = ["first", "second", "third"]
        for dataset_id in dataset_ids:
            request = BacktestCreateRequest(
                dataset_id=dataset_id,
                strategy_config=sample_strategy_config,
            )
            backtest_store.create_backtest(request)
            time.sleep(0.01)  # Ensure different timestamps

        summaries, _ = backtest_store.list_backtests()
        # Most recent should be first
        assert summaries[0].dataset_id == "third"
        assert summaries[2].dataset_id == "first"

    def test_list_filter_by_dataset_id(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test filtering backtests by dataset_id."""
        # Create backtests for different datasets
        for i in range(3):
            request = BacktestCreateRequest(
                dataset_id="dataset-A",
                strategy_config=sample_strategy_config,
            )
            backtest_store.create_backtest(request)

        for i in range(2):
            request = BacktestCreateRequest(
                dataset_id="dataset-B",
                strategy_config=sample_strategy_config,
            )
            backtest_store.create_backtest(request)

        # Filter by dataset A
        summaries, total = backtest_store.list_backtests(dataset_id="dataset-A")
        assert total == 3
        assert all(s.dataset_id == "dataset-A" for s in summaries)

        # Filter by dataset B
        summaries, total = backtest_store.list_backtests(dataset_id="dataset-B")
        assert total == 2
        assert all(s.dataset_id == "dataset-B" for s in summaries)


# =============================================================================
# API endpoint tests
# =============================================================================


class TestBacktestsAPICreate:
    """Tests for the POST /api/backtests endpoint."""

    def test_create_backtest_endpoint(self, client: TestClient) -> None:
        """Test creating a backtest via API."""
        response = client.post(
            "/api/backtests",
            json={
                "dataset_id": "test-dataset",
                "strategy_config": {
                    "name": "buy_favorite",
                    "entry_days_to_exp": 90.0,
                    "exit_rule": "expiry",
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["dataset_id"] == "test-dataset"
        assert data["strategy_config"]["name"] == "buy_favorite"
        assert data["status"] == "pending"

    def test_create_backtest_with_full_config(self, client: TestClient) -> None:
        """Test creating a backtest with full strategy configuration."""
        response = client.post(
            "/api/backtests",
            json={
                "dataset_id": "test-dataset",
                "strategy_config": {
                    "name": "buy_favorite",
                    "entry_days_to_exp": 90.0,
                    "exit_rule": "expiry",
                    "favorite_rule": "max_yes_price",
                    "fee_rate": 0.01,
                    "slippage_rate": 0.005,
                    "position_size": 0.5,
                    "extra_params": {"custom_param": "value"},
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["strategy_config"]["fee_rate"] == 0.01
        assert data["strategy_config"]["slippage_rate"] == 0.005
        assert data["strategy_config"]["extra_params"]["custom_param"] == "value"

    def test_create_backtest_missing_required_fields(self, client: TestClient) -> None:
        """Test that missing required fields return 422."""
        response = client.post(
            "/api/backtests",
            json={"dataset_id": "test-dataset"},  # Missing strategy_config
        )
        assert response.status_code == 422


class TestBacktestsAPIList:
    """Tests for the GET /api/backtests endpoint."""

    def test_list_backtests_endpoint(self, client: TestClient) -> None:
        """Test listing backtests via API."""
        # Create a backtest first
        client.post(
            "/api/backtests",
            json={
                "dataset_id": "test-dataset",
                "strategy_config": {"name": "test_strategy"},
            },
        )

        response = client.get("/api/backtests")
        assert response.status_code == 200
        data = response.json()
        assert "backtests" in data
        assert "total_count" in data
        assert "has_more" in data
        assert data["total_count"] >= 1

    def test_list_backtests_pagination(self, client: TestClient) -> None:
        """Test pagination in list endpoint."""
        # Create multiple backtests
        for i in range(5):
            client.post(
                "/api/backtests",
                json={
                    "dataset_id": f"dataset-{i}",
                    "strategy_config": {"name": "test_strategy"},
                },
            )

        response = client.get("/api/backtests?limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["backtests"]) == 2
        assert data["total_count"] >= 5
        assert data["has_more"] is True

    def test_list_backtests_filter_by_dataset(self, client: TestClient) -> None:
        """Test filtering by dataset_id."""
        # Create backtests for different datasets
        client.post(
            "/api/backtests",
            json={
                "dataset_id": "filter-test-A",
                "strategy_config": {"name": "test_strategy"},
            },
        )
        client.post(
            "/api/backtests",
            json={
                "dataset_id": "filter-test-B",
                "strategy_config": {"name": "test_strategy"},
            },
        )

        response = client.get("/api/backtests?dataset_id=filter-test-A")
        assert response.status_code == 200
        data = response.json()
        assert all(b["dataset_id"] == "filter-test-A" for b in data["backtests"])


class TestBacktestsAPIGet:
    """Tests for the GET /api/backtests/{backtest_id} endpoint."""

    def test_get_backtest_endpoint(self, client: TestClient) -> None:
        """Test getting a backtest by ID."""
        # Create a backtest
        create_response = client.post(
            "/api/backtests",
            json={
                "dataset_id": "test-dataset",
                "strategy_config": {"name": "test_strategy"},
            },
        )
        backtest_id = create_response.json()["id"]

        response = client.get(f"/api/backtests/{backtest_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == backtest_id

    def test_get_nonexistent_backtest_returns_404(self, client: TestClient) -> None:
        """Test that getting a nonexistent backtest returns 404."""
        response = client.get("/api/backtests/nonexistent-uuid")
        assert response.status_code == 404


class TestBacktestsAPIUpdate:
    """Tests for the PUT /api/backtests/{backtest_id} endpoint."""

    def test_update_backtest_status(self, client: TestClient) -> None:
        """Test updating backtest status."""
        # Create a backtest
        create_response = client.post(
            "/api/backtests",
            json={
                "dataset_id": "test-dataset",
                "strategy_config": {"name": "test_strategy"},
            },
        )
        backtest_id = create_response.json()["id"]

        response = client.put(
            f"/api/backtests/{backtest_id}",
            json={"status": "running"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "running"

    def test_update_backtest_with_results(self, client: TestClient) -> None:
        """Test updating backtest with full results."""
        # Create a backtest
        create_response = client.post(
            "/api/backtests",
            json={
                "dataset_id": "test-dataset",
                "strategy_config": {"name": "test_strategy"},
            },
        )
        backtest_id = create_response.json()["id"]

        response = client.put(
            f"/api/backtests/{backtest_id}",
            json={
                "status": "completed",
                "metrics": {
                    "total_return": 25.5,
                    "trade_count": 10,
                    "win_rate": 0.7,
                },
                "trades": [
                    {
                        "market_id": "m1",
                        "entry_time": "2024-01-01T00:00:00Z",
                        "entry_price": 0.5,
                        "exit_time": "2024-03-01T00:00:00Z",
                        "exit_price": 1.0,
                        "position_size": 100,
                        "pnl": 50.0,
                    }
                ],
                "equity_curve": [
                    {"timestamp": "2024-01-01", "equity": 10000},
                    {"timestamp": "2024-03-01", "equity": 12550},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["metrics"]["total_return"] == 25.5
        assert len(data["trades"]) == 1
        assert len(data["equity_curve"]) == 2
        assert data["completed_at"] is not None

    def test_update_nonexistent_backtest_returns_404(self, client: TestClient) -> None:
        """Test that updating a nonexistent backtest returns 404."""
        response = client.put(
            "/api/backtests/nonexistent-uuid",
            json={"status": "completed"},
        )
        assert response.status_code == 404


class TestBacktestsAPIDelete:
    """Tests for the DELETE /api/backtests/{backtest_id} endpoint."""

    def test_delete_backtest_endpoint(self, client: TestClient) -> None:
        """Test deleting a backtest."""
        # Create a backtest
        create_response = client.post(
            "/api/backtests",
            json={
                "dataset_id": "test-dataset",
                "strategy_config": {"name": "test_strategy"},
            },
        )
        backtest_id = create_response.json()["id"]

        response = client.delete(f"/api/backtests/{backtest_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify it's gone
        get_response = client.get(f"/api/backtests/{backtest_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_backtest_returns_404(self, client: TestClient) -> None:
        """Test that deleting a nonexistent backtest returns 404."""
        response = client.delete("/api/backtests/nonexistent-uuid")
        assert response.status_code == 404


# =============================================================================
# Integration tests
# =============================================================================


class TestBacktestsIntegration:
    """Integration tests for the backtests feature."""

    def test_full_crud_workflow(
        self, backtest_store: BacktestStore, sample_strategy_config: StrategyConfig
    ) -> None:
        """Test complete CRUD workflow."""
        # Create
        request = BacktestCreateRequest(
            dataset_id="integration-test",
            strategy_config=sample_strategy_config,
        )
        created = backtest_store.create_backtest(request)
        assert created.status == "pending"

        # Read
        retrieved = backtest_store.get_backtest(created.id)
        assert retrieved.id == created.id

        # Update to running
        backtest_store.update_backtest(
            created.id, BacktestUpdateRequest(status="running")
        )
        running = backtest_store.get_backtest(created.id)
        assert running.status == "running"

        # Update to completed with results
        metrics = BacktestMetrics(
            total_return=20.0,
            trade_count=15,
            win_rate=0.6,
        )
        trades = [
            BacktestTradeRecord(
                market_id="m1",
                entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                entry_price=0.4,
                exit_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
                exit_price=1.0,
                position_size=100,
                pnl=60.0,
            )
        ]
        backtest_store.update_backtest(
            created.id,
            BacktestUpdateRequest(status="completed", metrics=metrics, trades=trades),
        )

        completed = backtest_store.get_backtest(created.id)
        assert completed.status == "completed"
        assert completed.metrics is not None
        assert completed.trades is not None
        assert completed.completed_at is not None

        # List
        summaries, total = backtest_store.list_backtests()
        assert total >= 1
        matching = [s for s in summaries if s.id == created.id]
        assert len(matching) == 1
        assert matching[0].total_return == 20.0

        # Delete
        backtest_store.delete_backtest(created.id)
        with pytest.raises(BacktestNotFoundError):
            backtest_store.get_backtest(created.id)

    def test_multiple_backtests_for_same_dataset(
        self, backtest_store: BacktestStore
    ) -> None:
        """Test running multiple backtests on the same dataset with different configs."""
        dataset_id = "shared-dataset"

        # Create backtests with different strategies
        configs = [
            StrategyConfig(name="buy_favorite", entry_days_to_exp=90.0),
            StrategyConfig(name="buy_favorite", entry_days_to_exp=60.0),
            StrategyConfig(name="buy_favorite", entry_days_to_exp=30.0),
        ]

        backtest_ids = []
        for config in configs:
            request = BacktestCreateRequest(
                dataset_id=dataset_id,
                strategy_config=config,
            )
            backtest = backtest_store.create_backtest(request)
            backtest_ids.append(backtest.id)

        # List backtests for dataset
        summaries, total = backtest_store.list_backtests(dataset_id=dataset_id)
        assert total == 3

        # All backtests should be for the same dataset
        assert all(s.dataset_id == dataset_id for s in summaries)

        # Each backtest remains independently accessible
        for backtest_id in backtest_ids:
            backtest = backtest_store.get_backtest(backtest_id)
            assert backtest.dataset_id == dataset_id
