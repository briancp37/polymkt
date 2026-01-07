"""Tests for the backtest engine.

Tests the "buy the favorite" strategy execution including:
- Entry/exit trade finding
- PnL calculation with fees and slippage
- Aggregate metrics computation
- API endpoint integration
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app
from polymkt.backtest.engine import BacktestEngine
from polymkt.models.schemas import (
    BacktestCreateRequest,
    BacktestMetrics,
    BacktestTradeRecord,
    DatasetCreateRequest,
    StrategyConfig,
)
from polymkt.signals.favorites import FavoriteSignal, FavoriteSignalStore
from polymkt.storage.backtests import BacktestStore
from polymkt.storage.datasets import DatasetStore
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.election_groups import ElectionGroupStore


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory structure for tests."""
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_markets_parquet(temp_dir):
    """Create sample markets Parquet file."""
    parquet_dir = temp_dir / "parquet"

    # Create markets with closed_time 90 days in the future from trade timestamps
    base_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
    closed_time = base_time + timedelta(days=90)

    markets = pa.table(
        {
            "id": ["market_trump", "market_biden", "market_harris", "market_other"],
            "question": [
                "Will Trump win?",
                "Will Biden win?",
                "Will Harris win?",
                "Other market",
            ],
            "created_at": [base_time - timedelta(days=180)] * 4,
            "answer1": ["Yes"] * 4,
            "answer2": ["No"] * 4,
            "neg_risk": [False] * 4,
            "market_slug": ["trump-win", "biden-win", "harris-win", "other"] ,
            "token1": ["token1"] * 4,
            "token2": ["token2"] * 4,
            "condition_id": ["cond1", "cond2", "cond3", "cond4"],
            "volume": [1000000.0, 800000.0, 600000.0, 100000.0],
            "ticker": ["TRUMP", "BIDEN", "HARRIS", "OTHER"],
            "closed_time": [closed_time, closed_time, closed_time, None],
            "description": ["Trump market", "Biden market", "Harris market", "Other"],
            "category": ["election", "election", "election", "sports"],
            "event_id": ["event1", "event1", "event1", "event2"],
            "tags": [["politics"], ["politics"], ["politics"], ["sports"]],
        }
    )
    pq.write_table(markets, parquet_dir / "markets.parquet")
    return parquet_dir / "markets.parquet"


@pytest.fixture
def sample_trades_parquet(temp_dir):
    """Create sample trades Parquet file with trades at various days_to_exp."""
    parquet_dir = temp_dir / "parquet"

    # Base time: 90 days before market close
    market_close = datetime(2024, 8, 30, tzinfo=timezone.utc)
    snapshot_time = market_close - timedelta(days=90)  # June 1, 2024
    near_expiry_time = market_close - timedelta(days=1)  # Aug 29, 2024

    trades = pa.table(
        {
            "timestamp": [
                # Trades at ~90 days to expiry (snapshot time)
                snapshot_time,
                snapshot_time + timedelta(hours=1),
                snapshot_time + timedelta(hours=2),
                # Trades near expiry (~1 day)
                near_expiry_time,
                near_expiry_time + timedelta(hours=1),
                near_expiry_time + timedelta(hours=2),
            ],
            "market_id": [
                "market_trump",
                "market_biden",
                "market_harris",
                "market_trump",
                "market_biden",
                "market_harris",
            ],
            "maker": ["0xmaker"] * 6,
            "taker": ["0xtaker"] * 6,
            "nonusdc_side": ["buy"] * 6,
            "maker_direction": ["sell"] * 6,
            "taker_direction": ["buy"] * 6,
            "price": [
                0.55,  # Trump at snapshot (favorite)
                0.30,  # Biden at snapshot
                0.15,  # Harris at snapshot
                0.95,  # Trump near expiry (winning)
                0.03,  # Biden near expiry (losing)
                0.02,  # Harris near expiry (losing)
            ],
            "usd_amount": [100.0] * 6,
            "token_amount": [100.0] * 6,
            "transaction_hash": [f"0xhash{i}" for i in range(6)],
        }
    )
    pq.write_table(trades, parquet_dir / "trades.parquet")
    return parquet_dir / "trades.parquet"


@pytest.fixture
def sample_order_filled_parquet(temp_dir):
    """Create sample order_filled Parquet file."""
    parquet_dir = temp_dir / "parquet"

    order_filled = pa.table(
        {
            "timestamp": [datetime.now(timezone.utc)],
            "maker": ["0xmaker"],
            "maker_asset_id": ["asset1"],
            "maker_amount_filled": [100.0],
            "taker": ["0xtaker"],
            "taker_asset_id": ["asset2"],
            "taker_amount_filled": [100.0],
            "transaction_hash": ["0xorder_hash"],
        }
    )
    pq.write_table(order_filled, parquet_dir / "order_filled.parquet")
    return parquet_dir / "order_filled.parquet"


@pytest.fixture
def duckdb_layer(temp_dir, sample_markets_parquet, sample_trades_parquet, sample_order_filled_parquet):
    """Create DuckDB layer with sample data."""
    parquet_dir = temp_dir / "parquet"
    duckdb_path = temp_dir / "test.duckdb"

    layer = DuckDBLayer(duckdb_path, parquet_dir)
    layer.create_views()
    return layer


@pytest.fixture
def stores(temp_dir):
    """Create all required stores."""
    metadata_db = temp_dir / "metadata.db"
    return {
        "dataset_store": DatasetStore(metadata_db),
        "backtest_store": BacktestStore(metadata_db),
        "election_group_store": ElectionGroupStore(metadata_db),
        "favorite_signal_store": FavoriteSignalStore(metadata_db),
    }


@pytest.fixture
def sample_dataset(stores):
    """Create a sample dataset with election markets."""
    dataset = stores["dataset_store"].create_dataset(
        DatasetCreateRequest(
            name="2024 Election Markets",
            description="Presidential election markets",
            market_ids=["market_trump", "market_biden", "market_harris"],
        )
    )
    return dataset


@pytest.fixture
def sample_election_group(stores):
    """Create a sample election group."""
    group = stores["election_group_store"].create_group(
        name="2024 Presidential",
        description="Presidential candidates",
        market_ids=["market_trump", "market_biden", "market_harris"],
    )
    return group


@pytest.fixture
def sample_favorite_signal(stores, sample_election_group):
    """Create a sample favorite signal at 90 days to expiry."""
    signal = FavoriteSignal(
        election_group_id=sample_election_group["id"],
        election_group_name=sample_election_group["name"],
        favorite_market_id="market_trump",
        favorite_price=0.55,
        favorite_question="Will Trump win?",
        snapshot_days_to_exp=90.0,
        computed_at=datetime.now(timezone.utc),
        all_market_prices={
            "market_trump": 0.55,
            "market_biden": 0.30,
            "market_harris": 0.15,
        },
    )
    stores["favorite_signal_store"].save_signals([signal])
    return signal


@pytest.fixture
def backtest_engine(duckdb_layer, stores):
    """Create a backtest engine with all dependencies."""
    return BacktestEngine(
        duckdb_layer=duckdb_layer,
        dataset_store=stores["dataset_store"],
        backtest_store=stores["backtest_store"],
        election_group_store=stores["election_group_store"],
        favorite_signal_store=stores["favorite_signal_store"],
    )


# =============================================================================
# Test entry/exit trade finding
# =============================================================================


class TestFindEntryTrade:
    """Tests for finding entry trades at the snapshot."""

    def test_find_entry_trade_at_snapshot(self, backtest_engine):
        """Verify entry trade is found at the correct days_to_exp."""
        result = backtest_engine._find_entry_trade(
            "market_trump", target_days_to_exp=90.0, tolerance=0.5
        )

        assert result is not None
        entry_time, entry_price = result
        assert entry_price == 0.55
        assert isinstance(entry_time, datetime)

    def test_find_entry_trade_returns_none_for_no_trades(self, backtest_engine):
        """Verify None is returned when no trades exist at snapshot."""
        result = backtest_engine._find_entry_trade(
            "nonexistent_market", target_days_to_exp=90.0, tolerance=0.5
        )

        assert result is None


class TestFindExitTrade:
    """Tests for finding exit trades at expiry."""

    def test_find_exit_at_expiry(self, backtest_engine):
        """Verify exit trade is found near expiry."""
        result = backtest_engine._find_exit_at_expiry("market_trump")

        assert result is not None
        exit_time, exit_price = result
        # The exit price is the resolved price based on last trade
        # Trump was trading at 0.95 (>0.5) near expiry, so resolves to 1.0
        assert exit_price == 1.0
        assert isinstance(exit_time, datetime)

    def test_find_exit_resolves_to_zero_for_losers(self, backtest_engine):
        """Verify losing markets resolve to 0.0."""
        result = backtest_engine._find_exit_at_expiry("market_biden")

        assert result is not None
        exit_time, exit_price = result
        # Biden was trading at 0.03 (<0.5) near expiry, so resolves to 0.0
        assert exit_price == 0.0


# =============================================================================
# Test PnL calculation
# =============================================================================


class TestComputeMetrics:
    """Tests for aggregate metrics computation."""

    def test_compute_metrics_with_winning_trades(self, backtest_engine):
        """Verify metrics are computed correctly for winning trades."""
        trades = [
            BacktestTradeRecord(
                market_id="market_trump",
                election_group_id="group1",
                entry_time=datetime.now(timezone.utc),
                entry_price=0.55,
                exit_time=datetime.now(timezone.utc) + timedelta(days=90),
                exit_price=1.0,
                position_size=1.0,
                pnl=0.45,  # 1.0 - 0.55
                fees_paid=0.02,
                slippage_cost=0.01,
            )
        ]

        config = StrategyConfig(
            name="buy_favorite",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            position_size=1.0,
        )

        metrics = backtest_engine._compute_metrics(trades, config)

        assert metrics.trade_count == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 1.0
        assert metrics.total_pnl == 0.45
        assert metrics.avg_trade_pnl == 0.45

    def test_compute_metrics_with_losing_trades(self, backtest_engine):
        """Verify metrics are computed correctly for losing trades."""
        trades = [
            BacktestTradeRecord(
                market_id="market_biden",
                election_group_id="group1",
                entry_time=datetime.now(timezone.utc),
                entry_price=0.60,
                exit_time=datetime.now(timezone.utc) + timedelta(days=90),
                exit_price=0.0,
                position_size=1.0,
                pnl=-0.60,  # 0.0 - 0.60
                fees_paid=0.02,
                slippage_cost=0.01,
            )
        ]

        config = StrategyConfig(
            name="buy_favorite",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            position_size=1.0,
        )

        metrics = backtest_engine._compute_metrics(trades, config)

        assert metrics.trade_count == 1
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == -0.60

    def test_compute_metrics_with_mixed_trades(self, backtest_engine):
        """Verify metrics are computed correctly for mixed win/loss trades."""
        trades = [
            BacktestTradeRecord(
                market_id="market_trump",
                election_group_id="group1",
                entry_time=datetime.now(timezone.utc),
                entry_price=0.55,
                exit_time=datetime.now(timezone.utc) + timedelta(days=90),
                exit_price=1.0,
                position_size=1.0,
                pnl=0.45,
                fees_paid=0.0,
                slippage_cost=0.0,
            ),
            BacktestTradeRecord(
                market_id="market_loser",
                election_group_id="group2",
                entry_time=datetime.now(timezone.utc),
                entry_price=0.60,
                exit_time=datetime.now(timezone.utc) + timedelta(days=90),
                exit_price=0.0,
                position_size=1.0,
                pnl=-0.60,
                fees_paid=0.0,
                slippage_cost=0.0,
            ),
        ]

        config = StrategyConfig(
            name="buy_favorite",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            position_size=1.0,
        )

        metrics = backtest_engine._compute_metrics(trades, config)

        assert metrics.trade_count == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5
        assert metrics.total_pnl == pytest.approx(-0.15, abs=0.01)

    def test_compute_metrics_empty_trades(self, backtest_engine):
        """Verify metrics are computed correctly for empty trade list."""
        config = StrategyConfig(
            name="buy_favorite",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            position_size=1.0,
        )

        metrics = backtest_engine._compute_metrics([], config)

        assert metrics.trade_count == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0

    def test_compute_metrics_includes_fees_and_slippage(self, backtest_engine):
        """Verify that fees and slippage are reflected in trade PnL."""
        # PnL already includes fees/slippage in the BacktestTradeRecord
        trades = [
            BacktestTradeRecord(
                market_id="market_trump",
                election_group_id="group1",
                entry_time=datetime.now(timezone.utc),
                entry_price=0.55,
                exit_time=datetime.now(timezone.utc) + timedelta(days=90),
                exit_price=1.0,
                position_size=1.0,
                pnl=0.42,  # 0.45 gross - 0.02 fees - 0.01 slippage
                fees_paid=0.02,
                slippage_cost=0.01,
            )
        ]

        config = StrategyConfig(
            name="buy_favorite",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            position_size=1.0,
            fee_rate=0.01,
            slippage_rate=0.005,
        )

        metrics = backtest_engine._compute_metrics(trades, config)

        assert metrics.total_pnl == 0.42


# =============================================================================
# Test full backtest execution
# =============================================================================


class TestBacktestExecution:
    """Tests for full backtest execution flow."""

    def test_execute_backtest_success(
        self,
        backtest_engine,
        stores,
        sample_dataset,
        sample_election_group,
        sample_favorite_signal,
    ):
        """Verify successful backtest execution."""
        # Create a pending backtest
        backtest = stores["backtest_store"].create_backtest(
            BacktestCreateRequest(
                dataset_id=sample_dataset.id,
                strategy_config=StrategyConfig(
                    name="buy_favorite",
                    entry_days_to_exp=90.0,
                    exit_rule="expiry",
                    fee_rate=0.0,
                    slippage_rate=0.0,
                    position_size=1.0,
                ),
            )
        )

        # Execute the backtest
        result = backtest_engine.execute(backtest.id)

        assert result.status == "completed"
        assert result.metrics is not None
        assert result.metrics.trade_count >= 1
        assert len(result.trades) >= 1
        assert len(result.equity_curve) >= 1

    def test_execute_backtest_with_fees_and_slippage(
        self,
        backtest_engine,
        stores,
        sample_dataset,
        sample_election_group,
        sample_favorite_signal,
    ):
        """Verify backtest execution applies fees and slippage."""
        # Create a pending backtest with fees
        backtest = stores["backtest_store"].create_backtest(
            BacktestCreateRequest(
                dataset_id=sample_dataset.id,
                strategy_config=StrategyConfig(
                    name="buy_favorite",
                    entry_days_to_exp=90.0,
                    exit_rule="expiry",
                    fee_rate=0.01,
                    slippage_rate=0.005,
                    position_size=1.0,
                ),
            )
        )

        # Execute the backtest
        result = backtest_engine.execute(backtest.id)

        assert result.status == "completed"
        assert result.trades[0].fees_paid > 0
        assert result.trades[0].slippage_cost > 0

    def test_execute_backtest_updates_status(
        self,
        backtest_engine,
        stores,
        sample_dataset,
        sample_election_group,
        sample_favorite_signal,
    ):
        """Verify backtest status is updated correctly."""
        backtest = stores["backtest_store"].create_backtest(
            BacktestCreateRequest(
                dataset_id=sample_dataset.id,
                strategy_config=StrategyConfig(
                    name="buy_favorite",
                    entry_days_to_exp=90.0,
                    exit_rule="expiry",
                    position_size=1.0,
                ),
            )
        )

        assert backtest.status == "pending"

        result = backtest_engine.execute(backtest.id)

        # Check the stored backtest
        updated = stores["backtest_store"].get_backtest(backtest.id)
        assert updated.status == "completed"
        assert updated.metrics is not None
        assert updated.completed_at is not None

    def test_execute_backtest_fails_for_non_pending(
        self,
        backtest_engine,
        stores,
        sample_dataset,
    ):
        """Verify execution fails for non-pending backtests."""
        backtest = stores["backtest_store"].create_backtest(
            BacktestCreateRequest(
                dataset_id=sample_dataset.id,
                strategy_config=StrategyConfig(
                    name="buy_favorite",
                    entry_days_to_exp=90.0,
                    exit_rule="expiry",
                    position_size=1.0,
                ),
            )
        )

        # Mark as running
        from polymkt.models.schemas import BacktestUpdateRequest
        stores["backtest_store"].update_backtest(
            backtest.id, BacktestUpdateRequest(status="running")
        )

        with pytest.raises(ValueError, match="not pending"):
            backtest_engine.execute(backtest.id)

    def test_execute_backtest_fails_without_signals(
        self,
        backtest_engine,
        stores,
        sample_dataset,
    ):
        """Verify execution fails when no favorite signals exist."""
        backtest = stores["backtest_store"].create_backtest(
            BacktestCreateRequest(
                dataset_id=sample_dataset.id,
                strategy_config=StrategyConfig(
                    name="buy_favorite",
                    entry_days_to_exp=90.0,
                    exit_rule="expiry",
                    position_size=1.0,
                ),
            )
        )

        result = backtest_engine.execute(backtest.id)

        assert result.status == "failed"
        assert "No favorite signals found" in result.error_message


# =============================================================================
# Test reproducibility
# =============================================================================


class TestBacktestReproducibility:
    """Tests for backtest reproducibility."""

    def test_backtest_produces_reproducible_results(
        self,
        backtest_engine,
        stores,
        sample_dataset,
        sample_election_group,
        sample_favorite_signal,
    ):
        """Verify running the same backtest twice produces identical results."""
        config = StrategyConfig(
            name="buy_favorite",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            fee_rate=0.01,
            slippage_rate=0.005,
            position_size=1.0,
        )

        # Run first backtest
        backtest1 = stores["backtest_store"].create_backtest(
            BacktestCreateRequest(
                dataset_id=sample_dataset.id,
                strategy_config=config,
            )
        )
        result1 = backtest_engine.execute(backtest1.id)

        # Run second backtest with same config
        backtest2 = stores["backtest_store"].create_backtest(
            BacktestCreateRequest(
                dataset_id=sample_dataset.id,
                strategy_config=config,
            )
        )
        result2 = backtest_engine.execute(backtest2.id)

        # Compare metrics
        assert result1.metrics.total_pnl == result2.metrics.total_pnl
        assert result1.metrics.trade_count == result2.metrics.trade_count
        assert result1.metrics.win_rate == result2.metrics.win_rate

        # Compare trades
        assert len(result1.trades) == len(result2.trades)
        for t1, t2 in zip(result1.trades, result2.trades):
            assert t1.market_id == t2.market_id
            assert t1.entry_price == t2.entry_price
            assert t1.exit_price == t2.exit_price
            assert t1.pnl == t2.pnl


# =============================================================================
# Test API endpoint
# =============================================================================


class TestBacktestExecuteAPI:
    """Tests for the backtest execution API endpoint."""

    @pytest.fixture
    def client(self, temp_dir, sample_markets_parquet, sample_trades_parquet, sample_order_filled_parquet):
        """Create test client with configured settings."""
        from polymkt.config import settings

        # Override settings for tests
        settings.parquet_dir = temp_dir / "parquet"
        settings.duckdb_path = temp_dir / "test.duckdb"
        settings.metadata_db_path = temp_dir / "metadata.db"

        return TestClient(app)

    def test_execute_backtest_endpoint(
        self,
        client,
        temp_dir,
    ):
        """Test the execute backtest API endpoint."""
        from polymkt.config import settings

        # Create required stores
        dataset_store = DatasetStore(settings.metadata_db_path)
        backtest_store = BacktestStore(settings.metadata_db_path)
        election_group_store = ElectionGroupStore(settings.metadata_db_path)
        favorite_signal_store = FavoriteSignalStore(settings.metadata_db_path)

        # Create dataset
        dataset = dataset_store.create_dataset(
            DatasetCreateRequest(
                name="Test Dataset",
                market_ids=["market_trump", "market_biden", "market_harris"],
            )
        )

        # Create election group
        group = election_group_store.create_group(
            name="2024 Presidential",
            market_ids=["market_trump", "market_biden", "market_harris"],
        )

        # Create favorite signal
        signal = FavoriteSignal(
            election_group_id=group["id"],
            election_group_name=group["name"],
            favorite_market_id="market_trump",
            favorite_price=0.55,
            favorite_question="Will Trump win?",
            snapshot_days_to_exp=90.0,
            computed_at=datetime.now(timezone.utc),
            all_market_prices={
                "market_trump": 0.55,
                "market_biden": 0.30,
                "market_harris": 0.15,
            },
        )
        favorite_signal_store.save_signals([signal])

        # Create backtest
        backtest = backtest_store.create_backtest(
            BacktestCreateRequest(
                dataset_id=dataset.id,
                strategy_config=StrategyConfig(
                    name="buy_favorite",
                    entry_days_to_exp=90.0,
                    exit_rule="expiry",
                    position_size=1.0,
                ),
            )
        )

        # Execute via API
        response = client.post(f"/api/backtests/{backtest.id}/execute")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["metrics"] is not None
        assert data["trades"] is not None

    def test_execute_backtest_not_found(self, client):
        """Test execute endpoint returns 404 for non-existent backtest."""
        response = client.post("/api/backtests/nonexistent-id/execute")

        assert response.status_code == 404

    def test_execute_backtest_without_signals(
        self,
        client,
        temp_dir,
    ):
        """Test execute endpoint returns error when no signals computed."""
        from polymkt.config import settings

        # Create dataset and backtest without signals
        dataset_store = DatasetStore(settings.metadata_db_path)
        backtest_store = BacktestStore(settings.metadata_db_path)

        dataset = dataset_store.create_dataset(
            DatasetCreateRequest(
                name="Test Dataset",
                market_ids=["market_trump"],
            )
        )

        backtest = backtest_store.create_backtest(
            BacktestCreateRequest(
                dataset_id=dataset.id,
                strategy_config=StrategyConfig(
                    name="buy_favorite",
                    entry_days_to_exp=90.0,
                    exit_rule="expiry",
                    position_size=1.0,
                ),
            )
        )

        # Execute via API - should fail due to missing signals
        response = client.post(f"/api/backtests/{backtest.id}/execute")

        assert response.status_code == 500
        assert "No favorite signals found" in response.json()["detail"]
