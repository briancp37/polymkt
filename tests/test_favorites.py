"""Tests for the favorite signal computation feature."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app
from polymkt.signals.favorites import (
    FavoriteComputeResult,
    FavoriteSignal,
    FavoriteSignalStore,
    SnapshotPrice,
    compute_favorites_for_groups,
    compute_snapshot_prices,
    select_favorite,
)
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.election_groups import ElectionGroupStore


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def temp_db(temp_dir: Path) -> Path:
    """Create a temporary database file path."""
    return temp_dir / "test.db"


@pytest.fixture
def parquet_dir(temp_dir: Path) -> Path:
    """Create a temporary parquet directory."""
    pdir = temp_dir / "parquet"
    pdir.mkdir()
    return pdir


def create_test_markets(parquet_dir: Path, markets: list[dict]) -> None:
    """Create a test markets.parquet file."""
    df = pd.DataFrame(markets)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_dir / "markets.parquet")


def create_test_trades(parquet_dir: Path, trades: list[dict]) -> None:
    """Create a test trades.parquet file."""
    df = pd.DataFrame(trades)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_dir / "trades.parquet")


def create_test_order_filled(parquet_dir: Path, orders: list[dict] | None = None) -> None:
    """Create a test order_filled.parquet file."""
    if orders is None:
        # Create minimal order_filled file
        orders = [
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "maker": "0x1111",
                "maker_asset_id": "asset1",
                "maker_amount_filled": 100.0,
                "taker": "0x2222",
                "taker_asset_id": "asset2",
                "taker_amount_filled": 100.0,
                "transaction_hash": "0xhash1",
            }
        ]
    df = pd.DataFrame(orders)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_dir / "order_filled.parquet")


@pytest.fixture
def sample_election_data(parquet_dir: Path, temp_db: Path) -> tuple[DuckDBLayer, ElectionGroupStore]:
    """Create sample election group data with trades at 90-day snapshot.

    Creates an election group with 3 candidates:
    - market-biden: price 0.45 at 90 days
    - market-trump: price 0.50 at 90 days (FAVORITE)
    - market-rfk: price 0.05 at 90 days
    """
    # Markets close 100 days from base time
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    closed_time = base_time + timedelta(days=100)

    # Create markets
    markets = [
        {
            "id": "market-biden",
            "question": "Will Biden win the 2024 election?",
            "created_at": base_time - timedelta(days=200),
            "closed_time": closed_time,
            "category": "politics",
            "description": "Biden market",
            "answer1": "Yes",
            "answer2": "No",
            "neg_risk": False,
            "market_slug": "biden-2024",
            "token1": "token1",
            "token2": "token2",
            "condition_id": "cond1",
            "volume": 1000000.0,
            "ticker": "BIDEN",
            "event_id": "event-2024-election",
            "tags": None,
        },
        {
            "id": "market-trump",
            "question": "Will Trump win the 2024 election?",
            "created_at": base_time - timedelta(days=200),
            "closed_time": closed_time,
            "category": "politics",
            "description": "Trump market",
            "answer1": "Yes",
            "answer2": "No",
            "neg_risk": False,
            "market_slug": "trump-2024",
            "token1": "token3",
            "token2": "token4",
            "condition_id": "cond2",
            "volume": 2000000.0,
            "ticker": "TRUMP",
            "event_id": "event-2024-election",
            "tags": None,
        },
        {
            "id": "market-rfk",
            "question": "Will RFK Jr win the 2024 election?",
            "created_at": base_time - timedelta(days=200),
            "closed_time": closed_time,
            "category": "politics",
            "description": "RFK Jr market",
            "answer1": "Yes",
            "answer2": "No",
            "neg_risk": False,
            "market_slug": "rfk-2024",
            "token1": "token5",
            "token2": "token6",
            "condition_id": "cond3",
            "volume": 100000.0,
            "ticker": "RFK",
            "event_id": "event-2024-election",
            "tags": None,
        },
    ]

    # Create trades at ~90 days to expiry (base_time + 10 days = 90 days before close)
    trade_time_90 = base_time + timedelta(days=10)
    trades = [
        # Biden trades - price 0.45
        {
            "timestamp": trade_time_90,
            "market_id": "market-biden",
            "maker": "0x1111",
            "taker": "0x2222",
            "nonusdc_side": "YES",
            "maker_direction": "sell",
            "taker_direction": "buy",
            "price": 0.45,
            "usd_amount": 1000.0,
            "token_amount": 2222.22,
            "transaction_hash": "0xhash1",
        },
        # Trump trades - price 0.50 (HIGHEST = FAVORITE)
        {
            "timestamp": trade_time_90,
            "market_id": "market-trump",
            "maker": "0x3333",
            "taker": "0x4444",
            "nonusdc_side": "YES",
            "maker_direction": "sell",
            "taker_direction": "buy",
            "price": 0.50,
            "usd_amount": 2000.0,
            "token_amount": 4000.0,
            "transaction_hash": "0xhash2",
        },
        # RFK trades - price 0.05
        {
            "timestamp": trade_time_90,
            "market_id": "market-rfk",
            "maker": "0x5555",
            "taker": "0x6666",
            "nonusdc_side": "YES",
            "maker_direction": "sell",
            "taker_direction": "buy",
            "price": 0.05,
            "usd_amount": 100.0,
            "token_amount": 2000.0,
            "transaction_hash": "0xhash3",
        },
    ]

    create_test_markets(parquet_dir, markets)
    create_test_trades(parquet_dir, trades)
    create_test_order_filled(parquet_dir)  # Add order_filled.parquet

    # Create DuckDB layer
    duckdb_path = temp_db.parent / "test.duckdb"
    duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir)
    duckdb_layer.create_views()

    # Create election group
    group_store = ElectionGroupStore(temp_db)
    group_store.create_group(
        name="2024 Presidential Election",
        description="US 2024 presidential election candidates",
        market_ids=["market-biden", "market-trump", "market-rfk"],
    )

    return duckdb_layer, group_store


# =============================================================================
# Unit Tests: select_favorite
# =============================================================================


class TestSelectFavorite:
    """Test the select_favorite function."""

    def test_select_favorite_returns_highest_price(self) -> None:
        """Test that select_favorite returns the market with highest YES price."""
        snapshot_prices = {
            "market-a": SnapshotPrice(
                market_id="market-a",
                price=0.3,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                days_to_exp=90.0,
            ),
            "market-b": SnapshotPrice(
                market_id="market-b",
                price=0.7,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                days_to_exp=90.0,
            ),
            "market-c": SnapshotPrice(
                market_id="market-c",
                price=0.5,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                days_to_exp=90.0,
            ),
        }

        result = select_favorite(snapshot_prices)
        assert result is not None
        favorite_id, favorite_price = result
        assert favorite_id == "market-b"
        assert favorite_price == 0.7

    def test_select_favorite_handles_ties_deterministically(self) -> None:
        """Test that ties are broken by market_id (alphabetically)."""
        snapshot_prices = {
            "market-c": SnapshotPrice(
                market_id="market-c",
                price=0.5,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                days_to_exp=90.0,
            ),
            "market-a": SnapshotPrice(
                market_id="market-a",
                price=0.5,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                days_to_exp=90.0,
            ),
            "market-b": SnapshotPrice(
                market_id="market-b",
                price=0.5,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                days_to_exp=90.0,
            ),
        }

        result = select_favorite(snapshot_prices)
        assert result is not None
        favorite_id, favorite_price = result
        # market-a should win due to alphabetical tie-break
        assert favorite_id == "market-a"
        assert favorite_price == 0.5

    def test_select_favorite_empty_returns_none(self) -> None:
        """Test that select_favorite returns None for empty input."""
        result = select_favorite({})
        assert result is None

    def test_select_favorite_single_market(self) -> None:
        """Test select_favorite with single market."""
        snapshot_prices = {
            "market-only": SnapshotPrice(
                market_id="market-only",
                price=0.6,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                days_to_exp=90.0,
            ),
        }

        result = select_favorite(snapshot_prices)
        assert result is not None
        favorite_id, favorite_price = result
        assert favorite_id == "market-only"
        assert favorite_price == 0.6


# =============================================================================
# Unit Tests: compute_snapshot_prices
# =============================================================================


class TestComputeSnapshotPrices:
    """Test snapshot price computation."""

    def test_compute_snapshot_prices_returns_last_trade(
        self, sample_election_data: tuple[DuckDBLayer, ElectionGroupStore]
    ) -> None:
        """Test that compute_snapshot_prices returns the last trade price at snapshot."""
        duckdb_layer, _ = sample_election_data

        prices = compute_snapshot_prices(
            duckdb_layer,
            market_ids=["market-biden", "market-trump", "market-rfk"],
            target_days_to_exp=90.0,
            tolerance=0.5,
        )

        assert len(prices) == 3
        assert prices["market-biden"].price == 0.45
        assert prices["market-trump"].price == 0.50
        assert prices["market-rfk"].price == 0.05

    def test_compute_snapshot_prices_empty_market_list(
        self, sample_election_data: tuple[DuckDBLayer, ElectionGroupStore]
    ) -> None:
        """Test that empty market list returns empty dict."""
        duckdb_layer, _ = sample_election_data

        prices = compute_snapshot_prices(
            duckdb_layer,
            market_ids=[],
            target_days_to_exp=90.0,
            tolerance=0.5,
        )

        assert prices == {}

    def test_compute_snapshot_prices_no_trades_at_snapshot(
        self, sample_election_data: tuple[DuckDBLayer, ElectionGroupStore]
    ) -> None:
        """Test markets with no trades at snapshot return empty."""
        duckdb_layer, _ = sample_election_data

        # Use a snapshot where there are no trades (e.g., 30 days)
        prices = compute_snapshot_prices(
            duckdb_layer,
            market_ids=["market-biden", "market-trump", "market-rfk"],
            target_days_to_exp=30.0,  # No trades at 30 days
            tolerance=0.5,
        )

        assert prices == {}


# =============================================================================
# Integration Tests: compute_favorites_for_groups
# =============================================================================


class TestComputeFavoritesForGroups:
    """Test the full favorite computation pipeline."""

    def test_compute_favorites_identifies_correct_favorite(
        self, sample_election_data: tuple[DuckDBLayer, ElectionGroupStore]
    ) -> None:
        """Test that compute_favorites identifies Trump as favorite at 90 days."""
        duckdb_layer, group_store = sample_election_data

        result = compute_favorites_for_groups(
            duckdb_layer=duckdb_layer,
            election_group_store=group_store,
            target_days_to_exp=90.0,
            tolerance=0.5,
        )

        assert result.groups_processed == 1
        assert result.groups_with_data == 1
        assert result.groups_without_data == 0
        assert len(result.signals) == 1

        signal = result.signals[0]
        assert signal.favorite_market_id == "market-trump"
        assert signal.favorite_price == 0.50
        assert signal.election_group_name == "2024 Presidential Election"
        assert len(signal.all_market_prices) == 3

    def test_compute_favorites_includes_all_prices(
        self, sample_election_data: tuple[DuckDBLayer, ElectionGroupStore]
    ) -> None:
        """Test that all_market_prices includes all markets in group."""
        duckdb_layer, group_store = sample_election_data

        result = compute_favorites_for_groups(
            duckdb_layer=duckdb_layer,
            election_group_store=group_store,
            target_days_to_exp=90.0,
            tolerance=0.5,
        )

        signal = result.signals[0]
        assert "market-biden" in signal.all_market_prices
        assert "market-trump" in signal.all_market_prices
        assert "market-rfk" in signal.all_market_prices
        assert signal.all_market_prices["market-biden"] == 0.45
        assert signal.all_market_prices["market-trump"] == 0.50
        assert signal.all_market_prices["market-rfk"] == 0.05

    def test_compute_favorites_handles_empty_groups(
        self, sample_election_data: tuple[DuckDBLayer, ElectionGroupStore]
    ) -> None:
        """Test that empty groups are handled gracefully."""
        duckdb_layer, group_store = sample_election_data

        # Create an additional empty group (reuses existing parquet files)
        group_store.create_group(name="Empty Group", market_ids=[])

        # Get just the empty group
        groups, _ = group_store.list_groups()
        empty_group_id = [g["id"] for g in groups if g["name"] == "Empty Group"][0]

        result = compute_favorites_for_groups(
            duckdb_layer=duckdb_layer,
            election_group_store=group_store,
            target_days_to_exp=90.0,
            tolerance=0.5,
            group_ids=[empty_group_id],
        )

        assert result.groups_processed == 1
        assert result.groups_with_data == 0
        assert result.groups_without_data == 1
        assert len(result.signals) == 0


# =============================================================================
# Unit Tests: FavoriteSignalStore
# =============================================================================


class TestFavoriteSignalStore:
    """Test the favorite signal persistence store."""

    def test_store_creates_table(self, temp_db: Path) -> None:
        """Test that store initialization creates the required table."""
        import sqlite3

        store = FavoriteSignalStore(temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='favorite_signals'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_save_and_get_signals(self, temp_db: Path) -> None:
        """Test saving and retrieving signals."""
        store = FavoriteSignalStore(temp_db)

        signals = [
            FavoriteSignal(
                election_group_id="group-1",
                election_group_name="Test Group",
                favorite_market_id="market-a",
                favorite_price=0.65,
                favorite_question="Test question?",
                snapshot_days_to_exp=90.0,
                computed_at=datetime.now(timezone.utc),
                all_market_prices={"market-a": 0.65, "market-b": 0.35},
            )
        ]

        saved = store.save_signals(signals)
        assert saved == 1

        retrieved = store.get_signals_for_snapshot(90.0)
        assert len(retrieved) == 1
        assert retrieved[0]["favorite_market_id"] == "market-a"
        assert retrieved[0]["favorite_price"] == 0.65
        assert retrieved[0]["all_market_prices"] == {"market-a": 0.65, "market-b": 0.35}

    def test_get_signal_for_group(self, temp_db: Path) -> None:
        """Test retrieving signal for a specific group."""
        store = FavoriteSignalStore(temp_db)

        signals = [
            FavoriteSignal(
                election_group_id="group-1",
                election_group_name="Group 1",
                favorite_market_id="market-a",
                favorite_price=0.65,
                favorite_question="Question 1?",
                snapshot_days_to_exp=90.0,
                computed_at=datetime.now(timezone.utc),
                all_market_prices={"market-a": 0.65},
            ),
            FavoriteSignal(
                election_group_id="group-2",
                election_group_name="Group 2",
                favorite_market_id="market-b",
                favorite_price=0.55,
                favorite_question="Question 2?",
                snapshot_days_to_exp=90.0,
                computed_at=datetime.now(timezone.utc),
                all_market_prices={"market-b": 0.55},
            ),
        ]

        store.save_signals(signals)

        signal = store.get_signal_for_group("group-1", 90.0)
        assert signal is not None
        assert signal["favorite_market_id"] == "market-a"

        signal2 = store.get_signal_for_group("group-2", 90.0)
        assert signal2 is not None
        assert signal2["favorite_market_id"] == "market-b"

        # Non-existent group
        signal3 = store.get_signal_for_group("group-nonexistent", 90.0)
        assert signal3 is None

    def test_clear_signals(self, temp_db: Path) -> None:
        """Test clearing signals for a snapshot."""
        store = FavoriteSignalStore(temp_db)

        signals = [
            FavoriteSignal(
                election_group_id="group-1",
                election_group_name="Test Group",
                favorite_market_id="market-a",
                favorite_price=0.65,
                favorite_question="Test?",
                snapshot_days_to_exp=90.0,
                computed_at=datetime.now(timezone.utc),
                all_market_prices={"market-a": 0.65},
            )
        ]

        store.save_signals(signals)
        assert len(store.get_signals_for_snapshot(90.0)) == 1

        deleted = store.clear_signals_for_snapshot(90.0)
        assert deleted == 1
        assert len(store.get_signals_for_snapshot(90.0)) == 0

    def test_list_snapshots(self, temp_db: Path) -> None:
        """Test listing all unique snapshots."""
        store = FavoriteSignalStore(temp_db)

        signals = [
            FavoriteSignal(
                election_group_id="group-1",
                election_group_name="Group 1",
                favorite_market_id="market-a",
                favorite_price=0.65,
                favorite_question="Q?",
                snapshot_days_to_exp=90.0,
                computed_at=datetime.now(timezone.utc),
                all_market_prices={"market-a": 0.65},
            ),
            FavoriteSignal(
                election_group_id="group-2",
                election_group_name="Group 2",
                favorite_market_id="market-b",
                favorite_price=0.55,
                favorite_question="Q?",
                snapshot_days_to_exp=90.0,
                computed_at=datetime.now(timezone.utc),
                all_market_prices={"market-b": 0.55},
            ),
            FavoriteSignal(
                election_group_id="group-1",
                election_group_name="Group 1",
                favorite_market_id="market-c",
                favorite_price=0.75,
                favorite_question="Q?",
                snapshot_days_to_exp=60.0,  # Different snapshot
                computed_at=datetime.now(timezone.utc),
                all_market_prices={"market-c": 0.75},
            ),
        ]

        store.save_signals(signals)

        snapshots = store.list_snapshots()
        assert len(snapshots) == 2
        # Should be ordered by snapshot_days_to_exp
        assert snapshots[0]["snapshot_days_to_exp"] == 60.0
        assert snapshots[0]["signal_count"] == 1
        assert snapshots[1]["snapshot_days_to_exp"] == 90.0
        assert snapshots[1]["signal_count"] == 2


# =============================================================================
# API Tests
# =============================================================================


@pytest.fixture
def api_client(
    temp_dir: Path, parquet_dir: Path, temp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """Create a test client with temp directories configured."""
    from polymkt import config

    # Create test data
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    closed_time = base_time + timedelta(days=100)

    markets = [
        {
            "id": "market-a",
            "question": "Test market A?",
            "created_at": base_time - timedelta(days=200),
            "closed_time": closed_time,
            "category": "test",
            "description": "Test A",
            "answer1": "Yes",
            "answer2": "No",
            "neg_risk": False,
            "market_slug": "test-a",
            "token1": "t1",
            "token2": "t2",
            "condition_id": "c1",
            "volume": 1000.0,
            "ticker": "A",
            "event_id": "event-1",
            "tags": None,
        },
        {
            "id": "market-b",
            "question": "Test market B?",
            "created_at": base_time - timedelta(days=200),
            "closed_time": closed_time,
            "category": "test",
            "description": "Test B",
            "answer1": "Yes",
            "answer2": "No",
            "neg_risk": False,
            "market_slug": "test-b",
            "token1": "t3",
            "token2": "t4",
            "condition_id": "c2",
            "volume": 2000.0,
            "ticker": "B",
            "event_id": "event-1",
            "tags": None,
        },
    ]

    trade_time_90 = base_time + timedelta(days=10)
    trades = [
        {
            "timestamp": trade_time_90,
            "market_id": "market-a",
            "maker": "0x1111",
            "taker": "0x2222",
            "nonusdc_side": "YES",
            "maker_direction": "sell",
            "taker_direction": "buy",
            "price": 0.40,
            "usd_amount": 1000.0,
            "token_amount": 2500.0,
            "transaction_hash": "0xhash1",
        },
        {
            "timestamp": trade_time_90,
            "market_id": "market-b",
            "maker": "0x3333",
            "taker": "0x4444",
            "nonusdc_side": "YES",
            "maker_direction": "sell",
            "taker_direction": "buy",
            "price": 0.60,
            "usd_amount": 2000.0,
            "token_amount": 3333.33,
            "transaction_hash": "0xhash2",
        },
    ]

    create_test_markets(parquet_dir, markets)
    create_test_trades(parquet_dir, trades)
    create_test_order_filled(parquet_dir)  # Add order_filled.parquet

    # Create election group
    group_store = ElectionGroupStore(temp_db)
    group_store.create_group(
        name="Test Election",
        market_ids=["market-a", "market-b"],
    )

    # Patch settings
    monkeypatch.setattr(config.settings, "parquet_dir", parquet_dir)
    monkeypatch.setattr(config.settings, "metadata_db_path", temp_db)
    monkeypatch.setattr(config.settings, "duckdb_path", temp_dir / "test.duckdb")

    return TestClient(app)


class TestFavoriteSignalsAPI:
    """Test API endpoints for favorite signals."""

    def test_compute_favorite_signals(self, api_client: TestClient) -> None:
        """Test POST /api/favorite-signals/compute endpoint."""
        response = api_client.post(
            "/api/favorite-signals/compute",
            json={
                "target_days_to_exp": 90.0,
                "tolerance": 0.5,
                "clear_existing": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["signals_computed"] == 1
        assert data["signals_saved"] == 1
        assert data["groups_processed"] == 1
        assert data["groups_with_data"] == 1

    def test_get_favorite_signals(self, api_client: TestClient) -> None:
        """Test GET /api/favorite-signals endpoint."""
        # First compute signals
        api_client.post(
            "/api/favorite-signals/compute",
            json={"target_days_to_exp": 90.0, "tolerance": 0.5},
        )

        # Then get them
        response = api_client.get("/api/favorite-signals?snapshot_days_to_exp=90.0")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["signals"][0]["favorite_market_id"] == "market-b"
        assert data["signals"][0]["favorite_price"] == 0.60

    def test_get_favorite_signal_for_group(self, api_client: TestClient) -> None:
        """Test GET /api/favorite-signals/group/{group_id} endpoint."""
        # First compute signals
        compute_resp = api_client.post(
            "/api/favorite-signals/compute",
            json={"target_days_to_exp": 90.0, "tolerance": 0.5},
        )
        assert compute_resp.status_code == 200

        # Get the group ID from the signals
        signals_resp = api_client.get("/api/favorite-signals?snapshot_days_to_exp=90.0")
        group_id = signals_resp.json()["signals"][0]["election_group_id"]

        # Get signal for specific group
        response = api_client.get(f"/api/favorite-signals/group/{group_id}?snapshot_days_to_exp=90.0")

        assert response.status_code == 200
        data = response.json()
        assert data["favorite_market_id"] == "market-b"

    def test_list_snapshots(self, api_client: TestClient) -> None:
        """Test GET /api/favorite-signals/snapshots endpoint."""
        # Compute signals at two different snapshots
        api_client.post(
            "/api/favorite-signals/compute",
            json={"target_days_to_exp": 90.0, "tolerance": 0.5},
        )

        response = api_client.get("/api/favorite-signals/snapshots")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    def test_clear_favorite_signals(self, api_client: TestClient) -> None:
        """Test DELETE /api/favorite-signals endpoint."""
        # First compute signals
        api_client.post(
            "/api/favorite-signals/compute",
            json={"target_days_to_exp": 90.0, "tolerance": 0.5},
        )

        # Verify they exist
        resp1 = api_client.get("/api/favorite-signals?snapshot_days_to_exp=90.0")
        assert resp1.json()["count"] == 1

        # Clear them
        response = api_client.delete("/api/favorite-signals?snapshot_days_to_exp=90.0")
        assert response.status_code == 200
        assert response.json()["deleted_count"] == 1

        # Verify they're gone
        resp2 = api_client.get("/api/favorite-signals?snapshot_days_to_exp=90.0")
        assert resp2.json()["count"] == 0


# =============================================================================
# Determinism Tests
# =============================================================================


class TestFavoriteDeterminism:
    """Test that favorite selection is deterministic."""

    def test_favorite_selection_is_reproducible(
        self, sample_election_data: tuple[DuckDBLayer, ElectionGroupStore]
    ) -> None:
        """Test that running compute multiple times gives same result."""
        duckdb_layer, group_store = sample_election_data

        # Run computation multiple times
        results = []
        for _ in range(5):
            result = compute_favorites_for_groups(
                duckdb_layer=duckdb_layer,
                election_group_store=group_store,
                target_days_to_exp=90.0,
                tolerance=0.5,
            )
            results.append(result)

        # All should have same favorite
        favorites = [r.signals[0].favorite_market_id for r in results]
        assert all(f == favorites[0] for f in favorites)

        prices = [r.signals[0].favorite_price for r in results]
        assert all(p == prices[0] for p in prices)
