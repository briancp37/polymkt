"""Tests for the Backtesting Agent natural language strategy execution feature."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polymkt.agents.backtesting_agent import (
    BacktestAgentRequest,
    BacktestingAgent,
    StrategyParser,
)
from polymkt.models.schemas import DatasetCreateRequest
from polymkt.signals.favorites import FavoriteSignalStore
from polymkt.storage.backtests import BacktestStore
from polymkt.storage.datasets import DatasetStore
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.election_groups import ElectionGroupStore


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        parquet_dir = tmp_path / "parquet"
        parquet_dir.mkdir()
        duckdb_path = tmp_path / "data.duckdb"
        db_path = tmp_path / "metadata.db"
        yield tmp_path, parquet_dir, duckdb_path, db_path


@pytest.fixture
def sample_data():
    """Create sample markets and trades data."""
    base_time = datetime.now(timezone.utc) + timedelta(days=90)
    created_time = datetime.now(timezone.utc) - timedelta(days=7)

    markets = pd.DataFrame({
        "id": ["market1", "market2", "market3", "market4"],
        "question": [
            "Will Trump win 2024?",
            "Will Biden win 2024?",
            "Will Lakers win championship?",
            "Will Bitcoin hit 100k?",
        ],
        "category": ["politics", "politics", "sports", "crypto"],
        "tags": [
            ["election", "2024"],
            ["election", "2024"],
            ["nba", "championship"],
            ["crypto", "bitcoin"],
        ],
        "closed_time": [base_time, base_time, base_time + timedelta(days=30), None],
        "event_id": ["election2024", "election2024", "nba2024", "crypto1"],
        "created_at": [created_time, created_time, created_time, created_time],
        "description": [
            "Trump 2024 prediction",
            "Biden 2024 prediction",
            "Lakers championship prediction",
            "Bitcoin price prediction",
        ],
        # Required columns for v_markets view
        "answer1": ["Yes", "Yes", "Yes", "Yes"],
        "answer2": ["No", "No", "No", "No"],
        "neg_risk": [False, False, False, False],
        "market_slug": ["trump-2024", "biden-2024", "lakers-champ", "btc-100k"],
        "token1": ["token1a", "token1b", "token1c", "token1d"],
        "token2": ["token2a", "token2b", "token2c", "token2d"],
        "condition_id": ["cond1", "cond2", "cond3", "cond4"],
        "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        "ticker": ["TRUMP", "BIDEN", "LAKERS", "BTC"],
    })

    # Create trades at ~90 days to expiry
    trade_time = datetime.now(timezone.utc)
    trades = pd.DataFrame({
        "timestamp": [trade_time, trade_time, trade_time, trade_time],
        "market_id": ["market1", "market2", "market3", "market4"],
        "price": [0.65, 0.35, 0.55, 0.40],
        "usd_amount": [100.0, 100.0, 100.0, 100.0],
        "token_amount": [100.0, 100.0, 100.0, 100.0],
        "transaction_hash": ["tx1", "tx2", "tx3", "tx4"],
        "maker": ["0x1234", "0x1234", "0x1234", "0x1234"],
        "taker": ["0x5678", "0x5678", "0x5678", "0x5678"],
        # Additional required columns for v_trades view
        "nonusdc_side": ["maker", "maker", "maker", "maker"],
        "maker_direction": ["buy", "buy", "buy", "buy"],
        "taker_direction": ["sell", "sell", "sell", "sell"],
        "asset_id": ["asset1", "asset2", "asset3", "asset4"],
        "fee_rate_bps": [10, 10, 10, 10],
    })

    return markets, trades


@pytest.fixture
def setup_test_env(temp_dirs, sample_data):
    """Set up a full test environment with markets, trades, and election groups."""
    tmp_path, parquet_dir, duckdb_path, db_path = temp_dirs
    markets, trades = sample_data

    # Write parquet files
    markets.to_parquet(parquet_dir / "markets.parquet")
    trades.to_parquet(parquet_dir / "trades.parquet")

    # Create order_filled parquet (required by DuckDB view definitions)
    order_filled = pa.table({
        "timestamp": [datetime.now(timezone.utc)],
        "maker": ["0xmaker"],
        "maker_asset_id": ["asset1"],
        "maker_amount_filled": [100.0],
        "taker": ["0xtaker"],
        "taker_asset_id": ["asset2"],
        "taker_amount_filled": [100.0],
        "transaction_hash": ["0xhash1"],
    })
    pq.write_table(order_filled, parquet_dir / "order_filled.parquet")

    # Create DuckDB layer
    duckdb_layer = DuckDBLayer(duckdb_path, parquet_dir, partitioned=False)
    duckdb_layer.create_views()

    # Create stores
    dataset_store = DatasetStore(db_path)
    backtest_store = BacktestStore(db_path)
    election_group_store = ElectionGroupStore(db_path)
    favorite_signal_store = FavoriteSignalStore(db_path)

    # Create an election group with the election markets
    election_group_store.create_group(
        name="2024 Presidential Election",
        description="Presidential candidates",
        market_ids=["market1", "market2"],
    )

    # Create a dataset with all politics markets
    dataset = dataset_store.create_dataset(
        DatasetCreateRequest(
            name="Election Markets",
            description="Markets about the 2024 election",
            market_ids=["market1", "market2"],
        )
    )

    yield {
        "duckdb_layer": duckdb_layer,
        "dataset_store": dataset_store,
        "backtest_store": backtest_store,
        "election_group_store": election_group_store,
        "favorite_signal_store": favorite_signal_store,
        "dataset_id": dataset.id,
        "db_path": db_path,
    }

    duckdb_layer.close()


class TestStrategyParser:
    """Tests for the StrategyParser class."""

    def test_parse_basic_strategy(self):
        """Test parsing a basic strategy."""
        parser = StrategyParser()
        result = parser.parse("buy favorite 90 days out, hold to expiry")

        assert result.entry_days_to_exp == 90.0
        assert result.favorite_rule == "max_yes_price"
        assert result.exit_rule == "expiry"

    def test_parse_extracts_days(self):
        """Test parsing extracts entry days."""
        parser = StrategyParser()

        result = parser.parse("buy favorite 60 days before expiration")
        assert result.entry_days_to_exp == 60.0

        result = parser.parse("buy at 30 days to exp")
        assert result.entry_days_to_exp == 30.0

    def test_parse_extracts_weeks(self):
        """Test parsing extracts weeks and converts to days."""
        parser = StrategyParser()
        result = parser.parse("buy favorite 12 weeks before close")

        assert result.entry_days_to_exp == 84.0  # 12 * 7

    def test_parse_extracts_months(self):
        """Test parsing extracts months and converts to days."""
        parser = StrategyParser()
        result = parser.parse("buy leader 3 months out")

        assert result.entry_days_to_exp == 90.0  # 3 * 30

    def test_parse_favorite_rule_buy_favorite(self):
        """Test parsing recognizes buy favorite."""
        parser = StrategyParser()
        result = parser.parse("buy the favorite at 90 days")

        assert result.favorite_rule == "max_yes_price"

    def test_parse_favorite_rule_highest_yes(self):
        """Test parsing recognizes highest YES price."""
        parser = StrategyParser()
        result = parser.parse("buy highest yes price 90 days out")

        assert result.favorite_rule == "max_yes_price"

    def test_parse_favorite_rule_underdog(self):
        """Test parsing recognizes underdog (lowest YES price)."""
        parser = StrategyParser()
        result = parser.parse("buy underdog at 90 days")

        assert result.favorite_rule == "min_yes_price"

    def test_parse_exit_rule_expiry(self):
        """Test parsing recognizes hold to expiry."""
        parser = StrategyParser()
        result = parser.parse("buy favorite, hold until expiration")

        assert result.exit_rule == "expiry"

    def test_parse_exit_rule_take_profit(self):
        """Test parsing recognizes take profit."""
        parser = StrategyParser()
        result = parser.parse("buy favorite, take profit at 90%")

        assert result.exit_rule == "take_profit"
        assert result.extra_params is not None
        assert result.extra_params.get("take_profit_pct") == 90.0

    def test_parse_fee_rate(self):
        """Test parsing extracts fee rate."""
        parser = StrategyParser()
        result = parser.parse("buy favorite 90 days out with 1% fee")

        assert result.fee_rate == 0.01

    def test_parse_slippage_rate(self):
        """Test parsing extracts slippage rate."""
        parser = StrategyParser()
        result = parser.parse("buy favorite with 0.5% slippage")

        assert result.slippage_rate == 0.005

    def test_parse_uses_defaults(self):
        """Test parsing uses defaults when not specified."""
        parser = StrategyParser()
        result = parser.parse("buy something")

        assert result.entry_days_to_exp == 90.0  # default
        assert result.exit_rule == "expiry"  # default
        assert result.favorite_rule == "max_yes_price"  # default
        assert result.fee_rate == 0.0
        assert result.slippage_rate == 0.0

    def test_generates_strategy_name(self):
        """Test strategy name generation."""
        parser = StrategyParser()
        result = parser.parse("buy favorite 90 days out, hold to expiry")

        assert "favorite" in result.name
        assert "90" in result.name
        assert "expiry" in result.name

    def test_parse_horizon_syntax(self):
        """Test parsing with 'horizon' syntax."""
        parser = StrategyParser()
        result = parser.parse("buy favorite with 60 days horizon")

        assert result.entry_days_to_exp == 60.0


class TestBacktestingAgent:
    """Tests for the BacktestingAgent class."""

    def test_prepare_backtest_returns_confirmation(self, setup_test_env):
        """Test prepare_backtest returns confirmation."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id=setup_test_env["dataset_id"],
            natural_language_strategy="buy favorite 90 days out, hold to expiry",
        )
        confirmation = agent.prepare_backtest(request)

        assert confirmation.session_id is not None
        assert confirmation.dataset_id == setup_test_env["dataset_id"]
        assert confirmation.market_count == 2
        assert confirmation.parsed_strategy.entry_days_to_exp == 90.0

    def test_prepare_backtest_stores_session(self, setup_test_env):
        """Test prepare_backtest stores the session."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id=setup_test_env["dataset_id"],
            natural_language_strategy="buy favorite 90 days out",
        )
        confirmation = agent.prepare_backtest(request)

        session = agent.get_session(confirmation.session_id)
        assert session is not None
        assert session.session_id == confirmation.session_id

    def test_prepare_backtest_with_overrides(self, setup_test_env):
        """Test prepare_backtest with fee/slippage overrides."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id=setup_test_env["dataset_id"],
            natural_language_strategy="buy favorite 90 days out",
            fee_rate=0.02,
            slippage_rate=0.01,
        )
        confirmation = agent.prepare_backtest(request)

        assert confirmation.parsed_strategy.fee_rate == 0.02
        assert confirmation.parsed_strategy.slippage_rate == 0.01

    def test_prepare_backtest_generates_summary(self, setup_test_env):
        """Test prepare_backtest generates a human-readable summary."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id=setup_test_env["dataset_id"],
            natural_language_strategy="buy favorite 90 days out, hold to expiry",
        )
        confirmation = agent.prepare_backtest(request)

        assert "Strategy:" in confirmation.summary
        assert "Dataset:" in confirmation.summary
        assert "Entry:" in confirmation.summary
        assert "90" in confirmation.summary

    def test_prepare_backtest_warns_missing_signals(self, setup_test_env):
        """Test prepare_backtest warns if signals are missing."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id=setup_test_env["dataset_id"],
            natural_language_strategy="buy favorite 90 days out",
        )
        confirmation = agent.prepare_backtest(request)

        # Should have a warning about missing signals
        assert len(confirmation.warnings) > 0
        assert any("signal" in w.lower() for w in confirmation.warnings)

    def test_modify_strategy_updates_entry_days(self, setup_test_env):
        """Test modify_strategy updates entry days."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id=setup_test_env["dataset_id"],
            natural_language_strategy="buy favorite 90 days out",
        )
        confirmation = agent.prepare_backtest(request)

        # Modify to 60 days
        updated = agent.modify_strategy(
            session_id=confirmation.session_id,
            entry_days_to_exp=60.0,
        )

        assert updated.parsed_strategy.entry_days_to_exp == 60.0
        assert "60" in updated.summary

    def test_modify_strategy_updates_fee_rate(self, setup_test_env):
        """Test modify_strategy updates fee rate."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id=setup_test_env["dataset_id"],
            natural_language_strategy="buy favorite 90 days out",
        )
        confirmation = agent.prepare_backtest(request)

        updated = agent.modify_strategy(
            session_id=confirmation.session_id,
            fee_rate=0.03,
        )

        assert updated.parsed_strategy.fee_rate == 0.03

    def test_modify_invalid_session_raises(self, setup_test_env):
        """Test modify_strategy raises for invalid session."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        with pytest.raises(ValueError, match="Session not found"):
            agent.modify_strategy(
                session_id="invalid-session",
                entry_days_to_exp=60.0,
            )

    def test_prepare_backtest_invalid_dataset_raises(self, setup_test_env):
        """Test prepare_backtest raises for invalid dataset."""
        agent = BacktestingAgent(
            duckdb_layer=setup_test_env["duckdb_layer"],
            dataset_store=setup_test_env["dataset_store"],
            backtest_store=setup_test_env["backtest_store"],
            election_group_store=setup_test_env["election_group_store"],
            favorite_signal_store=setup_test_env["favorite_signal_store"],
        )

        request = BacktestAgentRequest(
            dataset_id="invalid-dataset-id",
            natural_language_strategy="buy favorite 90 days out",
        )

        from polymkt.storage.datasets import DatasetNotFoundError
        with pytest.raises(DatasetNotFoundError):
            agent.prepare_backtest(request)


class TestBacktestingAgentSchemas:
    """Tests for the Backtesting Agent API schema validation."""

    def test_request_schema_validates_empty_strategy(self):
        """Test request schema rejects empty strategies."""
        from pydantic import ValidationError
        from polymkt.models.schemas import BacktestAgentRequestSchema

        with pytest.raises(ValidationError):
            BacktestAgentRequestSchema(
                dataset_id="test-id",
                natural_language_strategy="",
            )

    def test_request_schema_validates_short_strategy(self):
        """Test request schema rejects too-short strategies."""
        from pydantic import ValidationError
        from polymkt.models.schemas import BacktestAgentRequestSchema

        with pytest.raises(ValidationError):
            BacktestAgentRequestSchema(
                dataset_id="test-id",
                natural_language_strategy="ab",  # too short
            )

    def test_request_schema_accepts_valid_strategy(self):
        """Test request schema accepts valid strategies."""
        from polymkt.models.schemas import BacktestAgentRequestSchema

        request = BacktestAgentRequestSchema(
            dataset_id="test-id",
            natural_language_strategy="buy favorite 90 days out, hold to expiry",
        )
        assert request.natural_language_strategy == "buy favorite 90 days out, hold to expiry"

    def test_request_schema_validates_fee_rate_bounds(self):
        """Test request schema validates fee rate bounds."""
        from pydantic import ValidationError
        from polymkt.models.schemas import BacktestAgentRequestSchema

        with pytest.raises(ValidationError):
            BacktestAgentRequestSchema(
                dataset_id="test-id",
                natural_language_strategy="buy favorite",
                fee_rate=1.5,  # > 1.0
            )

    def test_modify_schema_validates_entry_days_positive(self):
        """Test modify schema validates entry days is positive."""
        from pydantic import ValidationError
        from polymkt.models.schemas import BacktestAgentModifyRequestSchema

        with pytest.raises(ValidationError):
            BacktestAgentModifyRequestSchema(
                session_id="test-session",
                entry_days_to_exp=-10,  # negative
            )

    def test_parsed_strategy_schema_structure(self):
        """Test parsed strategy schema has expected structure."""
        from polymkt.models.schemas import ParsedStrategySchema

        strategy = ParsedStrategySchema(
            name="buy_favorite_90d_expiry",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            favorite_rule="max_yes_price",
            fee_rate=0.01,
            slippage_rate=0.005,
            position_size=1.0,
        )

        assert strategy.name == "buy_favorite_90d_expiry"
        assert strategy.entry_days_to_exp == 90.0
        assert strategy.exit_rule == "expiry"

    def test_confirmation_schema_structure(self):
        """Test confirmation schema has expected structure."""
        from polymkt.models.schemas import (
            ParsedStrategySchema,
            StrategyConfirmationSchema,
        )

        parsed = ParsedStrategySchema(
            name="buy_favorite_90d_expiry",
            entry_days_to_exp=90.0,
            exit_rule="expiry",
            favorite_rule="max_yes_price",
            fee_rate=0.0,
            slippage_rate=0.0,
            position_size=1.0,
        )

        confirmation = StrategyConfirmationSchema(
            session_id="test-session",
            dataset_id="dataset-123",
            dataset_name="Test Dataset",
            market_count=10,
            parsed_strategy=parsed,
            summary="Test summary",
            warnings=["Warning 1"],
        )

        assert confirmation.session_id == "test-session"
        assert confirmation.market_count == 10
        assert len(confirmation.warnings) == 1


class TestBacktestingAgentParsingEdgeCases:
    """Tests for parsing edge cases."""

    def test_parse_dte_notation(self):
        """Test parsing DTE (days to expiry) notation."""
        parser = StrategyParser()
        result = parser.parse("buy favorite at 90 DTE")

        assert result.entry_days_to_exp == 90.0

    def test_parse_front_runner(self):
        """Test parsing 'front-runner' as favorite."""
        parser = StrategyParser()
        result = parser.parse("buy front-runner 90 days out")

        assert result.favorite_rule == "max_yes_price"

    def test_parse_best_odds(self):
        """Test parsing 'best odds' as favorite."""
        parser = StrategyParser()
        result = parser.parse("buy best odds at 90 days")

        assert result.favorite_rule == "max_yes_price"

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        parser = StrategyParser()

        result = parser.parse("BUY FAVORITE 90 DAYS OUT")
        assert result.entry_days_to_exp == 90.0
        assert result.favorite_rule == "max_yes_price"

        result = parser.parse("Buy Highest YES Price 60 Days Out")
        assert result.entry_days_to_exp == 60.0
        assert result.favorite_rule == "max_yes_price"

    def test_parse_complex_strategy(self):
        """Test parsing a complex strategy with multiple components."""
        parser = StrategyParser()
        result = parser.parse(
            "buy the favorite 60 days before expiration, hold to expiry, with 0.5% fee and 0.1% slippage"
        )

        assert result.entry_days_to_exp == 60.0
        assert result.favorite_rule == "max_yes_price"
        assert result.exit_rule == "expiry"
        assert result.fee_rate == 0.005
        assert result.slippage_rate == 0.001
