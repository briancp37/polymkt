"""Backtesting Agent for natural language strategy requests.

This module provides the BacktestingAgent class that accepts natural language
strategy requests (e.g., "buy favorite 90 days out, hold to expiry") and
executes backtests on a selected dataset.

The agent uses the existing backtest engine and signal infrastructure to:
1. Parse natural language strategy descriptions into StrategyConfig
2. Display a confirmation summary of parsed rules before execution
3. Execute the backtest and return results

PRD Reference:
- "Backtesting Agent: accept natural-language strategy requests and run backtests on a selected dataset"
- "Submit a strategy request in natural language (e.g., 'buy favorite 90 days out, hold to expiry')"
- "Verify the agent produces a confirmation summary of parsed rules"
- "Run the backtest and verify results (equity curve + metrics + trades)"
- "Modify a parameter (e.g., horizon to 60 days) and verify the agent re-runs correctly"
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from polymkt.backtest.engine import BacktestEngine
from polymkt.models.schemas import (
    BacktestCreateRequest,
    BacktestMetrics,
    BacktestSchema,
    BacktestTradeRecord,
    StrategyConfig,
)
from polymkt.signals.favorites import FavoriteSignalStore
from polymkt.storage.backtests import BacktestStore
from polymkt.storage.datasets import DatasetStore
from polymkt.storage.duckdb_layer import DuckDBLayer
from polymkt.storage.election_groups import ElectionGroupStore

logger = structlog.get_logger()


@dataclass
class ParsedStrategy:
    """Parsed strategy configuration from natural language."""

    name: str
    entry_days_to_exp: float
    exit_rule: str
    favorite_rule: str
    fee_rate: float
    slippage_rate: float
    position_size: float
    extra_params: dict[str, Any] | None = None

    def to_strategy_config(self) -> StrategyConfig:
        """Convert to StrategyConfig for backtest creation."""
        return StrategyConfig(
            name=self.name,
            entry_days_to_exp=self.entry_days_to_exp,
            exit_rule=self.exit_rule,
            favorite_rule=self.favorite_rule,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            position_size=self.position_size,
            extra_params=self.extra_params,
        )


@dataclass
class StrategyConfirmation:
    """Confirmation summary of parsed strategy before execution."""

    session_id: str
    dataset_id: str
    dataset_name: str
    market_count: int
    parsed_strategy: ParsedStrategy
    summary: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class BacktestAgentRequest:
    """Request for the backtesting agent."""

    dataset_id: str
    natural_language_strategy: str
    fee_rate: float | None = None
    slippage_rate: float | None = None
    position_size: float | None = None


@dataclass
class BacktestAgentResult:
    """Result of executing a backtest via the agent."""

    backtest_id: str
    status: str
    metrics: BacktestMetrics | None
    trades: list[BacktestTradeRecord]
    equity_curve: list[dict[str, Any]]
    error_message: str | None = None


class StrategyParser:
    """Parse natural language strategy descriptions into structured configurations.

    This rule-based parser extracts:
    - Entry timing (e.g., "90 days out", "3 months before expiry")
    - Favorite rule (e.g., "highest YES price", "buy favorite")
    - Exit rule (e.g., "hold to expiry", "take profit at 90%")
    - Fee/slippage parameters if mentioned
    """

    # Patterns for entry timing (days to expiry)
    ENTRY_TIMING_PATTERNS = [
        (r"(\d+)\s*days?\s*(out|before|to|from)\s*(expiry|expiration|close)", "days"),
        (r"(\d+)\s*days?\s*to\s*exp", "days"),
        (r"(\d+)\s*d(?:ays?)?\s*dte", "days"),  # DTE notation
        (r"(\d+)\s*weeks?\s*(out|before|to|from)", "weeks"),
        (r"(\d+)\s*months?\s*(out|before|to|from)", "months"),
        (r"at\s*(\d+)\s*days?", "days"),
        (r"(\d+)\s*days\s*horizon", "days"),
        (r"horizon\s*(?:of\s*)?(\d+)\s*days?", "days"),
        (r"(\d+)\s*days?\s*out\b", "days"),  # Simple "X days out" pattern
    ]

    # Patterns for favorite rule
    FAVORITE_PATTERNS = [
        (r"buy\s*(the\s*)?favorite", "max_yes_price"),
        (r"highest\s+yes\s*price", "max_yes_price"),
        (r"max\s+yes\s*price", "max_yes_price"),
        (r"front\s*-?\s*runner", "max_yes_price"),
        (r"leading\s+market", "max_yes_price"),
        (r"best\s+odds", "max_yes_price"),
        (r"top\s+candidate", "max_yes_price"),
        (r"buy\s+leader", "max_yes_price"),
        (r"buy\s+underdog", "min_yes_price"),
        (r"lowest\s+yes\s*price", "min_yes_price"),
        (r"min\s+yes\s*price", "min_yes_price"),
    ]

    # Patterns for exit rule
    EXIT_PATTERNS = [
        (r"hold\s*(to|until)\s*(expiry|expiration|close|end)", "expiry"),
        (r"hold\s*to\s*exp", "expiry"),
        (r"expiry\s*exit", "expiry"),
        (r"until\s*(market\s*)?(closes?|ends?)", "expiry"),
        (r"take\s*profit\s*(?:at\s*)?(\d+)%?", "take_profit"),
        (r"stop\s*loss\s*(?:at\s*)?(\d+)%?", "stop_loss"),
        (r"exit\s*at\s*(\d+)%", "target"),
    ]

    # Patterns for fees
    FEE_PATTERNS = [
        (r"(\d+(?:\.\d+)?)\s*%?\s*fee", "fee"),
        (r"fee\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%?", "fee"),
        (r"(\d+(?:\.\d+)?)\s*%?\s*slippage", "slippage"),
        (r"slippage\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%?", "slippage"),
    ]

    def __init__(self) -> None:
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._entry_patterns = [
            (re.compile(pattern, re.IGNORECASE), unit)
            for pattern, unit in self.ENTRY_TIMING_PATTERNS
        ]
        self._favorite_patterns = [
            (re.compile(pattern, re.IGNORECASE), rule)
            for pattern, rule in self.FAVORITE_PATTERNS
        ]
        self._exit_patterns = [
            (re.compile(pattern, re.IGNORECASE), rule)
            for pattern, rule in self.EXIT_PATTERNS
        ]
        self._fee_patterns = [
            (re.compile(pattern, re.IGNORECASE), fee_type)
            for pattern, fee_type in self.FEE_PATTERNS
        ]

    def parse(
        self,
        strategy_description: str,
        default_entry_days: float = 90.0,
        default_exit_rule: str = "expiry",
        default_favorite_rule: str = "max_yes_price",
    ) -> ParsedStrategy:
        """
        Parse a natural language strategy description.

        Args:
            strategy_description: Natural language strategy description
            default_entry_days: Default entry days to expiry if not detected
            default_exit_rule: Default exit rule if not detected
            default_favorite_rule: Default favorite rule if not detected

        Returns:
            ParsedStrategy with extracted configuration
        """
        # Extract entry timing
        entry_days = default_entry_days
        for pattern, unit in self._entry_patterns:
            match = pattern.search(strategy_description)
            if match:
                value = float(match.group(1))
                if unit == "weeks":
                    entry_days = value * 7
                elif unit == "months":
                    entry_days = value * 30
                else:
                    entry_days = value
                break

        # Extract favorite rule
        favorite_rule = default_favorite_rule
        for pattern, rule in self._favorite_patterns:
            if pattern.search(strategy_description):
                favorite_rule = rule
                break

        # Extract exit rule
        exit_rule = default_exit_rule
        extra_params: dict[str, Any] = {}
        for pattern, rule in self._exit_patterns:
            match = pattern.search(strategy_description)
            if match:
                exit_rule = rule
                # Extract target percentage if available
                if rule in ("take_profit", "stop_loss", "target") and match.lastindex:
                    try:
                        target_pct = float(match.group(match.lastindex))
                        extra_params[f"{rule}_pct"] = target_pct
                    except (ValueError, IndexError):
                        pass
                break

        # Extract fees
        fee_rate = 0.0
        slippage_rate = 0.0
        for pattern, fee_type in self._fee_patterns:
            match = pattern.search(strategy_description)
            if match:
                try:
                    value = float(match.group(1))
                    # Convert percentage to decimal (values are always expressed as percentages)
                    # e.g., "1% fee" means 1%, not 100%, so we always divide by 100
                    value = value / 100.0
                    if fee_type == "fee":
                        fee_rate = value
                    elif fee_type == "slippage":
                        slippage_rate = value
                except (ValueError, IndexError):
                    pass

        # Generate strategy name
        name = self._generate_strategy_name(entry_days, favorite_rule, exit_rule)

        return ParsedStrategy(
            name=name,
            entry_days_to_exp=entry_days,
            exit_rule=exit_rule,
            favorite_rule=favorite_rule,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size=1.0,  # Default to full position
            extra_params=extra_params if extra_params else None,
        )

    def _generate_strategy_name(
        self, entry_days: float, favorite_rule: str, exit_rule: str
    ) -> str:
        """Generate a descriptive strategy name."""
        rule_display = "favorite" if favorite_rule == "max_yes_price" else "underdog"
        exit_display = exit_rule.replace("_", " ")
        return f"buy_{rule_display}_{int(entry_days)}d_{exit_display}"


class BacktestingAgent:
    """Agent for executing backtests from natural language strategy requests.

    The BacktestingAgent processes natural language strategy descriptions,
    presents a confirmation summary, and executes backtests using the
    existing backtest engine.

    Example usage:
        agent = BacktestingAgent(...)

        # Process a strategy request
        confirmation = agent.prepare_backtest(
            BacktestAgentRequest(
                dataset_id="dataset-123",
                natural_language_strategy="buy favorite 90 days out, hold to expiry"
            )
        )

        # Review the confirmation summary
        print(confirmation.summary)

        # Execute the backtest
        result = agent.execute_backtest(confirmation.session_id)
    """

    def __init__(
        self,
        duckdb_layer: DuckDBLayer,
        dataset_store: DatasetStore,
        backtest_store: BacktestStore,
        election_group_store: ElectionGroupStore,
        favorite_signal_store: FavoriteSignalStore,
    ) -> None:
        """
        Initialize the BacktestingAgent.

        Args:
            duckdb_layer: DuckDB layer for querying trades/markets
            dataset_store: Store for dataset persistence
            backtest_store: Store for backtest persistence
            election_group_store: Store for election groups
            favorite_signal_store: Store for favorite signals
        """
        self.duckdb = duckdb_layer
        self.datasets = dataset_store
        self.backtests = backtest_store
        self.election_groups = election_group_store
        self.favorite_signals = favorite_signal_store

        self._parser = StrategyParser()
        self._sessions: dict[str, StrategyConfirmation] = {}

    def prepare_backtest(
        self, request: BacktestAgentRequest
    ) -> StrategyConfirmation:
        """
        Prepare a backtest from a natural language strategy request.

        This method parses the strategy, validates prerequisites, and returns
        a confirmation summary for user review before execution.

        Args:
            request: BacktestAgentRequest with dataset_id and strategy description

        Returns:
            StrategyConfirmation with parsed rules and summary

        Raises:
            ValueError: If dataset not found or has no markets
        """
        # Get the dataset
        dataset = self.datasets.get_dataset(request.dataset_id)
        market_count = len(dataset.market_ids) - len(dataset.excluded_market_ids)

        if market_count == 0:
            raise ValueError(
                f"Dataset {request.dataset_id} has no markets after exclusions"
            )

        # Parse the strategy
        parsed = self._parser.parse(request.natural_language_strategy)

        # Apply overrides from request
        if request.fee_rate is not None:
            parsed.fee_rate = request.fee_rate
        if request.slippage_rate is not None:
            parsed.slippage_rate = request.slippage_rate
        if request.position_size is not None:
            parsed.position_size = request.position_size

        # Check for warnings
        warnings: list[str] = []

        # Check if favorite signals exist at the entry days
        signals = self.favorite_signals.get_signals_for_snapshot(
            snapshot_days_to_exp=parsed.entry_days_to_exp,
            tolerance=0.5,
        )
        if not signals:
            warnings.append(
                f"No favorite signals found at {parsed.entry_days_to_exp} days to expiry. "
                "You may need to run /api/favorite-signals/compute first."
            )

        # Check election groups exist
        try:
            _, total_count = self.election_groups.list_groups(limit=1)
            if total_count == 0:
                warnings.append(
                    "No election groups defined. "
                    "Import election groups before running the backtest."
                )
        except Exception:
            warnings.append("Could not verify election groups.")

        # Generate summary
        summary = self._generate_summary(dataset.name, market_count, parsed)

        session_id = str(uuid.uuid4())
        confirmation = StrategyConfirmation(
            session_id=session_id,
            dataset_id=request.dataset_id,
            dataset_name=dataset.name,
            market_count=market_count,
            parsed_strategy=parsed,
            summary=summary,
            warnings=warnings,
        )

        # Store session for later execution
        self._sessions[session_id] = confirmation

        logger.info(
            "backtest_agent_prepared",
            session_id=session_id,
            dataset_id=request.dataset_id,
            entry_days=parsed.entry_days_to_exp,
            favorite_rule=parsed.favorite_rule,
            exit_rule=parsed.exit_rule,
            warnings_count=len(warnings),
        )

        return confirmation

    def execute_backtest(self, session_id: str) -> BacktestAgentResult:
        """
        Execute a prepared backtest.

        Args:
            session_id: Session ID from prepare_backtest

        Returns:
            BacktestAgentResult with execution results

        Raises:
            ValueError: If session not found
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        confirmation = self._sessions[session_id]
        parsed = confirmation.parsed_strategy

        # Create the backtest record
        backtest_request = BacktestCreateRequest(
            dataset_id=confirmation.dataset_id,
            strategy_config=parsed.to_strategy_config(),
        )
        backtest = self.backtests.create_backtest(backtest_request)

        # Execute using the backtest engine
        engine = BacktestEngine(
            duckdb_layer=self.duckdb,
            dataset_store=self.datasets,
            backtest_store=self.backtests,
            election_group_store=self.election_groups,
            favorite_signal_store=self.favorite_signals,
        )

        result = engine.execute(backtest.id)

        # Clean up session
        del self._sessions[session_id]

        logger.info(
            "backtest_agent_executed",
            session_id=session_id,
            backtest_id=backtest.id,
            status=result.status,
            trade_count=len(result.trades),
        )

        return BacktestAgentResult(
            backtest_id=backtest.id,
            status=result.status,
            metrics=result.metrics,
            trades=result.trades,
            equity_curve=result.equity_curve,
            error_message=result.error_message,
        )

    def modify_strategy(
        self,
        session_id: str,
        entry_days_to_exp: float | None = None,
        exit_rule: str | None = None,
        favorite_rule: str | None = None,
        fee_rate: float | None = None,
        slippage_rate: float | None = None,
        position_size: float | None = None,
    ) -> StrategyConfirmation:
        """
        Modify a prepared strategy before execution.

        Args:
            session_id: Session ID from prepare_backtest
            entry_days_to_exp: New entry days to expiry
            exit_rule: New exit rule
            favorite_rule: New favorite rule
            fee_rate: New fee rate
            slippage_rate: New slippage rate
            position_size: New position size

        Returns:
            Updated StrategyConfirmation

        Raises:
            ValueError: If session not found
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        confirmation = self._sessions[session_id]
        parsed = confirmation.parsed_strategy

        # Apply modifications
        if entry_days_to_exp is not None:
            parsed.entry_days_to_exp = entry_days_to_exp
        if exit_rule is not None:
            parsed.exit_rule = exit_rule
        if favorite_rule is not None:
            parsed.favorite_rule = favorite_rule
        if fee_rate is not None:
            parsed.fee_rate = fee_rate
        if slippage_rate is not None:
            parsed.slippage_rate = slippage_rate
        if position_size is not None:
            parsed.position_size = position_size

        # Regenerate name
        parsed.name = self._parser._generate_strategy_name(
            parsed.entry_days_to_exp,
            parsed.favorite_rule,
            parsed.exit_rule,
        )

        # Regenerate summary
        confirmation.summary = self._generate_summary(
            confirmation.dataset_name,
            confirmation.market_count,
            parsed,
        )

        # Re-check warnings for new entry days
        confirmation.warnings = []
        signals = self.favorite_signals.get_signals_for_snapshot(
            snapshot_days_to_exp=parsed.entry_days_to_exp,
            tolerance=0.5,
        )
        if not signals:
            confirmation.warnings.append(
                f"No favorite signals found at {parsed.entry_days_to_exp} days to expiry. "
                "You may need to run /api/favorite-signals/compute first."
            )

        logger.info(
            "backtest_agent_modified",
            session_id=session_id,
            entry_days=parsed.entry_days_to_exp,
            exit_rule=parsed.exit_rule,
        )

        return confirmation

    def get_session(self, session_id: str) -> StrategyConfirmation | None:
        """
        Get an existing session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            StrategyConfirmation or None if not found
        """
        return self._sessions.get(session_id)

    def _generate_summary(
        self,
        dataset_name: str,
        market_count: int,
        strategy: ParsedStrategy,
    ) -> str:
        """Generate a human-readable summary of the strategy configuration."""
        lines = [
            f"Strategy: {strategy.name}",
            f"Dataset: {dataset_name} ({market_count} markets)",
            "",
            "Rules:",
            f"  - Entry: {int(strategy.entry_days_to_exp)} days before expiry",
            f"  - Favorite: {strategy.favorite_rule.replace('_', ' ')}",
            f"  - Exit: {strategy.exit_rule.replace('_', ' ')}",
            "",
            "Costs:",
            f"  - Fee rate: {strategy.fee_rate * 100:.2f}%",
            f"  - Slippage rate: {strategy.slippage_rate * 100:.2f}%",
            "",
            f"Position sizing: {strategy.position_size * 100:.0f}% per trade",
        ]
        return "\n".join(lines)


def create_backtesting_agent(
    duckdb_layer: DuckDBLayer,
    db_path: Path,
) -> BacktestingAgent:
    """
    Factory function to create a BacktestingAgent.

    Args:
        duckdb_layer: DuckDB layer for querying
        db_path: Path to SQLite database for persistence

    Returns:
        Initialized BacktestingAgent
    """
    dataset_store = DatasetStore(db_path)
    backtest_store = BacktestStore(db_path)
    election_group_store = ElectionGroupStore(db_path)
    favorite_signal_store = FavoriteSignalStore(db_path)

    return BacktestingAgent(
        duckdb_layer=duckdb_layer,
        dataset_store=dataset_store,
        backtest_store=backtest_store,
        election_group_store=election_group_store,
        favorite_signal_store=favorite_signal_store,
    )
