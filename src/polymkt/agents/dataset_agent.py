"""Dataset Agent for natural language dataset creation.

This module provides the DatasetAgent class that accepts natural language
filters (e.g., "find election markets about senate control") and produces
a dataset with an editable market list.

The agent uses the existing hybrid search infrastructure to find relevant
markets and allows users to modify the selection before saving.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import structlog

from polymkt.models.schemas import (
    DatasetCreateRequest,
    DatasetFilters,
    DatasetSchema,
    UnifiedMarketSearchResult,
)
from polymkt.storage.datasets import DatasetStore
from polymkt.storage.hybrid_search import HybridSearchIndex

logger = structlog.get_logger()


@dataclass
class MarketListItem:
    """A market in the agent's result with inclusion flag."""

    market_id: str
    question: str
    category: str | None
    tags: list[str] | None
    closed_time: datetime | None
    relevance_score: float
    included: bool = True


@dataclass
class DatasetAgentRequest:
    """Request for the dataset agent to process."""

    natural_language_query: str
    max_results: int = 100
    category_filter: str | None = None
    closed_time_min: str | None = None
    closed_time_max: str | None = None


@dataclass
class DatasetAgentResponse:
    """Response from the dataset agent."""

    query: str
    parsed_query: str
    category_filter: str | None
    closed_time_filter: dict[str, str | None] | None
    market_count: int
    markets: list[MarketListItem]
    summary: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DatasetAgentSaveResult:
    """Result of saving a dataset from agent response."""

    dataset_id: str
    dataset_name: str
    market_count: int
    excluded_count: int


class NaturalLanguageParser:
    """Parse natural language queries to extract search intent and filters.

    This is a rule-based parser that extracts:
    - Search keywords
    - Category filters (e.g., "politics", "sports")
    - Time filters (e.g., "closing soon", "closing in 2024")
    """

    # Known categories to recognize in queries
    KNOWN_CATEGORIES = [
        "politics",
        "elections",
        "sports",
        "crypto",
        "finance",
        "entertainment",
        "science",
        "technology",
        "weather",
    ]

    # Patterns for time expressions
    TIME_PATTERNS = [
        (r"closing in (\d{4})", "year"),
        (r"expiring in (\d{4})", "year"),
        (r"closing before (.+)", "before"),
        (r"closing after (.+)", "after"),
        (r"closing soon", "soon"),
        (r"expires? soon", "soon"),
    ]

    def __init__(self) -> None:
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._time_patterns = [
            (re.compile(pattern, re.IGNORECASE), time_type)
            for pattern, time_type in self.TIME_PATTERNS
        ]

    def parse(self, query: str) -> dict[str, Any]:
        """
        Parse a natural language query into structured filters.

        Args:
            query: Natural language query string

        Returns:
            Dictionary with:
            - search_query: Cleaned search keywords
            - category: Detected category filter (if any)
            - closed_time_min: Minimum closed time (if any)
            - closed_time_max: Maximum closed time (if any)
        """
        result: dict[str, Any] = {
            "search_query": query,
            "category": None,
            "closed_time_min": None,
            "closed_time_max": None,
        }

        query_lower = query.lower()
        cleaned_query = query

        # Detect category
        for category in self.KNOWN_CATEGORIES:
            if category in query_lower:
                result["category"] = category
                # Remove category from search query to avoid redundant matching
                pattern = re.compile(rf"\b{category}\b\s*(markets?)?", re.IGNORECASE)
                cleaned_query = pattern.sub("", cleaned_query).strip()
                break

        # Detect time expressions
        for pattern, time_type in self._time_patterns:
            match = pattern.search(query)
            if match:
                if time_type == "year":
                    year = match.group(1)
                    result["closed_time_min"] = f"{year}-01-01T00:00:00Z"
                    result["closed_time_max"] = f"{year}-12-31T23:59:59Z"
                elif time_type == "soon":
                    # "Soon" means within the next 30 days
                    now = datetime.now(timezone.utc)
                    result["closed_time_max"] = now.replace(
                        day=min(now.day + 30, 28)
                    ).isoformat().replace("+00:00", "Z")
                elif time_type == "before":
                    result["closed_time_max"] = match.group(1)
                elif time_type == "after":
                    result["closed_time_min"] = match.group(1)

                # Remove time expression from search query
                cleaned_query = pattern.sub("", cleaned_query).strip()
                break

        # Handle common filter phrases
        filter_phrases = [
            r"find\s+",
            r"search for\s+",
            r"show me\s+",
            r"get\s+",
            r"list\s+",
            r"about\s+",
            r"related to\s+",
            r"involving\s+",
        ]
        for phrase in filter_phrases:
            cleaned_query = re.sub(phrase, "", cleaned_query, flags=re.IGNORECASE)

        # Clean up extra whitespace and punctuation
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()
        cleaned_query = re.sub(r"[?!.,;]+$", "", cleaned_query).strip()

        result["search_query"] = cleaned_query if cleaned_query else query

        return result


class DatasetAgent:
    """Agent for creating datasets from natural language queries.

    The DatasetAgent processes natural language requests to find markets,
    presents results with inclusion flags, and allows saving to datasets.

    Example usage:
        agent = DatasetAgent(conn, db_path)

        # Process a natural language request
        response = agent.process_query("find election markets about senate control")

        # Modify the market list
        response.markets[0].included = False  # Exclude first market

        # Save to dataset
        result = agent.save_dataset(
            response=response,
            name="Senate Markets",
            description="Markets about senate control",
        )
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        db_path: Path,
        openai_api_key: str | None = None,
    ) -> None:
        """
        Initialize the DatasetAgent.

        Args:
            conn: DuckDB connection with search indices available
            db_path: Path to SQLite database for dataset storage
            openai_api_key: Optional OpenAI API key for semantic search
        """
        self.conn = conn
        self.db_path = db_path
        self.openai_api_key = openai_api_key

        self._parser = NaturalLanguageParser()
        self._search_index = HybridSearchIndex(
            conn=conn,
            openai_api_key=openai_api_key,
        )
        self._dataset_store = DatasetStore(db_path)

        # Track active sessions for modification
        self._sessions: dict[str, DatasetAgentResponse] = {}

    def process_query(self, request: DatasetAgentRequest | str) -> DatasetAgentResponse:
        """
        Process a natural language query and return a market list.

        Args:
            request: Natural language query or DatasetAgentRequest

        Returns:
            DatasetAgentResponse with markets and inclusion flags

        Raises:
            RuntimeError: If search index is not available
        """
        if isinstance(request, str):
            request = DatasetAgentRequest(natural_language_query=request)

        # Parse the natural language query
        parsed = self._parser.parse(request.natural_language_query)
        search_query = parsed["search_query"]

        # Apply request-level filters (override parsed filters if provided)
        category = request.category_filter or parsed["category"]
        closed_time_min = request.closed_time_min or parsed["closed_time_min"]
        closed_time_max = request.closed_time_max or parsed["closed_time_max"]

        logger.info(
            "dataset_agent_processing",
            original_query=request.natural_language_query,
            parsed_query=search_query,
            category=category,
            closed_time_min=closed_time_min,
            closed_time_max=closed_time_max,
        )

        # Execute search
        try:
            results, total_count = self._search_index.search(
                query=search_query,
                limit=request.max_results,
                offset=0,
                category=category,
                closed_time_min=closed_time_min,
                closed_time_max=closed_time_max,
            )
        except RuntimeError as e:
            logger.error("dataset_agent_search_failed", error=str(e))
            raise

        # Convert results to MarketListItem
        markets = [
            MarketListItem(
                market_id=r["id"],
                question=r["question"],
                category=r.get("category"),
                tags=r.get("tags"),
                closed_time=r.get("closed_time"),
                relevance_score=r.get("score", 0.0),
                included=True,  # All included by default
            )
            for r in results
        ]

        # Generate summary
        summary = self._generate_summary(
            query=request.natural_language_query,
            search_query=search_query,
            category=category,
            market_count=len(markets),
            total_count=total_count,
        )

        response = DatasetAgentResponse(
            query=request.natural_language_query,
            parsed_query=search_query,
            category_filter=category,
            closed_time_filter={
                "min": closed_time_min,
                "max": closed_time_max,
            }
            if closed_time_min or closed_time_max
            else None,
            market_count=len(markets),
            markets=markets,
            summary=summary,
        )

        # Store session for later modification
        self._sessions[response.session_id] = response

        logger.info(
            "dataset_agent_query_completed",
            session_id=response.session_id,
            market_count=len(markets),
        )

        return response

    def modify_market_list(
        self,
        session_id: str,
        market_id: str,
        included: bool,
    ) -> DatasetAgentResponse:
        """
        Modify the inclusion status of a market in a session.

        Args:
            session_id: Session ID from a previous process_query call
            market_id: Market ID to modify
            included: Whether to include the market

        Returns:
            Updated DatasetAgentResponse

        Raises:
            ValueError: If session_id is not found or market_id is not in the list
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        response = self._sessions[session_id]

        # Find and update the market
        found = False
        for market in response.markets:
            if market.market_id == market_id:
                market.included = included
                found = True
                break

        if not found:
            raise ValueError(f"Market not found in session: {market_id}")

        # Update market count
        response.market_count = sum(1 for m in response.markets if m.included)

        logger.info(
            "dataset_agent_market_modified",
            session_id=session_id,
            market_id=market_id,
            included=included,
        )

        return response

    def bulk_modify_markets(
        self,
        session_id: str,
        market_ids: list[str],
        included: bool,
    ) -> DatasetAgentResponse:
        """
        Bulk modify inclusion status for multiple markets.

        Args:
            session_id: Session ID from a previous process_query call
            market_ids: List of market IDs to modify
            included: Whether to include the markets

        Returns:
            Updated DatasetAgentResponse

        Raises:
            ValueError: If session_id is not found
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        response = self._sessions[session_id]
        market_id_set = set(market_ids)

        modified_count = 0
        for market in response.markets:
            if market.market_id in market_id_set:
                market.included = included
                modified_count += 1

        # Update market count
        response.market_count = sum(1 for m in response.markets if m.included)

        logger.info(
            "dataset_agent_bulk_modify",
            session_id=session_id,
            requested_count=len(market_ids),
            modified_count=modified_count,
            included=included,
        )

        return response

    def get_session(self, session_id: str) -> DatasetAgentResponse | None:
        """
        Get an existing session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            DatasetAgentResponse or None if not found
        """
        return self._sessions.get(session_id)

    def save_dataset(
        self,
        response: DatasetAgentResponse,
        name: str,
        description: str | None = None,
    ) -> DatasetAgentSaveResult:
        """
        Save the agent response as a dataset.

        Args:
            response: DatasetAgentResponse from process_query
            name: Name for the dataset
            description: Optional description

        Returns:
            DatasetAgentSaveResult with the created dataset info
        """
        # Separate included and excluded markets
        included_market_ids = [m.market_id for m in response.markets if m.included]
        excluded_market_ids = [m.market_id for m in response.markets if not m.included]

        # Build filters from parsed query
        closed_time_min_val = None
        closed_time_max_val = None
        if response.closed_time_filter:
            min_str = response.closed_time_filter.get("min")
            max_str = response.closed_time_filter.get("max")
            if min_str:
                closed_time_min_val = datetime.fromisoformat(min_str.replace("Z", "+00:00"))
            if max_str:
                closed_time_max_val = datetime.fromisoformat(max_str.replace("Z", "+00:00"))

        filters = DatasetFilters(
            query=response.parsed_query,
            category=response.category_filter,
            closed_time_min=closed_time_min_val,
            closed_time_max=closed_time_max_val,
        )

        # Create dataset
        create_request = DatasetCreateRequest(
            name=name,
            description=description,
            filters=filters,
            market_ids=included_market_ids,
            excluded_market_ids=excluded_market_ids,
        )

        dataset = self._dataset_store.create_dataset(create_request)

        # Clean up session
        if response.session_id in self._sessions:
            del self._sessions[response.session_id]

        logger.info(
            "dataset_agent_dataset_saved",
            dataset_id=dataset.id,
            name=name,
            included_count=len(included_market_ids),
            excluded_count=len(excluded_market_ids),
        )

        return DatasetAgentSaveResult(
            dataset_id=dataset.id,
            dataset_name=name,
            market_count=len(included_market_ids),
            excluded_count=len(excluded_market_ids),
        )

    def _generate_summary(
        self,
        query: str,
        search_query: str,
        category: str | None,
        market_count: int,
        total_count: int,
    ) -> str:
        """Generate a human-readable summary of the search results."""
        parts = []

        parts.append(f"Found {market_count} markets")

        if total_count > market_count:
            parts.append(f" (showing {market_count} of {total_count})")

        if category:
            parts.append(f" in category '{category}'")

        parts.append(f" matching '{search_query}'")

        return "".join(parts) + "."


def create_dataset_agent(
    conn: duckdb.DuckDBPyConnection,
    db_path: Path,
    openai_api_key: str | None = None,
) -> DatasetAgent:
    """
    Factory function to create a DatasetAgent.

    Args:
        conn: DuckDB connection with search indices available
        db_path: Path to SQLite database for dataset storage
        openai_api_key: Optional OpenAI API key for semantic search

    Returns:
        Initialized DatasetAgent
    """
    return DatasetAgent(
        conn=conn,
        db_path=db_path,
        openai_api_key=openai_api_key,
    )
