"""Election group storage for grouping related markets into election groups.

Election groups allow grouping related prediction markets (e.g., all candidates
in an election) so that strategies like "buy the favorite" can compare markets
within the same group.
"""

import csv
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class ElectionGroupNotFoundError(Exception):
    """Raised when an election group is not found."""

    pass


class ElectionGroupStore:
    """SQLite-backed storage for election groups and their market mappings.

    An election group is a collection of related markets (e.g., candidates in
    an election). This store allows:
    - Creating/updating/deleting election groups
    - Mapping markets to groups
    - Querying groups and their markets
    - Importing mappings from CSV/JSON files
    - Detecting unmapped markets from a dataset
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            # Election groups table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS election_groups (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            # Market-to-group mapping table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS election_group_markets (
                    election_group_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (election_group_id, market_id),
                    FOREIGN KEY (election_group_id) REFERENCES election_groups(id)
                        ON DELETE CASCADE
                )
            """)
            # Index for efficient market lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_egm_market_id
                ON election_group_markets(market_id)
            """)
            # Index for listing groups by update time
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eg_updated_at
                ON election_groups(updated_at DESC)
            """)
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Election Group CRUD
    # =========================================================================

    def create_group(
        self,
        name: str,
        description: str | None = None,
        market_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new election group.

        Args:
            name: Name for the group (e.g., "2024 Presidential Election")
            description: Optional description of the group
            market_ids: Optional list of market IDs to include in the group

        Returns:
            The created election group as a dict
        """
        now = datetime.now(timezone.utc)
        group_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO election_groups (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (group_id, name, description, now.isoformat(), now.isoformat()),
            )

            # Add market mappings if provided
            if market_ids:
                for market_id in market_ids:
                    conn.execute(
                        """
                        INSERT INTO election_group_markets (election_group_id, market_id, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (group_id, market_id, now.isoformat()),
                    )

            conn.commit()
            logger.info("election_group_created", group_id=group_id, name=name, market_count=len(market_ids or []))
        finally:
            conn.close()

        return {
            "id": group_id,
            "name": name,
            "description": description,
            "market_ids": market_ids or [],
            "created_at": now,
            "updated_at": now,
        }

    def get_group(self, group_id: str) -> dict[str, Any]:
        """Get an election group by ID.

        Args:
            group_id: The election group UUID

        Returns:
            The election group with all its market IDs

        Raises:
            ElectionGroupNotFoundError: If group not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM election_groups WHERE id = ?",
                (group_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ElectionGroupNotFoundError(f"Election group not found: {group_id}")

            # Get associated market IDs
            cursor = conn.execute(
                "SELECT market_id FROM election_group_markets WHERE election_group_id = ?",
                (group_id,),
            )
            market_ids = [r[0] for r in cursor.fetchall()]

            return {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "market_ids": market_ids,
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
            }
        finally:
            conn.close()

    def update_group(
        self,
        group_id: str,
        name: str | None = None,
        description: str | None = None,
        market_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an existing election group.

        Args:
            group_id: The election group UUID
            name: New name (if provided)
            description: New description (if provided)
            market_ids: New list of market IDs (if provided, replaces existing)

        Returns:
            The updated election group

        Raises:
            ElectionGroupNotFoundError: If group not found
        """
        # First check group exists
        existing = self.get_group(group_id)
        now = datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path)
        try:
            # Update group fields
            updates = []
            params: list[Any] = []
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            updates.append("updated_at = ?")
            params.append(now.isoformat())
            params.append(group_id)

            if updates:
                conn.execute(
                    f"UPDATE election_groups SET {', '.join(updates)} WHERE id = ?",
                    params,
                )

            # Replace market mappings if provided
            if market_ids is not None:
                conn.execute(
                    "DELETE FROM election_group_markets WHERE election_group_id = ?",
                    (group_id,),
                )
                for market_id in market_ids:
                    conn.execute(
                        """
                        INSERT INTO election_group_markets (election_group_id, market_id, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (group_id, market_id, now.isoformat()),
                    )

            conn.commit()
            logger.info("election_group_updated", group_id=group_id)
        finally:
            conn.close()

        return self.get_group(group_id)

    def delete_group(self, group_id: str) -> None:
        """Delete an election group.

        Args:
            group_id: The election group UUID

        Raises:
            ElectionGroupNotFoundError: If group not found
        """
        # First check group exists
        self.get_group(group_id)

        conn = sqlite3.connect(self.db_path)
        try:
            # Delete mappings first (due to foreign key)
            conn.execute(
                "DELETE FROM election_group_markets WHERE election_group_id = ?",
                (group_id,),
            )
            conn.execute("DELETE FROM election_groups WHERE id = ?", (group_id,))
            conn.commit()
            logger.info("election_group_deleted", group_id=group_id)
        finally:
            conn.close()

    def list_groups(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List election groups with pagination.

        Args:
            limit: Maximum number of groups to return
            offset: Number of groups to skip

        Returns:
            Tuple of (groups list, total_count)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Get total count
            cursor = conn.execute("SELECT COUNT(*) FROM election_groups")
            total_count = cursor.fetchone()[0]

            # Get paginated results
            cursor = conn.execute(
                """
                SELECT eg.id, eg.name, eg.description, eg.created_at, eg.updated_at,
                       COUNT(egm.market_id) as market_count
                FROM election_groups eg
                LEFT JOIN election_group_markets egm ON eg.id = egm.election_group_id
                GROUP BY eg.id
                ORDER BY eg.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

            groups = []
            for row in cursor.fetchall():
                groups.append({
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "market_count": row["market_count"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                    "updated_at": datetime.fromisoformat(row["updated_at"]),
                })

            return groups, total_count
        finally:
            conn.close()

    # =========================================================================
    # Market-to-Group Mapping Operations
    # =========================================================================

    def add_markets_to_group(self, group_id: str, market_ids: list[str]) -> int:
        """Add markets to an election group.

        Args:
            group_id: The election group UUID
            market_ids: Market IDs to add

        Returns:
            Number of markets added (excludes duplicates)

        Raises:
            ElectionGroupNotFoundError: If group not found
        """
        # First check group exists
        self.get_group(group_id)
        now = datetime.now(timezone.utc)
        added = 0

        conn = sqlite3.connect(self.db_path)
        try:
            for market_id in market_ids:
                try:
                    conn.execute(
                        """
                        INSERT INTO election_group_markets (election_group_id, market_id, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (group_id, market_id, now.isoformat()),
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    # Market already in group, skip
                    pass

            conn.execute(
                "UPDATE election_groups SET updated_at = ? WHERE id = ?",
                (now.isoformat(), group_id),
            )
            conn.commit()
            logger.info("markets_added_to_group", group_id=group_id, added=added)
        finally:
            conn.close()

        return added

    def remove_markets_from_group(self, group_id: str, market_ids: list[str]) -> int:
        """Remove markets from an election group.

        Args:
            group_id: The election group UUID
            market_ids: Market IDs to remove

        Returns:
            Number of markets removed

        Raises:
            ElectionGroupNotFoundError: If group not found
        """
        # First check group exists
        self.get_group(group_id)
        now = datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ", ".join(["?" for _ in market_ids])
            cursor = conn.execute(
                f"""
                DELETE FROM election_group_markets
                WHERE election_group_id = ? AND market_id IN ({placeholders})
                """,
                [group_id] + market_ids,
            )
            removed = cursor.rowcount

            conn.execute(
                "UPDATE election_groups SET updated_at = ? WHERE id = ?",
                (now.isoformat(), group_id),
            )
            conn.commit()
            logger.info("markets_removed_from_group", group_id=group_id, removed=removed)
        finally:
            conn.close()

        return removed

    def get_group_for_market(self, market_id: str) -> dict[str, Any] | None:
        """Get the election group for a market (if any).

        Args:
            market_id: The market ID

        Returns:
            The election group or None if market is not in any group
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT eg.*
                FROM election_groups eg
                JOIN election_group_markets egm ON eg.id = egm.election_group_id
                WHERE egm.market_id = ?
                """,
                (market_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            # Get all market IDs for this group
            cursor = conn.execute(
                "SELECT market_id FROM election_group_markets WHERE election_group_id = ?",
                (row["id"],),
            )
            market_ids = [r[0] for r in cursor.fetchall()]

            return {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "market_ids": market_ids,
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
            }
        finally:
            conn.close()

    def get_markets_by_group(self) -> dict[str, list[str]]:
        """Get all market-to-group mappings.

        Returns:
            Dict mapping election_group_id -> list of market_ids
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT election_group_id, market_id FROM election_group_markets ORDER BY election_group_id"
            )
            groups: dict[str, list[str]] = {}
            for row in cursor.fetchall():
                group_id = row[0]
                market_id = row[1]
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(market_id)
            return groups
        finally:
            conn.close()

    # =========================================================================
    # Import/Export and Validation
    # =========================================================================

    def import_from_csv(self, csv_path: Path) -> dict[str, Any]:
        """Import election group mappings from a CSV file.

        The CSV should have columns: election_group_id, market_id
        Optionally: election_group_name, election_group_description

        Args:
            csv_path: Path to the CSV file

        Returns:
            Import summary with counts
        """
        groups_created = 0
        markets_mapped = 0
        errors: list[str] = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            # Track groups we've created in this import
            group_cache: dict[str, str] = {}  # group_id -> name

            for row_num, row in enumerate(reader, start=2):
                try:
                    group_id = row.get("election_group_id", "").strip()
                    market_id = row.get("market_id", "").strip()

                    if not group_id or not market_id:
                        errors.append(f"Row {row_num}: Missing election_group_id or market_id")
                        continue

                    # Create group if it doesn't exist
                    if group_id not in group_cache:
                        try:
                            self.get_group(group_id)
                            group_cache[group_id] = "existing"
                        except ElectionGroupNotFoundError:
                            # Create new group
                            name = row.get("election_group_name", f"Group {group_id}").strip()
                            desc = row.get("election_group_description", "").strip() or None
                            self._create_group_with_id(group_id, name, desc)
                            group_cache[group_id] = name
                            groups_created += 1

                    # Add market to group
                    added = self.add_markets_to_group(group_id, [market_id])
                    markets_mapped += added

                except Exception as e:
                    errors.append(f"Row {row_num}: {e}")

        logger.info(
            "csv_import_complete",
            groups_created=groups_created,
            markets_mapped=markets_mapped,
            errors=len(errors),
        )

        return {
            "groups_created": groups_created,
            "markets_mapped": markets_mapped,
            "errors": errors,
        }

    def import_from_json(self, json_path: Path) -> dict[str, Any]:
        """Import election group mappings from a JSON file.

        The JSON should be a list of objects with structure:
        [
            {
                "id": "group-uuid" (optional, will be generated if missing),
                "name": "Group Name",
                "description": "Optional description",
                "market_ids": ["market1", "market2", ...]
            }
        ]

        Args:
            json_path: Path to the JSON file

        Returns:
            Import summary with counts
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        groups_created = 0
        markets_mapped = 0
        errors: list[str] = []

        for idx, group_data in enumerate(data):
            try:
                group_id = group_data.get("id")
                name = group_data.get("name", f"Group {idx + 1}")
                description = group_data.get("description")
                market_ids = group_data.get("market_ids", [])

                if not market_ids:
                    errors.append(f"Group {idx}: No market_ids provided")
                    continue

                if group_id:
                    # Check if group exists
                    try:
                        self.get_group(group_id)
                        # Update existing group
                        self.update_group(group_id, name=name, description=description, market_ids=market_ids)
                    except ElectionGroupNotFoundError:
                        # Create with specific ID
                        self._create_group_with_id(group_id, name, description, market_ids)
                        groups_created += 1
                else:
                    # Create new group with generated ID
                    self.create_group(name=name, description=description, market_ids=market_ids)
                    groups_created += 1

                markets_mapped += len(market_ids)

            except Exception as e:
                errors.append(f"Group {idx}: {e}")

        logger.info(
            "json_import_complete",
            groups_created=groups_created,
            markets_mapped=markets_mapped,
            errors=len(errors),
        )

        return {
            "groups_created": groups_created,
            "markets_mapped": markets_mapped,
            "errors": errors,
        }

    def _create_group_with_id(
        self,
        group_id: str,
        name: str,
        description: str | None = None,
        market_ids: list[str] | None = None,
    ) -> None:
        """Create an election group with a specific ID (used for imports)."""
        now = datetime.now(timezone.utc)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO election_groups (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (group_id, name, description, now.isoformat(), now.isoformat()),
            )

            if market_ids:
                for market_id in market_ids:
                    conn.execute(
                        """
                        INSERT INTO election_group_markets (election_group_id, market_id, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (group_id, market_id, now.isoformat()),
                    )

            conn.commit()
        finally:
            conn.close()

    def find_unmapped_markets(self, market_ids: list[str]) -> list[str]:
        """Find markets from a list that are not in any election group.

        Args:
            market_ids: List of market IDs to check

        Returns:
            List of market IDs that are not in any election group
        """
        if not market_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ", ".join(["?" for _ in market_ids])
            cursor = conn.execute(
                f"""
                SELECT DISTINCT market_id FROM election_group_markets
                WHERE market_id IN ({placeholders})
                """,
                market_ids,
            )
            mapped_ids = {row[0] for row in cursor.fetchall()}
            return [mid for mid in market_ids if mid not in mapped_ids]
        finally:
            conn.close()

    def validate_groups(self, min_markets: int = 2) -> dict[str, Any]:
        """Validate election groups and report issues.

        Checks:
        - Groups with fewer than min_markets markets
        - Groups with only one market (can't determine favorite)

        Args:
            min_markets: Minimum markets required per group (default 2)

        Returns:
            Validation report with issues
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT eg.id, eg.name, COUNT(egm.market_id) as market_count
                FROM election_groups eg
                LEFT JOIN election_group_markets egm ON eg.id = egm.election_group_id
                GROUP BY eg.id
            """)

            issues: list[dict[str, Any]] = []
            valid_groups = 0
            total_groups = 0

            for row in cursor.fetchall():
                total_groups += 1
                if row["market_count"] < min_markets:
                    issues.append({
                        "group_id": row["id"],
                        "group_name": row["name"],
                        "market_count": row["market_count"],
                        "issue": f"Group has {row['market_count']} markets, minimum is {min_markets}",
                    })
                else:
                    valid_groups += 1

            return {
                "total_groups": total_groups,
                "valid_groups": valid_groups,
                "invalid_groups": total_groups - valid_groups,
                "issues": issues,
                "min_markets_required": min_markets,
            }
        finally:
            conn.close()

    def export_to_json(self) -> list[dict[str, Any]]:
        """Export all election groups to a JSON-serializable format.

        Returns:
            List of election group dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT * FROM election_groups ORDER BY name")
            groups = []

            for row in cursor.fetchall():
                # Get market IDs for this group
                market_cursor = conn.execute(
                    "SELECT market_id FROM election_group_markets WHERE election_group_id = ?",
                    (row["id"],),
                )
                market_ids = [r[0] for r in market_cursor.fetchall()]

                groups.append({
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "market_ids": market_ids,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                })

            return groups
        finally:
            conn.close()
