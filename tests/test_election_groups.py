"""Tests for the election groups feature."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app
from polymkt.storage.election_groups import (
    ElectionGroupNotFoundError,
    ElectionGroupStore,
)


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def group_store(temp_db: Path) -> ElectionGroupStore:
    """Create an election group store with a temp database."""
    return ElectionGroupStore(temp_db)


@pytest.fixture
def client(temp_db: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client with a temp database."""
    from polymkt import config

    monkeypatch.setattr(config.settings, "metadata_db_path", temp_db)
    return TestClient(app)


# =============================================================================
# ElectionGroupStore Unit Tests
# =============================================================================


class TestElectionGroupStoreCreation:
    """Test store initialization."""

    def test_init_creates_table(self, group_store: ElectionGroupStore) -> None:
        """Test that initialization creates the required tables."""
        import sqlite3

        conn = sqlite3.connect(group_store.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='election_groups'"
        )
        assert cursor.fetchone() is not None

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='election_group_markets'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that initialization creates parent directories."""
        nested_path = tmp_path / "deep" / "nested" / "path" / "test.db"
        store = ElectionGroupStore(nested_path)
        assert nested_path.parent.exists()


class TestElectionGroupCreate:
    """Test group creation."""

    def test_create_group_basic(self, group_store: ElectionGroupStore) -> None:
        """Test basic group creation."""
        result = group_store.create_group(name="2024 Presidential Election")

        assert "id" in result
        assert result["name"] == "2024 Presidential Election"
        assert result["description"] is None
        assert result["market_ids"] == []
        assert "created_at" in result
        assert "updated_at" in result

    def test_create_group_with_description(self, group_store: ElectionGroupStore) -> None:
        """Test group creation with description."""
        result = group_store.create_group(
            name="2024 Presidential Election",
            description="Markets for the 2024 US presidential election candidates",
        )

        assert result["description"] == "Markets for the 2024 US presidential election candidates"

    def test_create_group_with_markets(self, group_store: ElectionGroupStore) -> None:
        """Test group creation with initial market IDs."""
        market_ids = ["market-biden", "market-trump", "market-rfk"]
        result = group_store.create_group(
            name="2024 Presidential Election",
            market_ids=market_ids,
        )

        assert result["market_ids"] == market_ids
        assert len(result["market_ids"]) == 3

    def test_create_multiple_groups(self, group_store: ElectionGroupStore) -> None:
        """Test creating multiple groups."""
        group1 = group_store.create_group(name="Group 1")
        group2 = group_store.create_group(name="Group 2")

        assert group1["id"] != group2["id"]


class TestElectionGroupGet:
    """Test group retrieval."""

    def test_get_group_by_id(self, group_store: ElectionGroupStore) -> None:
        """Test retrieving a group by ID."""
        created = group_store.create_group(
            name="Test Group",
            description="Test description",
            market_ids=["m1", "m2"],
        )

        retrieved = group_store.get_group(created["id"])

        assert retrieved["id"] == created["id"]
        assert retrieved["name"] == "Test Group"
        assert retrieved["description"] == "Test description"
        assert retrieved["market_ids"] == ["m1", "m2"]

    def test_get_nonexistent_group_raises(self, group_store: ElectionGroupStore) -> None:
        """Test that getting a non-existent group raises error."""
        with pytest.raises(ElectionGroupNotFoundError):
            group_store.get_group("nonexistent-id")


class TestElectionGroupUpdate:
    """Test group updates."""

    def test_update_group_name(self, group_store: ElectionGroupStore) -> None:
        """Test updating group name only."""
        created = group_store.create_group(name="Original Name")
        updated = group_store.update_group(created["id"], name="New Name")

        assert updated["name"] == "New Name"

    def test_update_group_description(self, group_store: ElectionGroupStore) -> None:
        """Test updating group description."""
        created = group_store.create_group(name="Test", description="Old desc")
        updated = group_store.update_group(created["id"], description="New desc")

        assert updated["description"] == "New desc"

    def test_update_group_market_ids(self, group_store: ElectionGroupStore) -> None:
        """Test updating market IDs (replaces existing)."""
        created = group_store.create_group(name="Test", market_ids=["m1", "m2"])
        updated = group_store.update_group(created["id"], market_ids=["m3", "m4", "m5"])

        assert updated["market_ids"] == ["m3", "m4", "m5"]

    def test_update_nonexistent_group_raises(self, group_store: ElectionGroupStore) -> None:
        """Test that updating a non-existent group raises error."""
        with pytest.raises(ElectionGroupNotFoundError):
            group_store.update_group("nonexistent-id", name="New Name")

    def test_update_preserves_created_at(self, group_store: ElectionGroupStore) -> None:
        """Test that update preserves created_at timestamp."""
        created = group_store.create_group(name="Test")
        original_created_at = created["created_at"]

        updated = group_store.update_group(created["id"], name="Updated")

        assert updated["created_at"] == original_created_at
        assert updated["updated_at"] > original_created_at


class TestElectionGroupDelete:
    """Test group deletion."""

    def test_delete_group(self, group_store: ElectionGroupStore) -> None:
        """Test deleting a group."""
        created = group_store.create_group(name="To Delete")
        group_store.delete_group(created["id"])

        with pytest.raises(ElectionGroupNotFoundError):
            group_store.get_group(created["id"])

    def test_delete_nonexistent_group_raises(self, group_store: ElectionGroupStore) -> None:
        """Test that deleting a non-existent group raises error."""
        with pytest.raises(ElectionGroupNotFoundError):
            group_store.delete_group("nonexistent-id")

    def test_delete_removes_market_mappings(self, group_store: ElectionGroupStore) -> None:
        """Test that deleting a group removes its market mappings."""
        created = group_store.create_group(name="Test", market_ids=["m1", "m2"])
        group_store.delete_group(created["id"])

        # Verify mappings are gone
        import sqlite3

        conn = sqlite3.connect(group_store.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM election_group_markets WHERE election_group_id = ?",
            (created["id"],),
        )
        assert cursor.fetchone()[0] == 0
        conn.close()


class TestElectionGroupList:
    """Test group listing."""

    def test_list_empty(self, group_store: ElectionGroupStore) -> None:
        """Test listing when no groups exist."""
        groups, total_count = group_store.list_groups()

        assert groups == []
        assert total_count == 0

    def test_list_returns_summaries(self, group_store: ElectionGroupStore) -> None:
        """Test that list returns summaries with market_count."""
        group_store.create_group(name="Group 1", market_ids=["m1", "m2"])
        group_store.create_group(name="Group 2", market_ids=["m3"])

        groups, total_count = group_store.list_groups()

        assert total_count == 2
        assert len(groups) == 2
        # Each summary should have market_count
        for g in groups:
            assert "market_count" in g

    def test_list_pagination_limit(self, group_store: ElectionGroupStore) -> None:
        """Test pagination with limit."""
        for i in range(5):
            group_store.create_group(name=f"Group {i}")

        groups, total_count = group_store.list_groups(limit=2)

        assert len(groups) == 2
        assert total_count == 5

    def test_list_pagination_offset(self, group_store: ElectionGroupStore) -> None:
        """Test pagination with offset."""
        for i in range(5):
            group_store.create_group(name=f"Group {i}")

        groups, total_count = group_store.list_groups(limit=2, offset=2)

        assert len(groups) == 2
        assert total_count == 5

    def test_list_ordered_by_updated_at(self, group_store: ElectionGroupStore) -> None:
        """Test that list is ordered by updated_at descending."""
        g1 = group_store.create_group(name="First")
        g2 = group_store.create_group(name="Second")
        # Update g1 to make it most recent
        group_store.update_group(g1["id"], name="First Updated")

        groups, _ = group_store.list_groups()

        # g1 should be first (most recently updated)
        assert groups[0]["name"] == "First Updated"


class TestElectionGroupMarketOperations:
    """Test market-to-group operations."""

    def test_add_markets_to_group(self, group_store: ElectionGroupStore) -> None:
        """Test adding markets to an existing group."""
        created = group_store.create_group(name="Test")
        added = group_store.add_markets_to_group(created["id"], ["m1", "m2"])

        assert added == 2
        group = group_store.get_group(created["id"])
        assert "m1" in group["market_ids"]
        assert "m2" in group["market_ids"]

    def test_add_markets_skips_duplicates(self, group_store: ElectionGroupStore) -> None:
        """Test that adding duplicate markets is skipped."""
        created = group_store.create_group(name="Test", market_ids=["m1"])
        added = group_store.add_markets_to_group(created["id"], ["m1", "m2"])

        assert added == 1  # Only m2 was added
        group = group_store.get_group(created["id"])
        assert len(group["market_ids"]) == 2

    def test_remove_markets_from_group(self, group_store: ElectionGroupStore) -> None:
        """Test removing markets from a group."""
        created = group_store.create_group(name="Test", market_ids=["m1", "m2", "m3"])
        removed = group_store.remove_markets_from_group(created["id"], ["m1", "m3"])

        assert removed == 2
        group = group_store.get_group(created["id"])
        assert group["market_ids"] == ["m2"]

    def test_get_group_for_market(self, group_store: ElectionGroupStore) -> None:
        """Test finding the group for a market."""
        created = group_store.create_group(name="Test", market_ids=["m1", "m2"])
        result = group_store.get_group_for_market("m1")

        assert result is not None
        assert result["id"] == created["id"]

    def test_get_group_for_market_not_found(self, group_store: ElectionGroupStore) -> None:
        """Test finding group for unmapped market returns None."""
        result = group_store.get_group_for_market("unmapped-market")

        assert result is None

    def test_get_markets_by_group(self, group_store: ElectionGroupStore) -> None:
        """Test getting all markets grouped by group ID."""
        g1 = group_store.create_group(name="G1", market_ids=["m1", "m2"])
        g2 = group_store.create_group(name="G2", market_ids=["m3", "m4", "m5"])

        result = group_store.get_markets_by_group()

        assert g1["id"] in result
        assert g2["id"] in result
        assert set(result[g1["id"]]) == {"m1", "m2"}
        assert set(result[g2["id"]]) == {"m3", "m4", "m5"}


class TestElectionGroupValidation:
    """Test group validation."""

    def test_validate_groups_all_valid(self, group_store: ElectionGroupStore) -> None:
        """Test validation when all groups are valid."""
        group_store.create_group(name="G1", market_ids=["m1", "m2"])
        group_store.create_group(name="G2", market_ids=["m3", "m4", "m5"])

        result = group_store.validate_groups(min_markets=2)

        assert result["total_groups"] == 2
        assert result["valid_groups"] == 2
        assert result["invalid_groups"] == 0
        assert result["issues"] == []

    def test_validate_groups_with_invalid(self, group_store: ElectionGroupStore) -> None:
        """Test validation detects groups with too few markets."""
        group_store.create_group(name="Valid", market_ids=["m1", "m2"])
        group_store.create_group(name="Invalid", market_ids=["m3"])

        result = group_store.validate_groups(min_markets=2)

        assert result["total_groups"] == 2
        assert result["valid_groups"] == 1
        assert result["invalid_groups"] == 1
        assert len(result["issues"]) == 1
        assert result["issues"][0]["group_name"] == "Invalid"

    def test_find_unmapped_markets(self, group_store: ElectionGroupStore) -> None:
        """Test finding unmapped markets."""
        group_store.create_group(name="Test", market_ids=["m1", "m2"])

        unmapped = group_store.find_unmapped_markets(["m1", "m2", "m3", "m4"])

        assert set(unmapped) == {"m3", "m4"}

    def test_find_unmapped_markets_all_mapped(self, group_store: ElectionGroupStore) -> None:
        """Test finding unmapped when all are mapped."""
        group_store.create_group(name="Test", market_ids=["m1", "m2"])

        unmapped = group_store.find_unmapped_markets(["m1", "m2"])

        assert unmapped == []

    def test_find_unmapped_markets_empty_list(self, group_store: ElectionGroupStore) -> None:
        """Test finding unmapped with empty input."""
        unmapped = group_store.find_unmapped_markets([])

        assert unmapped == []


class TestElectionGroupImport:
    """Test import functionality."""

    def test_import_from_csv(self, group_store: ElectionGroupStore, tmp_path: Path) -> None:
        """Test importing from CSV file."""
        csv_content = """election_group_id,market_id,election_group_name,election_group_description
group-1,market-a,Presidential Race,2024 US Presidential candidates
group-1,market-b,Presidential Race,2024 US Presidential candidates
group-2,market-c,Senate Race,Senate race markets
"""
        csv_path = tmp_path / "groups.csv"
        csv_path.write_text(csv_content)

        result = group_store.import_from_csv(csv_path)

        assert result["groups_created"] == 2
        assert result["markets_mapped"] == 3
        assert result["errors"] == []

        # Verify groups were created
        group1 = group_store.get_group("group-1")
        assert group1["name"] == "Presidential Race"
        assert set(group1["market_ids"]) == {"market-a", "market-b"}

    def test_import_from_json(self, group_store: ElectionGroupStore, tmp_path: Path) -> None:
        """Test importing from JSON file."""
        json_data = [
            {
                "name": "Presidential Race",
                "description": "2024 US Presidential candidates",
                "market_ids": ["market-a", "market-b"],
            },
            {
                "name": "Senate Race",
                "market_ids": ["market-c", "market-d"],
            },
        ]
        json_path = tmp_path / "groups.json"
        json_path.write_text(json.dumps(json_data))

        result = group_store.import_from_json(json_path)

        assert result["groups_created"] == 2
        assert result["markets_mapped"] == 4
        assert result["errors"] == []

    def test_import_from_json_with_id(self, group_store: ElectionGroupStore, tmp_path: Path) -> None:
        """Test importing JSON with explicit IDs."""
        json_data = [
            {
                "id": "custom-id-123",
                "name": "Custom Group",
                "market_ids": ["m1", "m2"],
            },
        ]
        json_path = tmp_path / "groups.json"
        json_path.write_text(json.dumps(json_data))

        result = group_store.import_from_json(json_path)

        group = group_store.get_group("custom-id-123")
        assert group["name"] == "Custom Group"

    def test_export_to_json(self, group_store: ElectionGroupStore) -> None:
        """Test exporting groups to JSON format."""
        group_store.create_group(name="Group 1", description="Desc 1", market_ids=["m1", "m2"])
        group_store.create_group(name="Group 2", market_ids=["m3"])

        result = group_store.export_to_json()

        assert len(result) == 2
        assert all("id" in g for g in result)
        assert all("name" in g for g in result)
        assert all("market_ids" in g for g in result)


# =============================================================================
# API Integration Tests
# =============================================================================


class TestElectionGroupAPICreate:
    """Test election group creation API."""

    def test_create_group_basic(self, client: TestClient) -> None:
        """Test creating a group via API."""
        response = client.post(
            "/api/election-groups",
            json={"name": "Test Group"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Group"
        assert "id" in data

    def test_create_group_with_markets(self, client: TestClient) -> None:
        """Test creating a group with markets via API."""
        response = client.post(
            "/api/election-groups",
            json={
                "name": "Test Group",
                "description": "Test description",
                "market_ids": ["m1", "m2", "m3"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["market_ids"] == ["m1", "m2", "m3"]


class TestElectionGroupAPIList:
    """Test election group listing API."""

    def test_list_groups_empty(self, client: TestClient) -> None:
        """Test listing when empty."""
        response = client.get("/api/election-groups")

        assert response.status_code == 200
        data = response.json()
        assert data["groups"] == []
        assert data["total_count"] == 0

    def test_list_groups_with_pagination(self, client: TestClient) -> None:
        """Test listing with pagination."""
        # Create 3 groups
        for i in range(3):
            client.post("/api/election-groups", json={"name": f"Group {i}"})

        response = client.get("/api/election-groups?limit=2&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["total_count"] == 3
        assert data["has_more"] is True


class TestElectionGroupAPIGet:
    """Test election group retrieval API."""

    def test_get_group_by_id(self, client: TestClient) -> None:
        """Test getting a group by ID."""
        create_response = client.post(
            "/api/election-groups",
            json={"name": "Test Group", "market_ids": ["m1", "m2"]},
        )
        group_id = create_response.json()["id"]

        response = client.get(f"/api/election-groups/{group_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == group_id
        assert data["market_ids"] == ["m1", "m2"]

    def test_get_nonexistent_group(self, client: TestClient) -> None:
        """Test getting a non-existent group."""
        response = client.get("/api/election-groups/nonexistent-id")

        assert response.status_code == 404


class TestElectionGroupAPIUpdate:
    """Test election group update API."""

    def test_update_group_name(self, client: TestClient) -> None:
        """Test updating group name."""
        create_response = client.post(
            "/api/election-groups",
            json={"name": "Original"},
        )
        group_id = create_response.json()["id"]

        response = client.put(
            f"/api/election-groups/{group_id}",
            json={"name": "Updated"},
        )

        assert response.status_code == 200
        assert response.json()["name"] == "Updated"

    def test_update_group_markets(self, client: TestClient) -> None:
        """Test updating group market IDs."""
        create_response = client.post(
            "/api/election-groups",
            json={"name": "Test", "market_ids": ["m1"]},
        )
        group_id = create_response.json()["id"]

        response = client.put(
            f"/api/election-groups/{group_id}",
            json={"market_ids": ["m2", "m3"]},
        )

        assert response.status_code == 200
        assert response.json()["market_ids"] == ["m2", "m3"]


class TestElectionGroupAPIDelete:
    """Test election group deletion API."""

    def test_delete_group(self, client: TestClient) -> None:
        """Test deleting a group."""
        create_response = client.post(
            "/api/election-groups",
            json={"name": "To Delete"},
        )
        group_id = create_response.json()["id"]

        response = client.delete(f"/api/election-groups/{group_id}")

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify it's gone
        get_response = client.get(f"/api/election-groups/{group_id}")
        assert get_response.status_code == 404


class TestElectionGroupAPIMarketOperations:
    """Test market add/remove operations via API."""

    def test_add_markets_to_group(self, client: TestClient) -> None:
        """Test adding markets via API."""
        create_response = client.post(
            "/api/election-groups",
            json={"name": "Test"},
        )
        group_id = create_response.json()["id"]

        response = client.post(
            f"/api/election-groups/{group_id}/markets",
            json={"market_ids": ["m1", "m2"]},
        )

        assert response.status_code == 200
        assert response.json()["markets_added"] == 2

    def test_remove_markets_from_group(self, client: TestClient) -> None:
        """Test removing markets via API."""
        create_response = client.post(
            "/api/election-groups",
            json={"name": "Test", "market_ids": ["m1", "m2", "m3"]},
        )
        group_id = create_response.json()["id"]

        response = client.post(
            f"/api/election-groups/{group_id}/markets/remove",
            json={"market_ids": ["m1", "m3"]},
        )

        assert response.status_code == 200
        assert response.json()["markets_removed"] == 2


class TestElectionGroupAPIValidation:
    """Test validation and unmapped markets API."""

    def test_validate_groups(self, client: TestClient) -> None:
        """Test group validation endpoint."""
        client.post(
            "/api/election-groups",
            json={"name": "Valid", "market_ids": ["m1", "m2"]},
        )
        client.post(
            "/api/election-groups",
            json={"name": "Invalid", "market_ids": ["m3"]},
        )

        response = client.post("/api/election-groups/validate?min_markets=2")

        assert response.status_code == 200
        data = response.json()
        assert data["valid_groups"] == 1
        assert data["invalid_groups"] == 1

    def test_find_unmapped_markets(self, client: TestClient) -> None:
        """Test finding unmapped markets."""
        client.post(
            "/api/election-groups",
            json={"name": "Test", "market_ids": ["m1", "m2"]},
        )

        response = client.post(
            "/api/election-groups/unmapped",
            json={"market_ids": ["m1", "m2", "m3", "m4"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert set(data["unmapped_market_ids"]) == {"m3", "m4"}
        assert data["unmapped_count"] == 2


class TestElectionGroupAPIMarketLookup:
    """Test looking up election group for a market."""

    def test_get_group_for_market(self, client: TestClient) -> None:
        """Test getting group for a mapped market."""
        create_response = client.post(
            "/api/election-groups",
            json={"name": "Test Group", "market_ids": ["m1", "m2"]},
        )
        group_id = create_response.json()["id"]

        response = client.get("/api/markets/m1/election-group")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == group_id

    def test_get_group_for_unmapped_market(self, client: TestClient) -> None:
        """Test getting group for an unmapped market."""
        response = client.get("/api/markets/unmapped-market/election-group")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"


class TestElectionGroupIntegration:
    """Integration tests for election group workflows."""

    def test_full_crud_workflow(self, client: TestClient) -> None:
        """Test complete CRUD workflow."""
        # Create
        create_response = client.post(
            "/api/election-groups",
            json={
                "name": "2024 Presidential",
                "description": "US Presidential Election",
                "market_ids": ["trump", "biden"],
            },
        )
        assert create_response.status_code == 201
        group_id = create_response.json()["id"]

        # Read
        get_response = client.get(f"/api/election-groups/{group_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "2024 Presidential"

        # Update
        update_response = client.put(
            f"/api/election-groups/{group_id}",
            json={"market_ids": ["trump", "biden", "rfk"]},
        )
        assert update_response.status_code == 200
        assert len(update_response.json()["market_ids"]) == 3

        # List
        list_response = client.get("/api/election-groups")
        assert list_response.json()["total_count"] == 1

        # Delete
        delete_response = client.delete(f"/api/election-groups/{group_id}")
        assert delete_response.status_code == 200

        # Verify deleted
        verify_response = client.get(f"/api/election-groups/{group_id}")
        assert verify_response.status_code == 404

    def test_edit_grouping_without_reimporting_trades(self, client: TestClient) -> None:
        """Test that grouping can be edited without re-importing trades.

        This verifies PRD requirement: 'grouping can be edited without re-importing all trades'.
        """
        # Create initial group
        create_response = client.post(
            "/api/election-groups",
            json={"name": "Test Group", "market_ids": ["m1", "m2"]},
        )
        group_id = create_response.json()["id"]

        # Modify group membership (this doesn't require any trade data operations)
        # Add a market
        add_response = client.post(
            f"/api/election-groups/{group_id}/markets",
            json={"market_ids": ["m3"]},
        )
        assert add_response.status_code == 200

        # Remove a market
        remove_response = client.post(
            f"/api/election-groups/{group_id}/markets/remove",
            json={"market_ids": ["m1"]},
        )
        assert remove_response.status_code == 200

        # Verify the group has correct markets
        get_response = client.get(f"/api/election-groups/{group_id}")
        assert set(get_response.json()["market_ids"]) == {"m2", "m3"}

        # This whole operation happened without touching any trade/parquet data
