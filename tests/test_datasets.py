"""Tests for the Datasets persistence feature."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from polymkt.api.main import app
from polymkt.models.schemas import (
    DatasetCreateRequest,
    DatasetFilters,
    DatasetUpdateRequest,
)
from polymkt.storage.datasets import DatasetNotFoundError, DatasetStore


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_metadata.db"


@pytest.fixture
def dataset_store(temp_db_path: Path) -> DatasetStore:
    """Create a DatasetStore for testing."""
    return DatasetStore(temp_db_path)


# =============================================================================
# DatasetStore unit tests
# =============================================================================


class TestDatasetStoreCreation:
    """Tests for DatasetStore initialization and table creation."""

    def test_init_creates_table(self, temp_db_path: Path) -> None:
        """Test that initializing the store creates the datasets table."""
        store = DatasetStore(temp_db_path)
        assert temp_db_path.exists()

        # Verify table exists by trying to list datasets
        summaries, total = store.list_datasets()
        assert summaries == []
        assert total == 0

    def test_init_creates_parent_dirs(self) -> None:
        """Test that initializing the store creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dirs" / "metadata.db"
            DatasetStore(db_path)
            assert db_path.exists()


class TestDatasetCreate:
    """Tests for creating datasets."""

    def test_create_dataset_basic(self, dataset_store: DatasetStore) -> None:
        """Test creating a basic dataset."""
        request = DatasetCreateRequest(
            name="Test Dataset",
            description="A test dataset",
            market_ids=["market1", "market2", "market3"],
        )
        dataset = dataset_store.create_dataset(request)

        assert dataset.id is not None
        assert len(dataset.id) == 36  # UUID format
        assert dataset.name == "Test Dataset"
        assert dataset.description == "A test dataset"
        assert dataset.market_ids == ["market1", "market2", "market3"]
        assert dataset.excluded_market_ids == []
        assert dataset.filters is None
        assert dataset.created_at is not None
        assert dataset.updated_at is not None
        assert dataset.created_at == dataset.updated_at

    def test_create_dataset_with_filters(self, dataset_store: DatasetStore) -> None:
        """Test creating a dataset with filters."""
        filters = DatasetFilters(
            query="election",
            category="Politics",
            tags=["2024", "presidential"],
            min_volume=1000.0,
        )
        request = DatasetCreateRequest(
            name="Election Dataset",
            market_ids=["m1", "m2"],
            filters=filters,
        )
        dataset = dataset_store.create_dataset(request)

        assert dataset.filters is not None
        assert dataset.filters.query == "election"
        assert dataset.filters.category == "Politics"
        assert dataset.filters.tags == ["2024", "presidential"]
        assert dataset.filters.min_volume == 1000.0

    def test_create_dataset_with_excluded_markets(self, dataset_store: DatasetStore) -> None:
        """Test creating a dataset with excluded markets."""
        request = DatasetCreateRequest(
            name="Filtered Dataset",
            market_ids=["m1", "m2", "m3", "m4"],
            excluded_market_ids=["m2", "m4"],
        )
        dataset = dataset_store.create_dataset(request)

        assert dataset.market_ids == ["m1", "m2", "m3", "m4"]
        assert dataset.excluded_market_ids == ["m2", "m4"]

    def test_create_multiple_datasets(self, dataset_store: DatasetStore) -> None:
        """Test creating multiple datasets."""
        for i in range(5):
            request = DatasetCreateRequest(
                name=f"Dataset {i}",
                market_ids=[f"market_{i}"],
            )
            dataset_store.create_dataset(request)

        summaries, total = dataset_store.list_datasets()
        assert total == 5
        assert len(summaries) == 5


class TestDatasetGet:
    """Tests for getting datasets by ID."""

    def test_get_dataset_by_id(self, dataset_store: DatasetStore) -> None:
        """Test getting a dataset by ID."""
        request = DatasetCreateRequest(
            name="Get Test",
            market_ids=["m1"],
        )
        created = dataset_store.create_dataset(request)

        retrieved = dataset_store.get_dataset(created.id)
        assert retrieved.id == created.id
        assert retrieved.name == created.name
        assert retrieved.market_ids == created.market_ids

    def test_get_nonexistent_dataset_raises(self, dataset_store: DatasetStore) -> None:
        """Test that getting a nonexistent dataset raises an error."""
        with pytest.raises(DatasetNotFoundError):
            dataset_store.get_dataset("nonexistent-uuid")

    def test_get_dataset_with_filters(self, dataset_store: DatasetStore) -> None:
        """Test that filters are properly retrieved."""
        filters = DatasetFilters(
            query="test query",
            closed_time_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_time_max=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        request = DatasetCreateRequest(
            name="Filter Test",
            market_ids=["m1"],
            filters=filters,
        )
        created = dataset_store.create_dataset(request)

        retrieved = dataset_store.get_dataset(created.id)
        assert retrieved.filters is not None
        assert retrieved.filters.query == "test query"
        assert retrieved.filters.closed_time_min == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert retrieved.filters.closed_time_max == datetime(2024, 12, 31, tzinfo=timezone.utc)


class TestDatasetUpdate:
    """Tests for updating datasets."""

    def test_update_dataset_name(self, dataset_store: DatasetStore) -> None:
        """Test updating just the dataset name."""
        request = DatasetCreateRequest(name="Original Name", market_ids=["m1"])
        created = dataset_store.create_dataset(request)

        update_request = DatasetUpdateRequest(name="Updated Name")
        updated = dataset_store.update_dataset(created.id, update_request)

        assert updated.name == "Updated Name"
        assert updated.market_ids == ["m1"]  # Unchanged
        assert updated.updated_at > created.updated_at

    def test_update_dataset_market_ids(self, dataset_store: DatasetStore) -> None:
        """Test updating the market list."""
        request = DatasetCreateRequest(name="Test", market_ids=["m1", "m2"])
        created = dataset_store.create_dataset(request)

        update_request = DatasetUpdateRequest(market_ids=["m1", "m2", "m3", "m4"])
        updated = dataset_store.update_dataset(created.id, update_request)

        assert updated.market_ids == ["m1", "m2", "m3", "m4"]
        assert updated.name == "Test"  # Unchanged

    def test_update_dataset_excluded_markets(self, dataset_store: DatasetStore) -> None:
        """Test updating excluded markets."""
        request = DatasetCreateRequest(name="Test", market_ids=["m1", "m2", "m3"])
        created = dataset_store.create_dataset(request)

        update_request = DatasetUpdateRequest(excluded_market_ids=["m2"])
        updated = dataset_store.update_dataset(created.id, update_request)

        assert updated.excluded_market_ids == ["m2"]

    def test_update_dataset_filters(self, dataset_store: DatasetStore) -> None:
        """Test updating filters."""
        request = DatasetCreateRequest(name="Test", market_ids=["m1"])
        created = dataset_store.create_dataset(request)
        assert created.filters is None

        new_filters = DatasetFilters(query="updated query", category="Sports")
        update_request = DatasetUpdateRequest(filters=new_filters)
        updated = dataset_store.update_dataset(created.id, update_request)

        assert updated.filters is not None
        assert updated.filters.query == "updated query"
        assert updated.filters.category == "Sports"

    def test_update_nonexistent_dataset_raises(self, dataset_store: DatasetStore) -> None:
        """Test that updating a nonexistent dataset raises an error."""
        update_request = DatasetUpdateRequest(name="New Name")
        with pytest.raises(DatasetNotFoundError):
            dataset_store.update_dataset("nonexistent-uuid", update_request)

    def test_update_preserves_created_at(self, dataset_store: DatasetStore) -> None:
        """Test that updating preserves the original created_at timestamp."""
        request = DatasetCreateRequest(name="Test", market_ids=["m1"])
        created = dataset_store.create_dataset(request)
        original_created_at = created.created_at

        update_request = DatasetUpdateRequest(name="Updated")
        updated = dataset_store.update_dataset(created.id, update_request)

        assert updated.created_at == original_created_at


class TestDatasetDelete:
    """Tests for deleting datasets."""

    def test_delete_dataset(self, dataset_store: DatasetStore) -> None:
        """Test deleting a dataset."""
        request = DatasetCreateRequest(name="To Delete", market_ids=["m1"])
        created = dataset_store.create_dataset(request)

        dataset_store.delete_dataset(created.id)

        with pytest.raises(DatasetNotFoundError):
            dataset_store.get_dataset(created.id)

    def test_delete_nonexistent_dataset_raises(self, dataset_store: DatasetStore) -> None:
        """Test that deleting a nonexistent dataset raises an error."""
        with pytest.raises(DatasetNotFoundError):
            dataset_store.delete_dataset("nonexistent-uuid")

    def test_delete_updates_count(self, dataset_store: DatasetStore) -> None:
        """Test that deleting updates the total count."""
        for i in range(3):
            request = DatasetCreateRequest(name=f"Dataset {i}", market_ids=["m1"])
            dataset_store.create_dataset(request)

        _, total = dataset_store.list_datasets()
        assert total == 3

        summaries, _ = dataset_store.list_datasets()
        dataset_store.delete_dataset(summaries[0].id)

        _, total = dataset_store.list_datasets()
        assert total == 2


class TestDatasetList:
    """Tests for listing datasets."""

    def test_list_empty(self, dataset_store: DatasetStore) -> None:
        """Test listing when no datasets exist."""
        summaries, total = dataset_store.list_datasets()
        assert summaries == []
        assert total == 0

    def test_list_returns_summaries(self, dataset_store: DatasetStore) -> None:
        """Test that list returns DatasetSummary objects."""
        request = DatasetCreateRequest(
            name="Summary Test",
            description="Test description",
            market_ids=["m1", "m2", "m3"],
        )
        dataset_store.create_dataset(request)

        summaries, total = dataset_store.list_datasets()
        assert total == 1
        assert len(summaries) == 1

        summary = summaries[0]
        assert summary.name == "Summary Test"
        assert summary.description == "Test description"
        assert summary.market_count == 3

    def test_list_pagination_limit(self, dataset_store: DatasetStore) -> None:
        """Test pagination with limit."""
        for i in range(10):
            request = DatasetCreateRequest(name=f"Dataset {i}", market_ids=["m1"])
            dataset_store.create_dataset(request)

        summaries, total = dataset_store.list_datasets(limit=5)
        assert len(summaries) == 5
        assert total == 10

    def test_list_pagination_offset(self, dataset_store: DatasetStore) -> None:
        """Test pagination with offset."""
        for i in range(10):
            request = DatasetCreateRequest(name=f"Dataset {i}", market_ids=["m1"])
            dataset_store.create_dataset(request)

        summaries, total = dataset_store.list_datasets(limit=5, offset=5)
        assert len(summaries) == 5
        assert total == 10

    def test_list_ordered_by_updated_at(self, dataset_store: DatasetStore) -> None:
        """Test that results are ordered by updated_at descending."""
        import time

        # Create datasets with slight delay to ensure different timestamps
        for i in range(3):
            request = DatasetCreateRequest(name=f"Dataset {i}", market_ids=["m1"])
            dataset_store.create_dataset(request)
            time.sleep(0.01)  # Small delay

        summaries, _ = dataset_store.list_datasets()

        # Most recently created should be first
        assert summaries[0].name == "Dataset 2"
        assert summaries[1].name == "Dataset 1"
        assert summaries[2].name == "Dataset 0"


# =============================================================================
# API endpoint tests
# =============================================================================


class TestDatasetsAPICreate:
    """Tests for POST /api/datasets endpoint."""

    def test_create_dataset_api(self, client: TestClient) -> None:
        """Test creating a dataset via API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.post(
                    "/api/datasets",
                    json={
                        "name": "API Test Dataset",
                        "description": "Created via API",
                        "market_ids": ["market1", "market2"],
                    },
                )

                assert response.status_code == 201
                data = response.json()
                assert data["name"] == "API Test Dataset"
                assert data["description"] == "Created via API"
                assert data["market_ids"] == ["market1", "market2"]
                assert "id" in data
                assert "created_at" in data
                assert "updated_at" in data

    def test_create_dataset_with_filters_api(self, client: TestClient) -> None:
        """Test creating a dataset with filters via API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.post(
                    "/api/datasets",
                    json={
                        "name": "Filtered Dataset",
                        "market_ids": ["m1"],
                        "filters": {
                            "query": "election",
                            "category": "Politics",
                        },
                    },
                )

                assert response.status_code == 201
                data = response.json()
                assert data["filters"]["query"] == "election"
                assert data["filters"]["category"] == "Politics"

    def test_create_dataset_validation_error(self, client: TestClient) -> None:
        """Test that validation errors return 422."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Missing required market_ids
                response = client.post(
                    "/api/datasets",
                    json={"name": "Invalid Dataset"},
                )

                assert response.status_code == 422

    def test_create_dataset_empty_name_error(self, client: TestClient) -> None:
        """Test that empty name returns validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.post(
                    "/api/datasets",
                    json={"name": "", "market_ids": ["m1"]},
                )

                assert response.status_code == 422

    def test_create_dataset_empty_market_ids_error(self, client: TestClient) -> None:
        """Test that empty market_ids returns validation error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.post(
                    "/api/datasets",
                    json={"name": "Test", "market_ids": []},
                )

                assert response.status_code == 422


class TestDatasetsAPIList:
    """Tests for GET /api/datasets endpoint."""

    def test_list_datasets_empty(self, client: TestClient) -> None:
        """Test listing datasets when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.get("/api/datasets")

                assert response.status_code == 200
                data = response.json()
                assert data["datasets"] == []
                assert data["count"] == 0
                assert data["total_count"] == 0
                assert data["has_more"] is False

    def test_list_datasets_with_data(self, client: TestClient) -> None:
        """Test listing datasets with data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Create a dataset first
                client.post(
                    "/api/datasets",
                    json={"name": "Test", "market_ids": ["m1", "m2"]},
                )

                response = client.get("/api/datasets")

                assert response.status_code == 200
                data = response.json()
                assert len(data["datasets"]) == 1
                assert data["datasets"][0]["name"] == "Test"
                assert data["datasets"][0]["market_count"] == 2
                assert data["count"] == 1
                assert data["total_count"] == 1

    def test_list_datasets_pagination(self, client: TestClient) -> None:
        """Test pagination in list endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Create multiple datasets
                for i in range(10):
                    client.post(
                        "/api/datasets",
                        json={"name": f"Dataset {i}", "market_ids": ["m1"]},
                    )

                # Get first page
                response = client.get("/api/datasets?limit=5&offset=0")
                assert response.status_code == 200
                data = response.json()
                assert len(data["datasets"]) == 5
                assert data["total_count"] == 10
                assert data["has_more"] is True

                # Get second page
                response = client.get("/api/datasets?limit=5&offset=5")
                data = response.json()
                assert len(data["datasets"]) == 5
                assert data["has_more"] is False


class TestDatasetsAPIGet:
    """Tests for GET /api/datasets/{dataset_id} endpoint."""

    def test_get_dataset(self, client: TestClient) -> None:
        """Test getting a single dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Create a dataset
                create_response = client.post(
                    "/api/datasets",
                    json={"name": "Test", "market_ids": ["m1", "m2"]},
                )
                dataset_id = create_response.json()["id"]

                # Get the dataset
                response = client.get(f"/api/datasets/{dataset_id}")

                assert response.status_code == 200
                data = response.json()
                assert data["id"] == dataset_id
                assert data["name"] == "Test"
                assert data["market_ids"] == ["m1", "m2"]

    def test_get_dataset_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.get("/api/datasets/nonexistent-uuid")

                assert response.status_code == 404
                assert "not found" in response.json()["detail"].lower()


class TestDatasetsAPIUpdate:
    """Tests for PUT /api/datasets/{dataset_id} endpoint."""

    def test_update_dataset(self, client: TestClient) -> None:
        """Test updating a dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Create a dataset
                create_response = client.post(
                    "/api/datasets",
                    json={"name": "Original", "market_ids": ["m1"]},
                )
                dataset_id = create_response.json()["id"]

                # Update the dataset
                response = client.put(
                    f"/api/datasets/{dataset_id}",
                    json={"name": "Updated", "market_ids": ["m1", "m2", "m3"]},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "Updated"
                assert data["market_ids"] == ["m1", "m2", "m3"]

    def test_update_dataset_partial(self, client: TestClient) -> None:
        """Test partial update of a dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Create a dataset
                create_response = client.post(
                    "/api/datasets",
                    json={
                        "name": "Original",
                        "description": "Original desc",
                        "market_ids": ["m1", "m2"],
                    },
                )
                dataset_id = create_response.json()["id"]

                # Update only the name
                response = client.put(
                    f"/api/datasets/{dataset_id}",
                    json={"name": "Updated"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "Updated"
                assert data["description"] == "Original desc"  # Unchanged
                assert data["market_ids"] == ["m1", "m2"]  # Unchanged

    def test_update_dataset_not_found(self, client: TestClient) -> None:
        """Test updating a non-existent dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.put(
                    "/api/datasets/nonexistent-uuid",
                    json={"name": "Updated"},
                )

                assert response.status_code == 404


class TestDatasetsAPIDelete:
    """Tests for DELETE /api/datasets/{dataset_id} endpoint."""

    def test_delete_dataset(self, client: TestClient) -> None:
        """Test deleting a dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Create a dataset
                create_response = client.post(
                    "/api/datasets",
                    json={"name": "To Delete", "market_ids": ["m1"]},
                )
                dataset_id = create_response.json()["id"]

                # Delete the dataset
                response = client.delete(f"/api/datasets/{dataset_id}")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "deleted"
                assert data["dataset_id"] == dataset_id

                # Verify it's gone
                get_response = client.get(f"/api/datasets/{dataset_id}")
                assert get_response.status_code == 404

    def test_delete_dataset_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                response = client.delete("/api/datasets/nonexistent-uuid")

                assert response.status_code == 404


class TestDatasetsIntegration:
    """Integration tests for the full dataset workflow."""

    def test_full_crud_workflow(self, client: TestClient) -> None:
        """Test the complete CRUD workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # CREATE
                create_response = client.post(
                    "/api/datasets",
                    json={
                        "name": "Integration Test",
                        "description": "Testing full workflow",
                        "market_ids": ["m1", "m2", "m3"],
                        "filters": {"query": "test", "category": "Sports"},
                    },
                )
                assert create_response.status_code == 201
                dataset_id = create_response.json()["id"]

                # READ
                get_response = client.get(f"/api/datasets/{dataset_id}")
                assert get_response.status_code == 200
                assert get_response.json()["name"] == "Integration Test"

                # LIST
                list_response = client.get("/api/datasets")
                assert list_response.status_code == 200
                assert list_response.json()["total_count"] == 1

                # UPDATE - modify market list
                update_response = client.put(
                    f"/api/datasets/{dataset_id}",
                    json={
                        "market_ids": ["m1", "m2", "m3", "m4", "m5"],
                        "excluded_market_ids": ["m3"],
                    },
                )
                assert update_response.status_code == 200
                assert len(update_response.json()["market_ids"]) == 5
                assert update_response.json()["excluded_market_ids"] == ["m3"]

                # Verify update persisted
                get_response2 = client.get(f"/api/datasets/{dataset_id}")
                assert get_response2.json()["market_ids"] == ["m1", "m2", "m3", "m4", "m5"]

                # DELETE
                delete_response = client.delete(f"/api/datasets/{dataset_id}")
                assert delete_response.status_code == 200

                # Verify deletion
                list_response2 = client.get("/api/datasets")
                assert list_response2.json()["total_count"] == 0

    def test_multiple_datasets_workflow(self, client: TestClient) -> None:
        """Test working with multiple datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("polymkt.api.main.settings") as mock_settings:
                mock_settings.metadata_db_path = Path(tmpdir) / "metadata.db"

                # Create multiple datasets
                dataset_ids = []
                for i in range(5):
                    response = client.post(
                        "/api/datasets",
                        json={
                            "name": f"Dataset {i}",
                            "market_ids": [f"m{j}" for j in range(i + 1)],
                        },
                    )
                    assert response.status_code == 201
                    dataset_ids.append(response.json()["id"])

                # List and verify counts
                list_response = client.get("/api/datasets")
                assert list_response.json()["total_count"] == 5

                # Verify market counts
                datasets = list_response.json()["datasets"]
                market_counts = sorted([d["market_count"] for d in datasets])
                assert market_counts == [1, 2, 3, 4, 5]

                # Delete some
                client.delete(f"/api/datasets/{dataset_ids[0]}")
                client.delete(f"/api/datasets/{dataset_ids[2]}")

                # Verify remaining
                list_response2 = client.get("/api/datasets")
                assert list_response2.json()["total_count"] == 3
