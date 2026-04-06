"""Unit tests for collection response models."""

from __future__ import annotations

import msgspec
import pytest

from pinecone.models.collections.collection_list import CollectionList
from pinecone.models.collections.collection_model import CollectionModel


class TestCollectionModel:
    """Tests for CollectionModel struct."""

    def test_attribute_access_required_fields(self) -> None:
        model = CollectionModel(
            name="my-collection",
            status="Ready",
            environment="us-east1-gcp",
        )
        assert model.name == "my-collection"
        assert model.status == "Ready"
        assert model.environment == "us-east1-gcp"

    def test_attribute_access_all_fields(self) -> None:
        model = CollectionModel(
            name="my-collection",
            status="Ready",
            environment="us-east1-gcp",
            size=10_000_000,
            dimension=1536,
            vector_count=120_000,
        )
        assert model.size == 10_000_000
        assert model.dimension == 1536
        assert model.vector_count == 120_000

    def test_optional_fields_default_to_none(self) -> None:
        model = CollectionModel(
            name="my-collection",
            status="Initializing",
            environment="us-east1-gcp",
        )
        assert model.size is None
        assert model.dimension is None
        assert model.vector_count is None

    def test_bracket_access(self) -> None:
        model = CollectionModel(
            name="my-collection",
            status="Ready",
            environment="us-east1-gcp",
            size=5_000_000,
        )
        assert model["name"] == "my-collection"
        assert model["status"] == "Ready"
        assert model["environment"] == "us-east1-gcp"
        assert model["size"] == 5_000_000

    def test_bracket_access_missing_key_raises_key_error(self) -> None:
        model = CollectionModel(
            name="my-collection",
            status="Ready",
            environment="us-east1-gcp",
        )
        with pytest.raises(KeyError, match="nonexistent"):
            model["nonexistent"]

    def test_construct_via_msgspec_convert(self) -> None:
        data = {
            "name": "test-collection",
            "status": "Ready",
            "environment": "us-east1-gcp",
            "size": 10_000_000,
            "dimension": 1536,
            "vector_count": 120_000,
        }
        model = msgspec.convert(data, CollectionModel)
        assert model.name == "test-collection"
        assert model.status == "Ready"
        assert model.environment == "us-east1-gcp"
        assert model.size == 10_000_000
        assert model.dimension == 1536
        assert model.vector_count == 120_000

    def test_construct_via_msgspec_convert_minimal(self) -> None:
        data = {
            "name": "minimal",
            "status": "Initializing",
            "environment": "us-west2-gcp",
        }
        model = msgspec.convert(data, CollectionModel)
        assert model.name == "minimal"
        assert model.size is None
        assert model.dimension is None
        assert model.vector_count is None


class TestCollectionList:
    """Tests for CollectionList wrapper."""

    def _make_models(self, count: int = 3) -> list[CollectionModel]:
        return [
            CollectionModel(
                name=f"collection-{i}",
                status="Ready",
                environment="us-east1-gcp",
            )
            for i in range(count)
        ]

    def test_iteration(self) -> None:
        models = self._make_models()
        cl = CollectionList(collections=models)
        items = list(cl)
        assert len(items) == 3
        assert all(isinstance(item, CollectionModel) for item in items)

    def test_len(self) -> None:
        models = self._make_models(5)
        cl = CollectionList(collections=models)
        assert len(cl) == 5

    def test_integer_index_access(self) -> None:
        models = self._make_models()
        cl = CollectionList(collections=models)
        assert cl[0].name == "collection-0"
        assert cl[2].name == "collection-2"

    def test_names(self) -> None:
        models = self._make_models()
        cl = CollectionList(collections=models)
        assert cl.names() == ["collection-0", "collection-1", "collection-2"]

    def test_empty_list(self) -> None:
        cl = CollectionList(collections=[])
        assert len(cl) == 0
        assert cl.names() == []
        assert list(cl) == []

    def test_repr(self) -> None:
        cl = CollectionList(collections=[])
        assert repr(cl) == "CollectionList(collections=[])"
