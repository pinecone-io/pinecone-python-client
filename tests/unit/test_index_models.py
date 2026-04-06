"""Unit tests for index response models."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import ByocSpec, PodSpec, ServerlessSpec
from tests.factories import make_index_list_response, make_index_response


class TestIndexStatus:
    def test_construct(self) -> None:
        status = IndexStatus(ready=True, state="Ready")
        assert status.ready is True
        assert status.state == "Ready"

    def test_not_ready(self) -> None:
        status = IndexStatus(ready=False, state="Initializing")
        assert status.ready is False
        assert status.state == "Initializing"


class TestIndexModel:
    def test_from_factory_dict(self) -> None:
        data = make_index_response()
        model = msgspec.convert(data, IndexModel)
        assert model.name == "test-index"
        assert model.dimension == 1536
        assert model.metric == "cosine"
        assert model.host == "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
        assert model.deletion_protection == "disabled"
        assert model.vector_type == "dense"
        assert model.status.ready is True
        assert model.status.state == "Ready"
        assert "serverless" in model.spec
        assert model.tags == {}

    def test_bracket_access(self) -> None:
        data = make_index_response()
        model = msgspec.convert(data, IndexModel)
        assert model["name"] == "test-index"
        assert model["dimension"] == 1536
        assert model["metric"] == "cosine"
        assert model["host"] == model.host
        assert model["status"].ready is True

    def test_bracket_access_missing_key(self) -> None:
        data = make_index_response()
        model = msgspec.convert(data, IndexModel)
        with pytest.raises(KeyError, match="nonexistent"):
            model["nonexistent"]

    def test_optional_dimension_none(self) -> None:
        data = make_index_response()
        del data["dimension"]
        model = msgspec.convert(data, IndexModel)
        assert model.dimension is None

    def test_optional_tags_none(self) -> None:
        data = make_index_response()
        del data["tags"]
        model = msgspec.convert(data, IndexModel)
        assert model.tags is None

    def test_default_vector_type(self) -> None:
        data = make_index_response()
        del data["vector_type"]
        model = msgspec.convert(data, IndexModel)
        assert model.vector_type == "dense"

    def test_default_deletion_protection(self) -> None:
        data = make_index_response()
        del data["deletion_protection"]
        model = msgspec.convert(data, IndexModel)
        assert model.deletion_protection == "disabled"

    def test_pod_spec(self) -> None:
        data = make_index_response(
            spec={
                "pod": {
                    "environment": "us-east1-gcp",
                    "pod_type": "p1.x1",
                    "replicas": 1,
                    "shards": 1,
                    "pods": 1,
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert "pod" in model.spec
        assert model.spec["pod"]["environment"] == "us-east1-gcp"

    def test_enum_string_values(self) -> None:
        """Both enum values and plain strings work since we store as str."""
        data = make_index_response(metric="euclidean", vector_type="sparse")
        model = msgspec.convert(data, IndexModel)
        assert model.metric == "euclidean"
        assert model.vector_type == "sparse"


class TestIndexList:
    def _make_list(self) -> IndexList:
        data = make_index_list_response(
            indexes=[
                make_index_response(name="index-a"),
                make_index_response(name="index-b"),
                make_index_response(name="index-c"),
            ]
        )
        indexes = [msgspec.convert(idx, IndexModel) for idx in data["indexes"]]
        return IndexList(indexes)

    def test_iteration(self) -> None:
        index_list = self._make_list()
        names = [idx.name for idx in index_list]
        assert names == ["index-a", "index-b", "index-c"]

    def test_len(self) -> None:
        index_list = self._make_list()
        assert len(index_list) == 3

    def test_getitem(self) -> None:
        index_list = self._make_list()
        assert index_list[0].name == "index-a"
        assert index_list[2].name == "index-c"

    def test_getitem_negative(self) -> None:
        index_list = self._make_list()
        assert index_list[-1].name == "index-c"

    def test_names(self) -> None:
        index_list = self._make_list()
        assert index_list.names() == ["index-a", "index-b", "index-c"]

    def test_empty_list(self) -> None:
        index_list = IndexList([])
        assert len(index_list) == 0
        assert index_list.names() == []
        assert list(index_list) == []


class TestServerlessSpec:
    def test_construct_and_encode(self) -> None:
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        encoded = msgspec.json.encode(spec)
        decoded: dict[str, Any] = msgspec.json.decode(encoded)
        assert decoded == {"cloud": "aws", "region": "us-east-1"}


class TestPodSpec:
    def test_construct_with_defaults(self) -> None:
        spec = PodSpec(environment="us-east1-gcp")
        assert spec.pod_type == "p1.x1"
        assert spec.replicas == 1
        assert spec.shards == 1
        assert spec.pods == 1
        assert spec.metadata_config is None
        assert spec.source_collection is None

    def test_construct_with_overrides(self) -> None:
        spec = PodSpec(
            environment="us-east1-gcp",
            pod_type="p2.x1",
            replicas=2,
            shards=2,
            pods=4,
            metadata_config={"indexed": ["genre"]},
            source_collection="my-collection",
        )
        assert spec.pod_type == "p2.x1"
        assert spec.replicas == 2
        assert spec.pods == 4
        assert spec.metadata_config == {"indexed": ["genre"]}
        assert spec.source_collection == "my-collection"

    def test_encode(self) -> None:
        spec = PodSpec(environment="us-east1-gcp")
        encoded = msgspec.json.encode(spec)
        decoded: dict[str, Any] = msgspec.json.decode(encoded)
        assert decoded["environment"] == "us-east1-gcp"
        assert decoded["pod_type"] == "p1.x1"


class TestByocSpec:
    def test_construct_and_encode(self) -> None:
        spec = ByocSpec(cloud="aws", region="us-east-1")
        encoded = msgspec.json.encode(spec)
        decoded: dict[str, Any] = msgspec.json.decode(encoded)
        assert decoded == {"cloud": "aws", "region": "us-east-1"}


class TestReExports:
    """Verify models are importable from the top-level models package."""

    def test_import_from_models(self) -> None:
        from pinecone.models import (
            ByocSpec,
            IndexList,
            IndexModel,
            IndexStatus,
            PodSpec,
            ServerlessSpec,
        )

        assert IndexModel is not None
        assert IndexStatus is not None
        assert IndexList is not None
        assert ServerlessSpec is not None
        assert PodSpec is not None
        assert ByocSpec is not None
