"""Unit tests for index response models."""

from __future__ import annotations

from typing import Any

import msgspec
import pytest

from pinecone.models.indexes.index import (
    ByocSpecInfo,
    IndexModel,
    IndexSpec,
    IndexStatus,
    PodSpecInfo,
    ServerlessSpecInfo,
)
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
        assert model.host == "https://test-index-abc1234.svc.us-east1-gcp.pinecone.io"
        assert model.deletion_protection == "disabled"
        assert model.vector_type == "dense"
        assert model.status.ready is True
        assert model.status.state == "Ready"
        assert isinstance(model.spec, IndexSpec)
        assert model.spec.serverless is not None
        assert isinstance(model.spec.serverless, ServerlessSpecInfo)
        assert model.spec.serverless.cloud == "aws"
        assert model.spec.serverless.region == "us-east-1"
        assert model.spec.pod is None
        assert model.spec.byoc is None
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
        assert model.spec.pod is not None
        assert isinstance(model.spec.pod, PodSpecInfo)
        assert model.spec.pod.environment == "us-east1-gcp"
        assert model.spec.pod.pod_type == "p1.x1"
        assert model.spec.pod.replicas == 1
        assert model.spec.pod.shards == 1
        assert model.spec.pod.pods == 1
        assert model.spec.pod.metadata_config is None
        assert model.spec.pod.source_collection is None
        assert model.spec.serverless is None

    def test_byoc_spec(self) -> None:
        data = make_index_response(
            spec={
                "byoc": {
                    "environment": "aws-us-east-1-b921",
                    "read_capacity": {"mode": "OnDemand"},
                }
            }
        )
        model = msgspec.convert(data, IndexModel)
        assert model.spec.byoc is not None
        assert isinstance(model.spec.byoc, ByocSpecInfo)
        assert model.spec.byoc.environment == "aws-us-east-1-b921"
        assert model.spec.byoc.read_capacity == {"mode": "OnDemand"}
        assert model.spec.serverless is None
        assert model.spec.pod is None

    def test_byoc_spec_no_read_capacity(self) -> None:
        data = make_index_response(spec={"byoc": {"environment": "aws-us-east-1-b921"}})
        model = msgspec.convert(data, IndexModel)
        assert model.spec.byoc is not None
        assert model.spec.byoc.read_capacity is None

    def test_enum_string_values(self) -> None:
        """Both enum values and plain strings work since we store as str."""
        data = make_index_response(metric="euclidean", vector_type="sparse")
        model = msgspec.convert(data, IndexModel)
        assert model.metric == "euclidean"
        assert model.vector_type == "sparse"

    def test_host_bare_gets_https_prefix(self) -> None:
        """IndexModel normalizes bare hostname to https:// on construction."""
        data = make_index_response(host="my-index-abc.svc.pinecone.io")
        model = msgspec.convert(data, IndexModel)
        assert model.host == "https://my-index-abc.svc.pinecone.io"

    def test_host_with_https_unchanged(self) -> None:
        """IndexModel preserves an already-prefixed https:// host."""
        data = make_index_response(host="https://my-index-abc.svc.pinecone.io")
        model = msgspec.convert(data, IndexModel)
        assert model.host == "https://my-index-abc.svc.pinecone.io"

    def test_index_model_null_host(self) -> None:
        """IndexModel must decode null host from backend without raising."""
        raw = b'{"name":"test","metric":"cosine","host":null,"status":{"ready":false,"state":"Initializing"},"spec":{"serverless":{"cloud":"aws","region":"us-east-1"}},"deletion_protection":"disabled","vector_type":"dense"}'
        model = msgspec.json.decode(raw, type=IndexModel)
        assert model.host is None
        assert model.name == "test"

    def test_index_model_missing_host(self) -> None:
        """IndexModel must decode when host field is absent from backend response."""
        raw = b'{"name":"test","metric":"cosine","status":{"ready":false,"state":"Initializing"},"spec":{"serverless":{"cloud":"aws","region":"us-east-1"}},"deletion_protection":"disabled","vector_type":"dense"}'
        model = msgspec.json.decode(raw, type=IndexModel)
        assert model.host is None
        assert model.name == "test"

    def test_index_model_non_null_host_normalized(self) -> None:
        """IndexModel still normalizes non-null host with https://."""
        raw = b'{"name":"test","metric":"cosine","host":"index-host.pinecone.io","status":{"ready":true,"state":"Ready"},"spec":{"serverless":{"cloud":"aws","region":"us-east-1"}},"deletion_protection":"disabled","vector_type":"dense"}'
        model = msgspec.json.decode(raw, type=IndexModel)
        assert model.host == "https://index-host.pinecone.io"

    def test_index_model_pod_spec_null_replicas(self) -> None:
        """PodSpecInfo must decode null replicas/shards/pods from backend."""
        raw = (
            b'{"name":"test","metric":"cosine","host":null,'
            b'"status":{"ready":false,"state":"Initializing"},'
            b'"spec":{"pod":{"environment":"us-east1-gcp","pod_type":"p1",'
            b'"replicas":null,"shards":null,"pods":1}},'
            b'"deletion_protection":"disabled","vector_type":"dense"}'
        )
        model = msgspec.json.decode(raw, type=IndexModel)
        assert model.spec.pod is not None
        assert model.spec.pod.replicas is None
        assert model.spec.pod.shards is None
        assert model.spec.pod.pods == 1

    def test_index_model_pod_spec_explicit_replicas(self) -> None:
        """PodSpecInfo still decodes explicit replicas/shards correctly."""
        raw = (
            b'{"name":"test","metric":"cosine","host":null,'
            b'"status":{"ready":false,"state":"Initializing"},'
            b'"spec":{"pod":{"environment":"us-east1-gcp","pod_type":"p1",'
            b'"replicas":2,"shards":1,"pods":2}},'
            b'"deletion_protection":"disabled","vector_type":"dense"}'
        )
        model = msgspec.json.decode(raw, type=IndexModel)
        assert model.spec.pod is not None
        assert model.spec.pod.replicas == 2
        assert model.spec.pod.shards == 1
        assert model.spec.pod.pods == 2

    def test_index_model_private_host_decoded(self) -> None:
        """IndexModel must expose private_host when returned by backend."""
        raw = (
            b'{"name":"test","metric":"cosine",'
            b'"host":"test.svc.pinecone.io",'
            b'"private_host":"test.svc.private.pinecone.io",'
            b'"status":{"ready":true,"state":"Ready"},'
            b'"spec":{"serverless":{"cloud":"aws","region":"us-east-1"}},'
            b'"deletion_protection":"disabled","vector_type":"dense"}'
        )
        model = msgspec.json.decode(raw, type=IndexModel)
        assert model.private_host == "https://test.svc.private.pinecone.io"
        assert model.host == "https://test.svc.pinecone.io"

    def test_index_model_private_host_absent(self) -> None:
        """IndexModel.private_host is None when backend omits the field."""
        raw = (
            b'{"name":"test","metric":"cosine","host":"test.svc.pinecone.io",'
            b'"status":{"ready":true,"state":"Ready"},'
            b'"spec":{"serverless":{"cloud":"aws","region":"us-east-1"}},'
            b'"deletion_protection":"disabled","vector_type":"dense"}'
        )
        model = msgspec.json.decode(raw, type=IndexModel)
        assert model.private_host is None


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

    def test_asdict_minimal(self) -> None:
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        result = spec.asdict()
        assert result == {"serverless": {"cloud": "aws", "region": "us-east-1"}}

    def test_asdict_with_read_capacity(self) -> None:
        spec = ServerlessSpec(cloud="aws", region="us-east-1", read_capacity={"mode": "OnDemand"})
        result = spec.asdict()
        assert result["serverless"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["serverless"]["cloud"] == "aws"
        assert result["serverless"]["region"] == "us-east-1"

    def test_asdict_with_schema(self) -> None:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            schema={"fields": {"genre": {"type": "string"}}},
        )
        result = spec.asdict()
        assert result["serverless"]["schema"] == {"fields": {"genre": {"type": "string"}}}
        assert "read_capacity" not in result["serverless"]

    def test_asdict_with_all_optional(self) -> None:
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            read_capacity={"mode": "OnDemand"},
            schema={"fields": {"genre": {"type": "string"}}},
        )
        result = spec.asdict()
        assert result["serverless"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["serverless"]["schema"] == {"fields": {"genre": {"type": "string"}}}


class TestPodSpec:
    def test_asdict_defaults(self) -> None:
        spec = PodSpec(environment="us-east1-gcp")
        result = spec.asdict()
        assert "pod" in result
        pod = result["pod"]
        assert pod["environment"] == "us-east1-gcp"
        assert pod["pod_type"] == "p1.x1"
        assert pod["replicas"] == 1
        assert pod["shards"] == 1
        assert pod["pods"] == 1
        assert "metadata_config" not in pod
        assert "source_collection" not in pod

    def test_asdict_with_metadata_config(self) -> None:
        spec = PodSpec(environment="us-east1-gcp", metadata_config={"indexed": ["genre"]})
        result = spec.asdict()
        assert result["pod"]["metadata_config"] == {"indexed": ["genre"]}
        assert "source_collection" not in result["pod"]

    def test_asdict_with_source_collection(self) -> None:
        spec = PodSpec(environment="us-east1-gcp", source_collection="my-coll")
        result = spec.asdict()
        assert result["pod"]["source_collection"] == "my-coll"
        assert "metadata_config" not in result["pod"]

    def test_asdict_with_all_optional(self) -> None:
        spec = PodSpec(
            environment="us-east1-gcp",
            metadata_config={"indexed": ["genre"]},
            source_collection="my-coll",
        )
        result = spec.asdict()
        assert result["pod"]["metadata_config"] == {"indexed": ["genre"]}
        assert result["pod"]["source_collection"] == "my-coll"

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
    def test_asdict_minimal(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921")
        result = spec.asdict()
        assert result == {"byoc": {"environment": "aws-us-east-1-b921"}}

    def test_asdict_with_read_capacity(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921", read_capacity={"mode": "OnDemand"})
        result = spec.asdict()
        assert result["byoc"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["byoc"]["environment"] == "aws-us-east-1-b921"

    def test_asdict_with_schema(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921", schema={"fields": {}})
        result = spec.asdict()
        assert result["byoc"]["schema"] == {"fields": {}}
        assert "read_capacity" not in result["byoc"]

    def test_asdict_with_all_optional(self) -> None:
        spec = ByocSpec(
            environment="aws-us-east-1-b921",
            read_capacity={"mode": "OnDemand"},
            schema={"fields": {}},
        )
        result = spec.asdict()
        assert result["byoc"]["read_capacity"] == {"mode": "OnDemand"}
        assert result["byoc"]["schema"] == {"fields": {}}

    def test_construct_and_encode(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921")
        encoded = msgspec.json.encode(spec)
        decoded: dict[str, Any] = msgspec.json.decode(encoded)
        assert decoded == {"environment": "aws-us-east-1-b921"}
        assert spec.environment == "aws-us-east-1-b921"

    def test_byoc_spec_with_read_capacity_on_demand(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921", read_capacity={"mode": "OnDemand"})
        assert spec.read_capacity == {"mode": "OnDemand"}

    def test_byoc_spec_with_read_capacity_dedicated(self) -> None:
        spec = ByocSpec(
            environment="aws-us-east-1-b921",
            read_capacity={
                "mode": "Dedicated",
                "dedicated": {
                    "node_type": "t1",
                    "scaling": "Manual",
                    "manual": {"replicas": 2, "shards": 1},
                },
            },
        )
        assert spec.read_capacity is not None
        assert spec.read_capacity["mode"] == "Dedicated"
        assert spec.read_capacity["dedicated"]["node_type"] == "t1"

    def test_byoc_spec_defaults_no_read_capacity(self) -> None:
        spec = ByocSpec(environment="aws-us-east-1-b921")
        assert spec.read_capacity is None


class TestReExports:
    """Verify models are importable from the top-level models package."""

    def test_import_from_models(self) -> None:
        from pinecone.models import (
            ByocSpec,
            ByocSpecInfo,
            IndexList,
            IndexModel,
            IndexSpec,
            IndexStatus,
            PodSpec,
            PodSpecInfo,
            ServerlessSpec,
            ServerlessSpecInfo,
        )

        assert IndexModel is not None
        assert IndexSpec is not None
        assert IndexStatus is not None
        assert IndexList is not None
        assert ServerlessSpec is not None
        assert ServerlessSpecInfo is not None
        assert PodSpec is not None
        assert PodSpecInfo is not None
        assert ByocSpec is not None
        assert ByocSpecInfo is not None
