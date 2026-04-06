"""Unit tests for vector and response models."""

from __future__ import annotations

from typing import Any

import msgspec

from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    NamespaceSummary,
    QueryResponse,
    ResponseInfo,
    UpsertResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector


class TestSparseValues:
    def test_construct(self) -> None:
        sv = SparseValues(indices=[0, 3, 7], values=[0.1, 0.2, 0.3])
        assert sv.indices == [0, 3, 7]
        assert sv.values == [0.1, 0.2, 0.3]

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {"indices": [1, 2], "values": [0.5, 0.6]}
        sv = msgspec.convert(data, SparseValues)
        assert sv.indices == [1, 2]
        assert sv.values == [0.5, 0.6]


class TestVector:
    def test_minimal(self) -> None:
        v = Vector(id="vec-1")
        assert v.id == "vec-1"
        assert v.values == []
        assert v.sparse_values is None
        assert v.metadata is None

    def test_with_values(self) -> None:
        v = Vector(id="vec-2", values=[1.0, 2.0, 3.0])
        assert v.values == [1.0, 2.0, 3.0]

    def test_with_sparse_values(self) -> None:
        sv = SparseValues(indices=[0], values=[1.0])
        v = Vector(id="vec-3", sparse_values=sv)
        assert v.sparse_values is not None
        assert v.sparse_values.indices == [0]

    def test_with_metadata(self) -> None:
        v = Vector(id="vec-4", metadata={"genre": "comedy", "year": 2024})
        assert v.metadata == {"genre": "comedy", "year": 2024}

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {
            "id": "vec-5",
            "values": [0.1, 0.2],
            "sparse_values": {"indices": [0], "values": [1.0]},
            "metadata": {"key": "val"},
        }
        v = msgspec.convert(data, Vector)
        assert v.id == "vec-5"
        assert v.values == [0.1, 0.2]
        assert v.sparse_values is not None
        assert v.metadata == {"key": "val"}


class TestScoredVector:
    def test_minimal(self) -> None:
        sv = ScoredVector(id="match-1", score=0.95)
        assert sv.id == "match-1"
        assert sv.score == 0.95
        assert sv.values == []
        assert sv.sparse_values is None
        assert sv.metadata is None

    def test_with_all_fields(self) -> None:
        sparse = SparseValues(indices=[1], values=[0.5])
        sv = ScoredVector(
            id="match-2",
            score=0.8,
            values=[1.0, 2.0],
            sparse_values=sparse,
            metadata={"tag": "test"},
        )
        assert sv.score == 0.8
        assert sv.values == [1.0, 2.0]
        assert sv.sparse_values is not None
        assert sv.metadata == {"tag": "test"}


class TestUsage:
    def test_defaults(self) -> None:
        u = Usage()
        assert u.read_units is None
        assert u.write_units is None

    def test_with_values(self) -> None:
        u = Usage(read_units=5, write_units=10)
        assert u.read_units == 5
        assert u.write_units == 10


class TestUpsertResponse:
    def test_construct(self) -> None:
        r = UpsertResponse(upserted_count=42)
        assert r.upserted_count == 42

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {"upserted_count": 100}
        r = msgspec.convert(data, UpsertResponse)
        assert r.upserted_count == 100


class TestQueryResponse:
    def test_defaults(self) -> None:
        r = QueryResponse()
        assert r.matches == []
        assert r.namespace == ""
        assert r.usage is None

    def test_namespace_defaults_to_empty_string(self) -> None:
        """Per unified-rs-0013: null namespace becomes empty string."""
        data: dict[str, Any] = {"matches": [], "usage": {"read_units": 1}}
        r = msgspec.convert(data, QueryResponse)
        assert r.namespace == ""

    def test_from_dict_with_matches(self) -> None:
        data: dict[str, Any] = {
            "matches": [
                {"id": "v1", "score": 0.99},
                {"id": "v2", "score": 0.85, "values": [1.0, 2.0]},
            ],
            "namespace": "test-ns",
            "usage": {"read_units": 5},
        }
        r = msgspec.convert(data, QueryResponse)
        assert len(r.matches) == 2
        assert r.matches[0].id == "v1"
        assert r.matches[0].score == 0.99
        assert r.matches[1].values == [1.0, 2.0]
        assert r.namespace == "test-ns"
        assert r.usage is not None
        assert r.usage.read_units == 5


class TestFetchResponse:
    def test_defaults(self) -> None:
        r = FetchResponse()
        assert r.vectors == {}
        assert r.namespace == ""
        assert r.usage is None

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {
            "vectors": {
                "vec-1": {"id": "vec-1", "values": [0.1, 0.2]},
                "vec-2": {"id": "vec-2", "values": [0.3, 0.4]},
            },
            "namespace": "my-ns",
            "usage": {"read_units": 2},
        }
        r = msgspec.convert(data, FetchResponse)
        assert len(r.vectors) == 2
        assert r.vectors["vec-1"].id == "vec-1"
        assert r.vectors["vec-1"].values == [0.1, 0.2]
        assert r.namespace == "my-ns"
        assert r.usage is not None
        assert r.usage.read_units == 2


class TestDescribeIndexStatsResponse:
    def test_defaults(self) -> None:
        r = DescribeIndexStatsResponse()
        assert r.namespaces == {}
        assert r.dimension is None
        assert r.index_fullness == 0.0
        assert r.total_vector_count == 0

    def test_from_dict(self) -> None:
        data: dict[str, Any] = {
            "namespaces": {
                "ns1": {"vector_count": 100},
                "ns2": {"vector_count": 200},
            },
            "dimension": 128,
            "index_fullness": 0.5,
            "total_vector_count": 300,
        }
        r = msgspec.convert(data, DescribeIndexStatsResponse)
        assert len(r.namespaces) == 2
        assert r.namespaces["ns1"].vector_count == 100
        assert r.namespaces["ns2"].vector_count == 200
        assert r.dimension == 128
        assert r.index_fullness == 0.5
        assert r.total_vector_count == 300


class TestResponseInfo:
    def test_default(self) -> None:
        r = ResponseInfo()
        assert r.request_id is None

    def test_with_request_id(self) -> None:
        r = ResponseInfo(request_id="abc-123")
        assert r.request_id == "abc-123"


class TestNamespaceSummary:
    def test_default(self) -> None:
        ns = NamespaceSummary()
        assert ns.vector_count == 0

    def test_with_count(self) -> None:
        ns = NamespaceSummary(vector_count=500)
        assert ns.vector_count == 500
