"""Unit tests for GrpcIndex — alias (BCG-141), query_namespaces, and fetch_by_metadata."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import pinecone.grpc
from pinecone.errors.exceptions import PineconeValueError, ValidationError
from pinecone.grpc import GRPCIndex, GrpcIndex
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.responses import FetchByMetadataResponse, Pagination, QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index_via_alias(mock_channel: MagicMock) -> GRPCIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GRPCIndex(
            host="https://x-abc.svc.pinecone.io",
            api_key="k",
        )


def test_legacy_GRPCIndex_alias_imports() -> None:  # noqa: N802
    from pinecone.grpc import GRPCIndex as _GRPCIndex  # noqa: F401


def test_legacy_GRPCIndex_alias_is_canonical() -> None:  # noqa: N802
    assert pinecone.grpc.GRPCIndex is pinecone.grpc.GrpcIndex


def test_legacy_GRPCIndex_alias_constructs() -> None:  # noqa: N802
    mock_channel = MagicMock()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}
    idx = _make_grpc_index_via_alias(mock_channel)
    assert isinstance(idx, GrpcIndex)


# ---------------------------------------------------------------------------
# GrpcIndex.query_namespaces unit tests
# ---------------------------------------------------------------------------


def test_grpc_query_namespaces_empty_namespaces_raises() -> None:
    """namespaces=[] raises ValidationError before any thread-pool work."""
    mock_channel = MagicMock()
    idx = _make_grpc_index_via_alias(mock_channel)
    with pytest.raises(ValidationError, match="namespaces must be a non-empty list"):
        idx.query_namespaces(vector=[0.1, 0.2], namespaces=[], metric="cosine", top_k=5)


def test_grpc_query_namespaces_missing_vector_raises() -> None:
    """Omitting both vector and sparse_vector raises ValidationError."""
    mock_channel = MagicMock()
    idx = _make_grpc_index_via_alias(mock_channel)
    with pytest.raises(
        ValidationError,
        match="at least one of 'vector' or 'sparse_vector' must be provided",
    ):
        idx.query_namespaces(namespaces=["ns1"], metric="cosine", top_k=5)


def test_grpc_query_namespaces_fans_out_per_namespace() -> None:
    """Duplicate namespaces are deduplicated; self.query is called once per unique namespace."""
    mock_channel = MagicMock()
    idx = _make_grpc_index_via_alias(mock_channel)

    mock_response = QueryResponse(matches=[], namespace="", usage=None)

    with patch.object(idx, "query", return_value=mock_response) as mock_query:
        idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=["ns1", "ns2", "ns2"],
            metric="cosine",
            top_k=5,
        )
        # "ns2" appears twice but must only be queried once after dedup
        assert mock_query.call_count == 2
        called_namespaces = {call.kwargs["namespace"] for call in mock_query.call_args_list}
        assert called_namespaces == {"ns1", "ns2"}


def test_grpc_query_namespaces_aggregates_results() -> None:
    """Results from two namespaces are merged and returned sorted by score (cosine, descending)."""
    mock_channel = MagicMock()
    idx = _make_grpc_index_via_alias(mock_channel)

    ns1_response = QueryResponse(
        matches=[
            ScoredVector(id="v1", score=0.9),
            ScoredVector(id="v2", score=0.7),
        ],
        namespace="ns1",
        usage=Usage(read_units=5),
    )
    ns2_response = QueryResponse(
        matches=[
            ScoredVector(id="v3", score=0.85),
            ScoredVector(id="v4", score=0.6),
        ],
        namespace="ns2",
        usage=Usage(read_units=3),
    )

    response_map: dict[str, QueryResponse] = {"ns1": ns1_response, "ns2": ns2_response}

    def fake_query(**kwargs: Any) -> QueryResponse:
        return response_map[kwargs["namespace"]]

    with patch.object(idx, "query", side_effect=fake_query):
        result = idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=["ns1", "ns2"],
            metric="cosine",
            top_k=3,
        )

    assert isinstance(result, QueryNamespacesResults)
    assert len(result.matches) == 3
    scores = [m.score for m in result.matches]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.85)
    assert scores[2] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# GrpcIndex.fetch_by_metadata unit tests
# ---------------------------------------------------------------------------


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host="https://x-abc.svc.pinecone.io",
            api_key="k",
        )


def test_grpc_fetch_by_metadata_routes_to_channel() -> None:
    """Wrapper parses channel dict into FetchByMetadataResponse and calls channel with expected kwargs."""
    mock_channel = MagicMock()
    mock_channel.fetch_by_metadata.return_value = {
        "vectors": {
            "v1": {"values": [0.1, 0.2, 0.3], "metadata": {"genre": "comedy"}},
            "v2": {"values": [0.4, 0.5, 0.6]},
        },
        "namespace": "movies",
        "usage": {"read_units": 2},
        "pagination": {"next": "tok123"},
    }

    idx = _make_grpc_index(mock_channel)
    response = idx.fetch_by_metadata(
        filter={"genre": {"$eq": "comedy"}},
        namespace="movies",
        limit=10,
        pagination_token="prev-tok",
        timeout=5.0,
    )

    mock_channel.fetch_by_metadata.assert_called_once_with(
        namespace="movies",
        filter={"genre": {"$eq": "comedy"}},
        limit=10,
        pagination_token="prev-tok",
        timeout_s=5.0,
    )

    assert isinstance(response, FetchByMetadataResponse)
    assert set(response.vectors.keys()) == {"v1", "v2"}
    assert response.vectors["v1"].values == [0.1, 0.2, 0.3]
    assert response.vectors["v1"].metadata == {"genre": "comedy"}
    assert response.namespace == "movies"
    assert response.usage is not None
    assert response.usage.read_units == 2
    assert isinstance(response.pagination, Pagination)
    assert response.pagination.next == "tok123"


def test_grpc_fetch_by_metadata_limit_validation() -> None:
    """Passing limit=0 or limit=-1 raises PineconeValueError via require_positive."""
    mock_channel = MagicMock()
    idx = _make_grpc_index(mock_channel)

    with pytest.raises(PineconeValueError):
        idx.fetch_by_metadata(filter={"genre": "comedy"}, limit=0)

    with pytest.raises(PineconeValueError):
        idx.fetch_by_metadata(filter={"genre": "comedy"}, limit=-1)

    mock_channel.fetch_by_metadata.assert_not_called()


def test_grpc_fetch_by_metadata_pagination_token_forwarded() -> None:
    """pagination_token is forwarded verbatim to the channel call."""
    mock_channel = MagicMock()
    mock_channel.fetch_by_metadata.return_value = {
        "vectors": {},
        "namespace": "",
    }

    idx = _make_grpc_index(mock_channel)
    idx.fetch_by_metadata(filter={"x": 1}, pagination_token="page-abc")

    call_kwargs = mock_channel.fetch_by_metadata.call_args.kwargs
    assert call_kwargs["pagination_token"] == "page-abc"


def test_grpc_fetch_by_metadata_empty_namespace_converts_to_none() -> None:
    """Empty-string namespace is passed as None to the channel."""
    mock_channel = MagicMock()
    mock_channel.fetch_by_metadata.return_value = {"vectors": {}, "namespace": ""}

    idx = _make_grpc_index(mock_channel)
    idx.fetch_by_metadata(filter={"x": 1}, namespace="")

    call_kwargs = mock_channel.fetch_by_metadata.call_args.kwargs
    assert call_kwargs["namespace"] is None


def test_grpc_fetch_by_metadata_no_usage_no_pagination() -> None:
    """Missing usage and pagination keys yield None in the response."""
    mock_channel = MagicMock()
    mock_channel.fetch_by_metadata.return_value = {
        "vectors": {"v1": {"values": [1.0, 2.0]}},
        "namespace": "ns",
    }

    idx = _make_grpc_index(mock_channel)
    response = idx.fetch_by_metadata(filter={"k": "v"})

    assert response.usage is None
    assert response.pagination is None
