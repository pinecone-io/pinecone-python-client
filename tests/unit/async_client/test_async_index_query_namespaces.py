"""Unit tests for AsyncIndex.query_namespaces."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"


def _make_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


def _make_query_response(
    matches: list[ScoredVector],
    namespace: str = "",
    read_units: int = 5,
) -> QueryResponse:
    return QueryResponse(
        matches=matches,
        namespace=namespace,
        usage=Usage(read_units=read_units),
    )


def _scored(id: str, score: float) -> ScoredVector:
    return ScoredVector(id=id, score=score)


class TestQueryNamespacesDenseHappyPath:
    @pytest.mark.asyncio
    async def test_query_namespaces_dense(self) -> None:
        """Dense query fans out to all namespaces with vector kwarg."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.9)])

        with patch.object(idx, "query", new_callable=AsyncMock, return_value=response) as mock_query:
            await idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=["ns1", "ns2"],
                metric="cosine",
                top_k=5,
            )
            assert mock_query.call_count == 2
            for call in mock_query.call_args_list:
                assert call.kwargs["vector"] == [0.1, 0.2, 0.3]


class TestQueryNamespacesSparseOnly:
    @pytest.mark.asyncio
    async def test_query_namespaces_sparse_only_omits_vector(self) -> None:
        """Sparse-only query must not pass vector kwarg to self.query."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.9)])

        with patch.object(idx, "query", new_callable=AsyncMock, return_value=response) as mock_query:
            await idx.query_namespaces(
                sparse_vector={"indices": [0, 1], "values": [0.1, 0.2]},
                namespaces=["ns1"],
                metric="dotproduct",
                top_k=3,
            )
            assert mock_query.await_count == 1
            call_kwargs = mock_query.call_args.kwargs
            assert "vector" not in call_kwargs
            assert call_kwargs["sparse_vector"] == {"indices": [0, 1], "values": [0.1, 0.2]}


class TestQueryNamespacesValidation:
    @pytest.mark.asyncio
    async def test_query_namespaces_requires_vector_or_sparse(self) -> None:
        """Calling with neither vector nor sparse_vector raises ValidationError."""
        idx = _make_index()
        with pytest.raises(
            ValidationError,
            match="at least one of 'vector' or 'sparse_vector' must be provided",
        ):
            await idx.query_namespaces(
                namespaces=["ns1"],
                metric="dotproduct",
            )

    @pytest.mark.asyncio
    async def test_query_namespaces_empty_vector_raises(self) -> None:
        """Passing vector=[] (falsy) without sparse_vector raises ValidationError."""
        idx = _make_index()
        with pytest.raises(
            ValidationError,
            match="at least one of 'vector' or 'sparse_vector' must be provided",
        ):
            await idx.query_namespaces(
                vector=[],
                namespaces=["ns1"],
                metric="cosine",
            )

    @pytest.mark.asyncio
    async def test_query_namespaces_empty_namespaces_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="namespaces must be a non-empty list"):
            await idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=[],
                metric="cosine",
            )

    @pytest.mark.asyncio
    async def test_query_namespaces_invalid_metric_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="Invalid metric 'badmetric'"):
            await idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1"],
                metric="badmetric",
            )
