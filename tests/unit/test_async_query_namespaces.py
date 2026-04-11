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


class TestAsyncQueryNamespacesFanOut:
    @pytest.mark.asyncio
    async def test_query_namespaces_fans_out(self) -> None:
        """Mock AsyncIndex.query to return a QueryResponse with 2 matches, call
        query_namespaces with 3 namespaces, verify query was awaited 3 times."""
        idx = _make_index()
        response = _make_query_response(
            [_scored("v1", 0.9), _scored("v2", 0.8)],
        )

        mock_query = AsyncMock(return_value=response)
        with patch.object(idx, "query", mock_query):
            await idx.query_namespaces(
                vector=[0.1, 0.2, 0.3],
                namespaces=["ns1", "ns2", "ns3"],
                metric="cosine",
                top_k=10,
            )
            assert mock_query.await_count == 3
            called_namespaces = sorted(c.kwargs["namespace"] for c in mock_query.call_args_list)
            assert called_namespaces == ["ns1", "ns2", "ns3"]


class TestAsyncQueryNamespacesMerge:
    @pytest.mark.asyncio
    async def test_query_namespaces_merges_results(self) -> None:
        """Mock AsyncIndex.query to return different scores per namespace, verify
        final results are sorted by score (cosine: descending)."""
        idx = _make_index()

        async def query_side_effect(**kwargs: object) -> QueryResponse:
            ns = kwargs["namespace"]
            if ns == "ns1":
                return _make_query_response([_scored("a", 0.9), _scored("b", 0.7)])
            else:
                return _make_query_response([_scored("c", 0.95), _scored("d", 0.6)])

        with patch.object(idx, "query", new_callable=AsyncMock, side_effect=query_side_effect):
            result = await idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1", "ns2"],
                metric="cosine",
                top_k=3,
            )
            ids = [m.id for m in result.matches]
            assert ids == ["c", "a", "b"]


class TestAsyncQueryNamespacesValidation:
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
    async def test_query_namespaces_empty_vector_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="vector must be a non-empty list"):
            await idx.query_namespaces(
                vector=[],
                namespaces=["ns1"],
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

    @pytest.mark.asyncio
    async def test_query_namespaces_invalid_metric_is_validation_error(self) -> None:
        from pinecone.errors.exceptions import PineconeError

        idx = _make_index()
        with pytest.raises(PineconeError):
            await idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1"],
                metric="invalid",
            )


class TestAsyncQueryNamespacesDedup:
    @pytest.mark.asyncio
    async def test_query_namespaces_deduplicates_namespaces(self) -> None:
        """Call with namespaces=["a", "b", "a"], verify query is awaited only twice."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        mock_query = AsyncMock(return_value=response)
        with patch.object(idx, "query", mock_query):
            await idx.query_namespaces(
                vector=[0.1],
                namespaces=["a", "b", "a"],
                metric="cosine",
            )
            assert mock_query.await_count == 2
            called_namespaces = sorted(c.kwargs["namespace"] for c in mock_query.call_args_list)
            assert called_namespaces == ["a", "b"]


class TestAsyncQueryNamespacesDefaultTopK:
    @pytest.mark.asyncio
    async def test_query_namespaces_default_top_k(self) -> None:
        """Don't pass top_k, verify aggregator uses 10 (mock query to return
        15 matches each, verify result has 10)."""
        idx = _make_index()
        many_matches = [_scored(f"v{i}", 0.9 - i * 0.01) for i in range(15)]
        response = _make_query_response(many_matches)

        with patch.object(idx, "query", new_callable=AsyncMock, return_value=response):
            result = await idx.query_namespaces(
                vector=[0.1, 0.2],
                namespaces=["ns1", "ns2"],
                metric="cosine",
            )
            assert len(result.matches) == 10


class TestAsyncQueryNamespacesDRNParams:
    @pytest.mark.asyncio
    async def test_query_namespaces_forwards_scan_factor(self) -> None:
        """Pass scan_factor, verify each query call receives it."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        mock_query = AsyncMock(return_value=response)
        with patch.object(idx, "query", mock_query):
            await idx.query_namespaces(
                vector=[0.1],
                namespaces=["ns1"],
                metric="cosine",
                scan_factor=2.0,
            )
            assert mock_query.await_count == 1
            assert mock_query.call_args_list[0].kwargs["scan_factor"] == 2.0

    @pytest.mark.asyncio
    async def test_query_namespaces_forwards_max_candidates(self) -> None:
        """Pass max_candidates, verify each query call receives it."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        mock_query = AsyncMock(return_value=response)
        with patch.object(idx, "query", mock_query):
            await idx.query_namespaces(
                vector=[0.1],
                namespaces=["ns1"],
                metric="cosine",
                max_candidates=5000,
            )
            assert mock_query.await_count == 1
            assert mock_query.call_args_list[0].kwargs["max_candidates"] == 5000

    @pytest.mark.asyncio
    async def test_query_namespaces_drn_params_default_none(self) -> None:
        """Call without DRN params, verify they default to None."""
        idx = _make_index()
        response = _make_query_response([_scored("v1", 0.5)])

        mock_query = AsyncMock(return_value=response)
        with patch.object(idx, "query", mock_query):
            await idx.query_namespaces(
                vector=[0.1],
                namespaces=["ns1"],
                metric="cosine",
            )
            assert mock_query.call_args_list[0].kwargs["scan_factor"] is None
            assert mock_query.call_args_list[0].kwargs["max_candidates"] is None
