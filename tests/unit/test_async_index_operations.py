"""Unit tests for AsyncIndex.query() and AsyncIndex.fetch() methods."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from pinecone import AsyncIndex
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import FetchResponse, QueryResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
QUERY_URL = f"{INDEX_HOST_HTTPS}/query"
FETCH_URL = f"{INDEX_HOST_HTTPS}/vectors/fetch"


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


def _make_query_response(
    *,
    matches: list[dict[str, object]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a realistic query API response payload."""
    return {
        "matches": matches or [],
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }


def _make_fetch_response(
    *,
    vectors: dict[str, dict[str, Any]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a realistic fetch API response payload."""
    return {
        "vectors": vectors or {},
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }


# ---------------------------------------------------------------------------
# AsyncIndex.query()
# ---------------------------------------------------------------------------


class TestAsyncQuery:
    """Async query operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_query_with_vector(self) -> None:
        respx.post(QUERY_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_query_response(
                    matches=[
                        {"id": "vec1", "score": 0.95},
                        {"id": "vec2", "score": 0.80},
                    ],
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.query(top_k=2, vector=[0.1, 0.2, 0.3])

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        assert result.matches[0].id == "vec1"
        assert result.matches[0].score == pytest.approx(0.95)
        assert result.matches[1].id == "vec2"

    @respx.mock
    @pytest.mark.asyncio
    async def test_query_with_id(self) -> None:
        route = respx.post(QUERY_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_query_response(
                    matches=[{"id": "vec1", "score": 1.0}],
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.query(top_k=1, id="vec1")

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 1

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["id"] == "vec1"
        assert "vector" not in body

    @pytest.mark.asyncio
    async def test_query_top_k_validation(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            await idx.query(top_k=0, vector=[0.1])

    @pytest.mark.asyncio
    async def test_query_both_vector_and_id_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="not both"):
            await idx.query(top_k=1, vector=[0.1], id="vec1")

    @pytest.mark.asyncio
    async def test_query_neither_vector_nor_id_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="got neither"):
            await idx.query(top_k=1)

    @pytest.mark.asyncio
    async def test_query_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(TypeError):
            await idx.query(10, [0.1])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AsyncIndex.fetch()
# ---------------------------------------------------------------------------


class TestAsyncFetch:
    """Async fetch operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_multiple_ids(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_fetch_response(
                    vectors={
                        "vec1": {"id": "vec1", "values": [0.1, 0.2, 0.3]},
                        "vec2": {"id": "vec2", "values": [0.4, 0.5, 0.6]},
                    },
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.fetch(ids=["vec1", "vec2"])

        assert isinstance(result, FetchResponse)
        assert len(result.vectors) == 2
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec1"].values == pytest.approx([0.1, 0.2, 0.3])
        assert result.vectors["vec2"].id == "vec2"

    @pytest.mark.asyncio
    async def test_fetch_empty_ids_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="ids must be a non-empty list"):
            await idx.fetch(ids=[])

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_nonexistent_ids_returns_empty(self) -> None:
        respx.get(FETCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_fetch_response(vectors={}),
            ),
        )
        idx = _make_async_index()
        result = await idx.fetch(ids=["does-not-exist"])

        assert isinstance(result, FetchResponse)
        assert result.vectors == {}
