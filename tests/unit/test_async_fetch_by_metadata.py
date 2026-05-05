"""Unit tests for AsyncIndex.fetch_by_metadata() method."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.models.vectors.responses import FetchByMetadataResponse

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
FETCH_BY_META_URL = f"{INDEX_HOST_HTTPS}/vectors/fetch_by_metadata"


def _make_response(
    *,
    vectors: dict[str, dict[str, Any]] | None = None,
    namespace: str = "",
    usage: dict[str, int] | None = None,
    pagination: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a realistic fetch-by-metadata API response payload."""
    resp: dict[str, object] = {
        "vectors": vectors or {},
        "namespace": namespace,
        "usage": usage or {"readUnits": 5},
    }
    if pagination is not None:
        resp["pagination"] = pagination
    return resp


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Basic success
# ---------------------------------------------------------------------------


class TestAsyncFetchByMetadataBasic:
    """fetch_by_metadata returns FetchByMetadataResponse with vectors."""

    @respx.mock
    @pytest.mark.anyio
    async def test_fetch_by_metadata_basic(self) -> None:
        respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(
                200,
                json=_make_response(
                    vectors={
                        "vec1": {"id": "vec1", "values": [0.1, 0.2]},
                        "vec2": {"id": "vec2", "values": [0.3, 0.4]},
                    },
                ),
            ),
        )
        idx = _make_async_index()
        result = await idx.fetch_by_metadata(filter={"genre": "comedy"})

        assert isinstance(result, FetchByMetadataResponse)
        assert len(result.vectors) == 2
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec2"].id == "vec2"


# ---------------------------------------------------------------------------
# Request body construction
# ---------------------------------------------------------------------------


class TestAsyncFetchByMetadataRequestBody:
    """Verify the POST body is built correctly from parameters."""

    @respx.mock
    @pytest.mark.anyio
    async def test_fetch_by_metadata_sends_filter(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_async_index()
        await idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["filter"] == {"genre": {"$eq": "comedy"}}

    @respx.mock
    @pytest.mark.anyio
    async def test_fetch_by_metadata_sends_namespace(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_async_index()
        await idx.fetch_by_metadata(filter={"a": 1}, namespace="my-ns")

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["namespace"] == "my-ns"

    @respx.mock
    @pytest.mark.anyio
    async def test_fetch_by_metadata_sends_limit(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_async_index()
        await idx.fetch_by_metadata(filter={"a": 1}, limit=50)

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["limit"] == 50

    @respx.mock
    @pytest.mark.anyio
    async def test_fetch_by_metadata_sends_pagination_token(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_async_index()
        await idx.fetch_by_metadata(filter={"a": 1}, pagination_token="abc")

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["paginationToken"] == "abc"

    @respx.mock
    @pytest.mark.anyio
    async def test_fetch_by_metadata_omits_optional_fields(self) -> None:
        route = respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_async_index()
        await idx.fetch_by_metadata(filter={"a": 1})

        request = route.calls.last.request
        body = json.loads(request.content)
        assert "namespace" not in body
        assert "limit" not in body
        assert "paginationToken" not in body


# ---------------------------------------------------------------------------
# Limit validation
# ---------------------------------------------------------------------------


class TestAsyncFetchByMetadataLimitValidation:
    """limit must be >= 1; limit=0 or negative raises before any HTTP call."""

    @pytest.mark.anyio
    async def test_fetch_by_metadata_limit_validation_zero(self) -> None:
        idx = _make_async_index()
        with pytest.raises(Exception, match="limit"):
            await idx.fetch_by_metadata(filter={"a": "b"}, limit=0)
        await idx.close()

    @pytest.mark.anyio
    async def test_fetch_by_metadata_limit_validation_negative(self) -> None:
        idx = _make_async_index()
        with pytest.raises(Exception, match="limit"):
            await idx.fetch_by_metadata(filter={"a": "b"}, limit=-1)
        await idx.close()

    @respx.mock
    @pytest.mark.anyio
    async def test_fetch_by_metadata_limit_validation_one_passes(self) -> None:
        respx.post(FETCH_BY_META_URL).mock(
            return_value=httpx.Response(200, json=_make_response()),
        )
        idx = _make_async_index()
        try:
            await idx.fetch_by_metadata(filter={"a": "b"}, limit=1)
        finally:
            await idx.close()
