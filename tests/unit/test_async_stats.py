"""Unit tests for AsyncIndex.describe_index_stats() method."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.models.vectors.responses import DescribeIndexStatsResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
STATS_URL = f"{INDEX_HOST_HTTPS}/describe_index_stats"

STATS_RESPONSE: dict[str, Any] = {
    "namespaces": {
        "ns1": {"vectorCount": 100},
        "ns2": {"vectorCount": 200},
    },
    "dimension": 128,
    "indexFullness": 0.5,
    "totalVectorCount": 300,
    "metric": "cosine",
    "vectorType": "dense",
}


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


class TestAsyncDescribeIndexStats:
    """Async describe_index_stats tests."""

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_basic(self) -> None:
        respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=STATS_RESPONSE),
        )
        idx = _make_async_index()
        result = await idx.describe_index_stats()

        assert isinstance(result, DescribeIndexStatsResponse)
        assert result.dimension == 128
        assert result.total_vector_count == 300
        assert result.index_fullness == pytest.approx(0.5)
        assert result.metric == "cosine"
        assert len(result.namespaces) == 2
        assert result.namespaces["ns1"].vector_count == 100
        assert result.namespaces["ns2"].vector_count == 200

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_with_filter(self) -> None:
        route = respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=STATS_RESPONSE),
        )
        idx = _make_async_index()
        await idx.describe_index_stats(filter={"genre": {"$eq": "sci-fi"}})

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["filter"] == {"genre": {"$eq": "sci-fi"}}

    @respx.mock
    @pytest.mark.anyio
    async def test_async_describe_index_stats_no_filter(self) -> None:
        route = respx.post(STATS_URL).mock(
            return_value=httpx.Response(200, json=STATS_RESPONSE),
        )
        idx = _make_async_index()
        await idx.describe_index_stats()

        request = route.calls.last.request
        body = json.loads(request.content)
        assert body == {}

    def test_async_describe_index_stats_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(TypeError):
            idx.describe_index_stats({"genre": {"$eq": "sci-fi"}})  # type: ignore[misc]
