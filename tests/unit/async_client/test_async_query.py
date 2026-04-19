"""Unit tests for AsyncIndex.query() sparse-only validation parity."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import ValidationError

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
QUERY_URL = f"{INDEX_HOST_HTTPS}/query"


def _make_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


def _make_query_response() -> dict[str, object]:
    return {"matches": [], "namespace": "", "usage": {"readUnits": 5}}


@pytest.mark.asyncio
@respx.mock
async def test_async_query_sparse_only_accepted() -> None:
    """Sparse-only query (no vector, no id) must be accepted and post sparseVector."""
    route = respx.post(QUERY_URL).mock(
        return_value=httpx.Response(200, json=_make_query_response()),
    )
    idx = _make_index()

    await idx.query(top_k=5, sparse_vector={"indices": [1], "values": [0.5]})

    import orjson

    body = orjson.loads(route.calls.last.request.content)
    assert body["sparseVector"] == {"indices": [1], "values": [0.5]}


@pytest.mark.asyncio
async def test_async_query_vector_id_sparse_none_rejected() -> None:
    """Calling query() with no vector, no id, and no sparse_vector raises ValidationError."""
    idx = _make_index()
    with pytest.raises(
        ValidationError,
        match="At least one of vector, id, or sparse_vector must be provided",
    ):
        await idx.query(top_k=5)
