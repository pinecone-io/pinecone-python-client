"""Unit tests for Preview.index() and AsyncPreview.index() factory methods."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview import AsyncPreview, Preview
from pinecone.preview.async_index import AsyncPreviewDocuments, AsyncPreviewIndex
from pinecone.preview.index import PreviewDocuments, PreviewIndex

BASE_URL = "https://api.test.pinecone.io"

_INDEX_RESPONSE: dict[str, object] = {
    "name": "my-index",
    "host": "my-index-abc.svc.pinecone.io",
    "status": {"ready": True, "state": "Ready"},
    "schema": {"fields": {"e": {"type": "dense_vector", "dimension": 4}}},
    "deployment": {
        "deployment_type": "managed",
        "environment": "us-east-1-aws",
        "cloud": "aws",
        "region": "us-east-1",
    },
    "deletion_protection": "disabled",
}


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key", host=BASE_URL)


@pytest.fixture
def preview(config: PineconeConfig) -> Preview:
    http = MagicMock()
    return Preview(http=http, config=config)


@pytest.fixture
def async_preview(config: PineconeConfig) -> AsyncPreview:
    http = MagicMock()
    return AsyncPreview(http=http, config=config)


# ---------------------------------------------------------------------------
# Sync — direct host path
# ---------------------------------------------------------------------------


def test_index_with_host_returns_preview_index(preview: Preview) -> None:
    idx = preview.index(host="my-index-abc.svc.pinecone.io")

    assert isinstance(idx, PreviewIndex)
    assert idx.host == "my-index-abc.svc.pinecone.io"


@respx.mock
def test_index_with_host_makes_no_http_calls(preview: Preview) -> None:
    # @respx.mock intercepts all traffic; any unexpected request raises an error.
    # If this passes without registering any routes, no HTTP call was made.
    preview.index(host="my-index-abc.svc.pinecone.io")


def test_index_with_host_has_documents_attribute(preview: Preview) -> None:
    idx = preview.index(host="my-index-abc.svc.pinecone.io")
    assert isinstance(idx.documents, PreviewDocuments)


# ---------------------------------------------------------------------------
# Sync — name-resolution path
# ---------------------------------------------------------------------------


@respx.mock
def test_index_with_name_resolves_host(preview: Preview) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    idx = preview.index(name="my-index")

    assert isinstance(idx, PreviewIndex)
    assert idx.host == "my-index-abc.svc.pinecone.io"


@respx.mock
def test_index_with_name_caches_host(preview: Preview) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    preview.index(name="my-index")
    preview.index(name="my-index")

    assert route.call_count == 1


@respx.mock
def test_index_with_two_different_names_triggers_two_describes(preview: Preview) -> None:
    route_a = respx.get(f"{BASE_URL}/indexes/index-a").mock(
        return_value=httpx.Response(
            200, json={**_INDEX_RESPONSE, "name": "index-a", "host": "a.svc.pinecone.io"}
        )
    )
    route_b = respx.get(f"{BASE_URL}/indexes/index-b").mock(
        return_value=httpx.Response(
            200, json={**_INDEX_RESPONSE, "name": "index-b", "host": "b.svc.pinecone.io"}
        )
    )

    idx_a = preview.index(name="index-a")
    idx_b = preview.index(name="index-b")

    assert route_a.call_count == 1
    assert route_b.call_count == 1
    assert idx_a.host == "a.svc.pinecone.io"
    assert idx_b.host == "b.svc.pinecone.io"

    idx_a_again = preview.index(name="index-a")
    assert route_a.call_count == 1
    assert idx_a_again.host == "a.svc.pinecone.io"


@respx.mock
def test_index_with_name_calls_describe_once_then_serves_cache(preview: Preview) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    idx1 = preview.index(name="my-index")
    idx2 = preview.index(name="my-index")

    assert route.call_count == 1
    assert idx1.host == idx2.host


# ---------------------------------------------------------------------------
# Sync — validation errors
# ---------------------------------------------------------------------------


def test_index_with_neither_raises(preview: Preview) -> None:
    with pytest.raises(PineconeValueError, match="Exactly one"):
        preview.index()


def test_index_with_both_raises(preview: Preview) -> None:
    with pytest.raises(PineconeValueError, match="Exactly one"):
        preview.index(name="my-index", host="my-index-abc.svc.pinecone.io")


# ---------------------------------------------------------------------------
# Async — factory is not a coroutine
# ---------------------------------------------------------------------------


def test_async_index_is_not_coroutine(async_preview: AsyncPreview) -> None:
    assert not asyncio.iscoroutinefunction(AsyncPreview.index)
    idx = async_preview.index(host="my-index-abc.svc.pinecone.io")
    assert isinstance(idx, AsyncPreviewIndex)


# ---------------------------------------------------------------------------
# Async — direct host path
# ---------------------------------------------------------------------------


def test_async_index_with_host_returns_async_preview_index(
    async_preview: AsyncPreview,
) -> None:
    idx = async_preview.index(host="my-index-abc.svc.pinecone.io")

    assert isinstance(idx, AsyncPreviewIndex)
    assert idx.host == "my-index-abc.svc.pinecone.io"


@respx.mock
def test_async_index_with_host_makes_no_http_calls(
    async_preview: AsyncPreview,
) -> None:
    # @respx.mock intercepts all traffic; any unexpected request raises an error.
    async_preview.index(host="my-index-abc.svc.pinecone.io")


@pytest.mark.asyncio
async def test_async_index_with_host_has_documents_attribute(
    async_preview: AsyncPreview,
) -> None:
    idx = async_preview.index(host="my-index-abc.svc.pinecone.io")
    assert isinstance(idx.documents, AsyncPreviewDocuments)


# ---------------------------------------------------------------------------
# Async — name-resolution path (deferred to first data-plane call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_name_resolves_host(async_preview: AsyncPreview) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    # Factory call makes zero HTTP requests.
    idx = async_preview.index(name="my-index")
    assert isinstance(idx, AsyncPreviewIndex)
    assert route.call_count == 0

    # Host is resolved on first _resolve_host() call.
    host = await idx._resolve_host()
    assert host == "my-index-abc.svc.pinecone.io"
    assert idx.host == "my-index-abc.svc.pinecone.io"
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_name_caches_host(async_preview: AsyncPreview) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    idx1 = async_preview.index(name="my-index")
    idx2 = async_preview.index(name="my-index")

    # The AsyncPreview._host_cache is shared via the closure, so the second
    # _resolve_host() call skips the describe round-trip.
    await idx1._resolve_host()
    await idx2._resolve_host()

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_two_different_names_triggers_two_describes(
    async_preview: AsyncPreview,
) -> None:
    route_a = respx.get(f"{BASE_URL}/indexes/index-a").mock(
        return_value=httpx.Response(
            200, json={**_INDEX_RESPONSE, "name": "index-a", "host": "a.svc.pinecone.io"}
        )
    )
    route_b = respx.get(f"{BASE_URL}/indexes/index-b").mock(
        return_value=httpx.Response(
            200, json={**_INDEX_RESPONSE, "name": "index-b", "host": "b.svc.pinecone.io"}
        )
    )

    idx_a = async_preview.index(name="index-a")
    idx_b = async_preview.index(name="index-b")

    assert route_a.call_count == 0
    assert route_b.call_count == 0

    await idx_a._resolve_host()
    await idx_b._resolve_host()

    assert route_a.call_count == 1
    assert route_b.call_count == 1
    assert idx_a.host == "a.svc.pinecone.io"
    assert idx_b.host == "b.svc.pinecone.io"

    await idx_a._resolve_host()
    assert route_a.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_name_serves_cached_host(async_preview: AsyncPreview) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    idx1 = async_preview.index(name="my-index")
    idx2 = async_preview.index(name="my-index")

    await idx1._resolve_host()
    await idx2._resolve_host()

    assert route.call_count == 1
    assert idx1.host == idx2.host


# ---------------------------------------------------------------------------
# Async — validation errors (synchronous, no await)
# ---------------------------------------------------------------------------


def test_async_index_with_neither_raises(async_preview: AsyncPreview) -> None:
    with pytest.raises(PineconeValueError, match="Exactly one"):
        async_preview.index()


def test_async_index_with_both_raises(async_preview: AsyncPreview) -> None:
    with pytest.raises(PineconeValueError, match="Exactly one"):
        async_preview.index(name="my-index", host="my-index-abc.svc.pinecone.io")
