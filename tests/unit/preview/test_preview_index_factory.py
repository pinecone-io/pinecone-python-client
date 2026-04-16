"""Unit tests for Preview.index() and AsyncPreview.index() factory methods."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import PineconeValueError
from pinecone.preview import AsyncPreview, Preview
from pinecone.preview.async_index import AsyncPreviewIndex
from pinecone.preview.index import PreviewDocuments, PreviewIndex

BASE_URL = "https://api.test.pinecone.io"

_INDEX_RESPONSE: dict = {
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
# Async — direct host path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_index_with_host_returns_async_preview_index(
    async_preview: AsyncPreview,
) -> None:
    idx = await async_preview.index(host="my-index-abc.svc.pinecone.io")

    assert isinstance(idx, AsyncPreviewIndex)
    assert idx.host == "my-index-abc.svc.pinecone.io"


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_host_makes_no_http_calls(
    async_preview: AsyncPreview,
) -> None:
    # @respx.mock intercepts all traffic; any unexpected request raises an error.
    await async_preview.index(host="my-index-abc.svc.pinecone.io")


# ---------------------------------------------------------------------------
# Async — name-resolution path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_name_resolves_host(async_preview: AsyncPreview) -> None:
    respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    idx = await async_preview.index(name="my-index")

    assert isinstance(idx, AsyncPreviewIndex)
    assert idx.host == "my-index-abc.svc.pinecone.io"


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_name_caches_host(async_preview: AsyncPreview) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    await async_preview.index(name="my-index")
    await async_preview.index(name="my-index")

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_index_with_name_serves_cached_host(async_preview: AsyncPreview) -> None:
    route = respx.get(f"{BASE_URL}/indexes/my-index").mock(
        return_value=httpx.Response(200, json=_INDEX_RESPONSE)
    )

    idx1 = await async_preview.index(name="my-index")
    idx2 = await async_preview.index(name="my-index")

    assert route.call_count == 1
    assert idx1.host == idx2.host


# ---------------------------------------------------------------------------
# Async — validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_index_with_neither_raises(async_preview: AsyncPreview) -> None:
    with pytest.raises(PineconeValueError, match="Exactly one"):
        await async_preview.index()


@pytest.mark.asyncio
async def test_async_index_with_both_raises(async_preview: AsyncPreview) -> None:
    with pytest.raises(PineconeValueError, match="Exactly one"):
        await async_preview.index(name="my-index", host="my-index-abc.svc.pinecone.io")
