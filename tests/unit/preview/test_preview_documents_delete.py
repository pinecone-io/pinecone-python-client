"""Unit tests for PreviewDocuments.delete and AsyncPreviewDocuments.delete."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient
from pinecone.errors.exceptions import ValidationError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_documents import AsyncPreviewDocuments
from pinecone.preview.documents import PreviewDocuments

INDEX_HOST = "https://idx-abc.svc.pinecone.io"
DELETE_URL = f"{INDEX_HOST}/namespaces/my-ns/documents/delete"


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key")


@pytest.fixture
def http(config: PineconeConfig) -> HTTPClient:
    return HTTPClient(config, "2025-10")


@pytest.fixture
def docs(http: HTTPClient, config: PineconeConfig) -> PreviewDocuments:
    return PreviewDocuments(http=http, config=config, host=INDEX_HOST)


@pytest.fixture
def async_docs(config: PineconeConfig) -> AsyncPreviewDocuments:
    async_http = AsyncHTTPClient(config, "2025-10")
    return AsyncPreviewDocuments(http=async_http, config=config, host=INDEX_HOST)


# ---------------------------------------------------------------------------
# Sync — happy path
# ---------------------------------------------------------------------------


@respx.mock
def test_delete_ids_only(docs: PreviewDocuments) -> None:
    import orjson

    route = respx.post(DELETE_URL).mock(return_value=httpx.Response(202))

    result = docs.delete(namespace="my-ns", ids=["doc-1", "doc-2"])

    assert result is None
    assert route.called
    request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == DELETE_URL
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION
    body = orjson.loads(request.content)
    assert body == {"ids": ["doc-1", "doc-2"]}


@respx.mock
def test_delete_filter_only(docs: PreviewDocuments) -> None:
    import orjson

    route = respx.post(DELETE_URL).mock(return_value=httpx.Response(202))

    result = docs.delete(namespace="my-ns", filter={"genre": "sci-fi"})

    assert result is None
    body = orjson.loads(route.calls.last.request.content)
    assert body == {"filter": {"genre": "sci-fi"}}


@respx.mock
def test_delete_all_only(docs: PreviewDocuments) -> None:
    import orjson

    route = respx.post(DELETE_URL).mock(return_value=httpx.Response(202))

    result = docs.delete(namespace="my-ns", delete_all=True)

    assert result is None
    body = orjson.loads(route.calls.last.request.content)
    assert body == {"delete_all": True}


@respx.mock
def test_delete_all_with_filter_allowed(docs: PreviewDocuments) -> None:
    """delete_all=True + filter is not prohibited by spec."""
    import orjson

    route = respx.post(DELETE_URL).mock(return_value=httpx.Response(202))

    result = docs.delete(namespace="my-ns", delete_all=True, filter={"k": "v"})

    assert result is None
    body = orjson.loads(route.calls.last.request.content)
    assert body == {"delete_all": True, "filter": {"k": "v"}}


@respx.mock
def test_delete_202_empty_body_returns_none(docs: PreviewDocuments) -> None:
    respx.post(DELETE_URL).mock(return_value=httpx.Response(202, content=b""))
    result = docs.delete(namespace="my-ns", delete_all=True)
    assert result is None


# ---------------------------------------------------------------------------
# Sync — validation
# ---------------------------------------------------------------------------


def test_delete_empty_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.delete(namespace="", ids=["a"])


def test_delete_whitespace_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.delete(namespace="   ", ids=["a"])


def test_delete_none_of_three_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        docs.delete(namespace="my-ns")


def test_delete_ids_and_delete_all_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        docs.delete(namespace="my-ns", ids=["a"], delete_all=True)


def test_delete_ids_and_filter_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        docs.delete(namespace="my-ns", ids=["a"], filter={"k": "v"})


# ---------------------------------------------------------------------------
# Async — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_delete_ids_only(async_docs: AsyncPreviewDocuments) -> None:
    import orjson

    route = respx.post(DELETE_URL).mock(return_value=httpx.Response(202))

    result = await async_docs.delete(namespace="my-ns", ids=["doc-1", "doc-2"])

    assert result is None
    assert route.called
    request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == DELETE_URL
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION
    body = orjson.loads(request.content)
    assert body == {"ids": ["doc-1", "doc-2"]}


@pytest.mark.asyncio
@respx.mock
async def test_async_delete_filter_only(async_docs: AsyncPreviewDocuments) -> None:
    import orjson

    route = respx.post(DELETE_URL).mock(return_value=httpx.Response(202))

    result = await async_docs.delete(namespace="my-ns", filter={"genre": "sci-fi"})

    assert result is None
    body = orjson.loads(route.calls.last.request.content)
    assert body == {"filter": {"genre": "sci-fi"}}


@pytest.mark.asyncio
@respx.mock
async def test_async_delete_all_only(async_docs: AsyncPreviewDocuments) -> None:
    import orjson

    route = respx.post(DELETE_URL).mock(return_value=httpx.Response(202))

    result = await async_docs.delete(namespace="my-ns", delete_all=True)

    assert result is None
    body = orjson.loads(route.calls.last.request.content)
    assert body == {"delete_all": True}


@pytest.mark.asyncio
@respx.mock
async def test_async_delete_202_empty_body_returns_none(
    async_docs: AsyncPreviewDocuments,
) -> None:
    respx.post(DELETE_URL).mock(return_value=httpx.Response(202, content=b""))
    result = await async_docs.delete(namespace="my-ns", delete_all=True)
    assert result is None


# ---------------------------------------------------------------------------
# Async — validation mirrors sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_delete_empty_namespace_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        await async_docs.delete(namespace="", ids=["a"])


@pytest.mark.asyncio
async def test_async_delete_none_of_three_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        await async_docs.delete(namespace="my-ns")


@pytest.mark.asyncio
async def test_async_delete_ids_and_delete_all_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        await async_docs.delete(namespace="my-ns", ids=["a"], delete_all=True)


@pytest.mark.asyncio
async def test_async_delete_ids_and_filter_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        await async_docs.delete(namespace="my-ns", ids=["a"], filter={"k": "v"})
