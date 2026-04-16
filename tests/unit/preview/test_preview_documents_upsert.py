"""Unit tests for PreviewDocuments.upsert and AsyncPreviewDocuments.upsert."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import HTTPClient
from pinecone.errors.exceptions import ValidationError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_documents import AsyncPreviewDocuments
from pinecone.preview.documents import PreviewDocuments
from pinecone.preview.models.documents import PreviewDocumentUpsertResponse

INDEX_HOST = "https://idx-abc.svc.pinecone.io"
UPSERT_URL = f"{INDEX_HOST}/namespaces/my-ns/documents/upsert"

_UPSERT_RESPONSE = {"upserted_count": 2}


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
    from pinecone._internal.http_client import AsyncHTTPClient

    async_http = AsyncHTTPClient(config, "2025-10")
    return AsyncPreviewDocuments(http=async_http, config=config, host=INDEX_HOST)


# ---------------------------------------------------------------------------
# Sync — happy path
# ---------------------------------------------------------------------------


@respx.mock
def test_upsert_sends_expected_body_and_header(
    docs: PreviewDocuments,
) -> None:
    route = respx.post(UPSERT_URL).mock(
        return_value=httpx.Response(200, json=_UPSERT_RESPONSE)
    )

    result = docs.upsert(
        namespace="my-ns",
        documents=[{"_id": "doc-1"}, {"_id": "doc-2"}],
    )

    assert route.called
    request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == UPSERT_URL
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION
    assert isinstance(result, PreviewDocumentUpsertResponse)
    assert result.upserted_count == 2


@respx.mock
def test_upsert_body_wraps_documents_array(docs: PreviewDocuments) -> None:
    import orjson

    route = respx.post(UPSERT_URL).mock(
        return_value=httpx.Response(200, json=_UPSERT_RESPONSE)
    )
    docs.upsert(namespace="my-ns", documents=[{"_id": "x", "title": "hi"}])
    body = orjson.loads(route.calls.last.request.content)
    assert "documents" in body
    assert body["documents"] == [{"_id": "x", "title": "hi"}]


# ---------------------------------------------------------------------------
# Sync — validation: namespace
# ---------------------------------------------------------------------------


def test_upsert_empty_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.upsert(namespace="", documents=[{"_id": "a"}])


def test_upsert_whitespace_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.upsert(namespace="   ", documents=[{"_id": "a"}])


# ---------------------------------------------------------------------------
# Sync — validation: documents list
# ---------------------------------------------------------------------------


def test_upsert_empty_documents_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="documents"):
        docs.upsert(namespace="ns", documents=[])


def test_upsert_101_documents_raises(docs: PreviewDocuments) -> None:
    many = [{"_id": str(i)} for i in range(101)]
    with pytest.raises(ValidationError, match="100"):
        docs.upsert(namespace="ns", documents=many)


def test_upsert_100_documents_accepted(docs: PreviewDocuments) -> None:
    hundred = [{"_id": str(i)} for i in range(100)]
    with respx.mock:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(200, json={"upserted_count": 100})
        )
        result = docs.upsert(namespace="my-ns", documents=hundred)
    assert result.upserted_count == 100


# ---------------------------------------------------------------------------
# Sync — validation: _id rules
# ---------------------------------------------------------------------------


def test_upsert_missing_id_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="_id"):
        docs.upsert(namespace="ns", documents=[{"title": "no id"}])


def test_upsert_non_string_id_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="_id"):
        docs.upsert(namespace="ns", documents=[{"_id": 42}])


def test_upsert_empty_string_id_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="_id"):
        docs.upsert(namespace="ns", documents=[{"_id": ""}])


def test_upsert_duplicate_id_raises_with_offending_id(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="dup-id"):
        docs.upsert(
            namespace="ns",
            documents=[{"_id": "dup-id"}, {"_id": "dup-id"}],
        )


# ---------------------------------------------------------------------------
# Async — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_upsert_sends_expected_body_and_header(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(UPSERT_URL).mock(
        return_value=httpx.Response(200, json=_UPSERT_RESPONSE)
    )

    result = await async_docs.upsert(
        namespace="my-ns",
        documents=[{"_id": "doc-1"}, {"_id": "doc-2"}],
    )

    assert route.called
    request = route.calls.last.request
    assert request.method == "POST"
    assert str(request.url) == UPSERT_URL
    assert request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION
    assert isinstance(result, PreviewDocumentUpsertResponse)
    assert result.upserted_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_upsert_body_wraps_documents_array(
    async_docs: AsyncPreviewDocuments,
) -> None:
    import orjson

    route = respx.post(UPSERT_URL).mock(
        return_value=httpx.Response(200, json=_UPSERT_RESPONSE)
    )
    await async_docs.upsert(namespace="my-ns", documents=[{"_id": "x", "val": 1}])
    body = orjson.loads(route.calls.last.request.content)
    assert body["documents"] == [{"_id": "x", "val": 1}]


# ---------------------------------------------------------------------------
# Async — validation mirrors sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_upsert_empty_namespace_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        await async_docs.upsert(namespace="", documents=[{"_id": "a"}])


@pytest.mark.asyncio
async def test_async_upsert_empty_documents_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="documents"):
        await async_docs.upsert(namespace="ns", documents=[])


@pytest.mark.asyncio
async def test_async_upsert_101_documents_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    many = [{"_id": str(i)} for i in range(101)]
    with pytest.raises(ValidationError, match="100"):
        await async_docs.upsert(namespace="ns", documents=many)


@pytest.mark.asyncio
async def test_async_upsert_missing_id_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="_id"):
        await async_docs.upsert(namespace="ns", documents=[{"title": "no id"}])


@pytest.mark.asyncio
async def test_async_upsert_non_string_id_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="_id"):
        await async_docs.upsert(namespace="ns", documents=[{"_id": 99}])


@pytest.mark.asyncio
async def test_async_upsert_empty_string_id_raises(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="_id"):
        await async_docs.upsert(namespace="ns", documents=[{"_id": ""}])


@pytest.mark.asyncio
async def test_async_upsert_duplicate_id_raises_with_offending_id(
    async_docs: AsyncPreviewDocuments,
) -> None:
    with pytest.raises(ValidationError, match="dup-id"):
        await async_docs.upsert(
            namespace="ns",
            documents=[{"_id": "dup-id"}, {"_id": "dup-id"}],
        )
