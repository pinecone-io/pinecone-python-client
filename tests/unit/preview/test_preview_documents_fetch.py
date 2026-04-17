"""Unit tests for PreviewDocuments.fetch and AsyncPreviewDocuments.fetch."""

from __future__ import annotations

import httpx
import orjson
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import ValidationError
from pinecone.preview._internal.constants import INDEXES_API_VERSION
from pinecone.preview.async_documents import AsyncPreviewDocuments
from pinecone.preview.documents import PreviewDocuments
from pinecone.preview.models.documents import PreviewDocument, PreviewDocumentFetchResponse

INDEX_HOST = "https://idx-abc.svc.pinecone.io"
FETCH_URL = f"{INDEX_HOST}/namespaces/my-ns/documents/fetch"

_FETCH_RESPONSE = {
    "documents": {
        "doc-1": {"_id": "doc-1", "title": "hello"},
        "doc-2": {"_id": "doc-2", "title": "world"},
    },
    "namespace": "my-ns",
    "usage": {"read_units": 3},
}

_PARTIAL_FETCH_RESPONSE = {
    "documents": {
        "doc-1": {"_id": "doc-1", "title": "hello"},
    },
    "namespace": "my-ns",
    "usage": {"read_units": 2},
}

_FILTER_FETCH_RESPONSE = {
    "documents": {
        "doc-3": {"_id": "doc-3", "category": "news"},
    },
    "namespace": "my-ns",
    "usage": {"read_units": 4},
}


@pytest.fixture
def config() -> PineconeConfig:
    return PineconeConfig(api_key="test-key")


@pytest.fixture
def docs(config: PineconeConfig) -> PreviewDocuments:
    return PreviewDocuments(config=config, host=INDEX_HOST)


@pytest.fixture
def async_docs(config: PineconeConfig) -> AsyncPreviewDocuments:
    return AsyncPreviewDocuments(config=config, host=INDEX_HOST)


# ---------------------------------------------------------------------------
# Sync — (a) by ids happy path
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_by_ids_happy_path(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    result = docs.fetch(namespace="my-ns", ids=["doc-1", "doc-2"])

    body = orjson.loads(route.calls.last.request.content)
    assert body["ids"] == ["doc-1", "doc-2"]
    assert isinstance(result, PreviewDocumentFetchResponse)
    assert result.namespace == "my-ns"
    assert result.usage is not None
    assert result.usage.read_units == 3
    assert len(result.documents) == 2
    assert "doc-1" in result.documents
    assert "doc-2" in result.documents
    assert isinstance(result.documents["doc-1"], PreviewDocument)
    assert result.documents["doc-1"]._id == "doc-1"
    assert result.documents["doc-1"].title == "hello"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Sync — (b) by filter happy path
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_by_filter_happy_path(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(
        return_value=httpx.Response(200, json=_FILTER_FETCH_RESPONSE)
    )

    result = docs.fetch(namespace="my-ns", filter={"category": {"$eq": "news"}})

    body = orjson.loads(route.calls.last.request.content)
    assert body["filter"] == {"category": {"$eq": "news"}}
    assert "ids" not in body
    assert isinstance(result, PreviewDocumentFetchResponse)
    assert "doc-3" in result.documents


# ---------------------------------------------------------------------------
# Sync — (c) empty namespace raises
# ---------------------------------------------------------------------------


def test_fetch_empty_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.fetch(namespace="")


# ---------------------------------------------------------------------------
# Sync — (d) requested ids not in namespace are absent, no error raised
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_missing_ids_silently_omitted(docs: PreviewDocuments) -> None:
    respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_PARTIAL_FETCH_RESPONSE))

    result = docs.fetch(namespace="my-ns", ids=["doc-1", "does-not-exist"])

    assert isinstance(result, PreviewDocumentFetchResponse)
    assert "doc-1" in result.documents
    assert "does-not-exist" not in result.documents


# ---------------------------------------------------------------------------
# Sync — (e) include_fields semantics
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_include_fields_none_omits_field_from_request(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    docs.fetch(namespace="my-ns", ids=["doc-1"], include_fields=None)

    body = orjson.loads(route.calls.last.request.content)
    assert "include_fields" not in body


@respx.mock
def test_fetch_include_fields_wildcard_passes_through(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    docs.fetch(namespace="my-ns", ids=["doc-1"], include_fields=["*"])

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["*"]


@respx.mock
def test_fetch_include_fields_named_fields_passes_through(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    docs.fetch(namespace="my-ns", ids=["doc-1"], include_fields=["title", "body"])

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["title", "body"]


@respx.mock
def test_fetch_empty_body_accepted_by_server(docs: PreviewDocuments) -> None:
    """Spec §5: server accepts empty body — client must not reject it."""
    route = respx.post(FETCH_URL).mock(
        return_value=httpx.Response(200, json={"documents": {}, "namespace": "my-ns"})
    )

    result = docs.fetch(namespace="my-ns")

    body = orjson.loads(route.calls.last.request.content)
    assert body == {}
    assert isinstance(result, PreviewDocumentFetchResponse)
    assert result.documents == {}


@respx.mock
def test_fetch_sends_correct_api_version_header(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    docs.fetch(namespace="my-ns", ids=["doc-1"])

    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


# ---------------------------------------------------------------------------
# Sync — filter omitted when not provided
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_filter_omitted_when_not_provided(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    docs.fetch(namespace="my-ns", ids=["doc-1"])

    body = orjson.loads(route.calls.last.request.content)
    assert "filter" not in body


# ---------------------------------------------------------------------------
# Sync — ids omitted when not provided
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_ids_omitted_when_not_provided(docs: PreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(
        return_value=httpx.Response(200, json=_FILTER_FETCH_RESPONSE)
    )

    docs.fetch(namespace="my-ns", filter={"category": "news"})

    body = orjson.loads(route.calls.last.request.content)
    assert "ids" not in body


# ---------------------------------------------------------------------------
# Async — (f) mirrors sync tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_by_ids_happy_path(async_docs: AsyncPreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    result = await async_docs.fetch(namespace="my-ns", ids=["doc-1", "doc-2"])

    body = orjson.loads(route.calls.last.request.content)
    assert body["ids"] == ["doc-1", "doc-2"]
    assert isinstance(result, PreviewDocumentFetchResponse)
    assert result.namespace == "my-ns"
    assert len(result.documents) == 2
    assert isinstance(result.documents["doc-1"], PreviewDocument)
    assert result.documents["doc-1"]._id == "doc-1"


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_by_filter_happy_path(async_docs: AsyncPreviewDocuments) -> None:
    route = respx.post(FETCH_URL).mock(
        return_value=httpx.Response(200, json=_FILTER_FETCH_RESPONSE)
    )

    result = await async_docs.fetch(namespace="my-ns", filter={"category": {"$eq": "news"}})

    body = orjson.loads(route.calls.last.request.content)
    assert body["filter"] == {"category": {"$eq": "news"}}
    assert isinstance(result, PreviewDocumentFetchResponse)
    assert "doc-3" in result.documents


@pytest.mark.asyncio
async def test_async_fetch_empty_namespace_raises(async_docs: AsyncPreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        await async_docs.fetch(namespace="")


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_missing_ids_silently_omitted(
    async_docs: AsyncPreviewDocuments,
) -> None:
    respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_PARTIAL_FETCH_RESPONSE))

    result = await async_docs.fetch(namespace="my-ns", ids=["doc-1", "does-not-exist"])

    assert isinstance(result, PreviewDocumentFetchResponse)
    assert "doc-1" in result.documents
    assert "does-not-exist" not in result.documents


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_include_fields_none_omits_field(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    await async_docs.fetch(namespace="my-ns", ids=["doc-1"], include_fields=None)

    body = orjson.loads(route.calls.last.request.content)
    assert "include_fields" not in body


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_include_fields_wildcard_passes_through(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    await async_docs.fetch(namespace="my-ns", ids=["doc-1"], include_fields=["*"])

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["*"]


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_empty_body_accepted_by_server(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(FETCH_URL).mock(
        return_value=httpx.Response(200, json={"documents": {}, "namespace": "my-ns"})
    )

    result = await async_docs.fetch(namespace="my-ns")

    body = orjson.loads(route.calls.last.request.content)
    assert body == {}
    assert isinstance(result, PreviewDocumentFetchResponse)
    assert result.documents == {}


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_sends_correct_api_version_header(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    await async_docs.fetch(namespace="my-ns", ids=["doc-1"])

    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_include_fields_named_fields_passes_through(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    await async_docs.fetch(namespace="my-ns", ids=["doc-1"], include_fields=["title", "body"])

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["title", "body"]


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_filter_omitted_when_not_provided(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(FETCH_URL).mock(return_value=httpx.Response(200, json=_FETCH_RESPONSE))

    await async_docs.fetch(namespace="my-ns", ids=["doc-1"])

    body = orjson.loads(route.calls.last.request.content)
    assert "filter" not in body


@pytest.mark.asyncio
@respx.mock
async def test_async_fetch_ids_omitted_when_not_provided(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(FETCH_URL).mock(
        return_value=httpx.Response(200, json=_FILTER_FETCH_RESPONSE)
    )

    await async_docs.fetch(namespace="my-ns", filter={"category": "news"})

    body = orjson.loads(route.calls.last.request.content)
    assert "ids" not in body
