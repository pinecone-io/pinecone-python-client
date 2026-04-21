"""Unit tests for PreviewDocuments.search and AsyncPreviewDocuments.search."""

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
from pinecone.preview.models.documents import PreviewDocument, PreviewDocumentSearchResponse
from pinecone.preview.models.score_by import PreviewDenseVectorQuery, PreviewTextQuery

INDEX_HOST = "https://idx-abc.svc.pinecone.io"
SEARCH_URL = f"{INDEX_HOST}/namespaces/my-ns/documents/search"

_SEARCH_RESPONSE = {
    "matches": [
        {"_id": "doc-1", "_score": 0.95, "title": "hello"},
        {"_id": "doc-2", "_score": 0.80},
    ],
    "namespace": "my-ns",
    "usage": {"read_units": 5},
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
# Sync — happy path: typed queries serialize to wire shapes (scenario a)
# ---------------------------------------------------------------------------


@respx.mock
def test_search_typed_text_query_serializes_to_wire_shape(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[PreviewTextQuery(field="title", query="hello world")],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["score_by"] == [{"type": "text", "field": "title", "query": "hello world"}]


@respx.mock
def test_search_typed_dense_vector_query_serializes_to_wire_shape(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=3,
        score_by=[PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["score_by"] == [
        {"type": "dense_vector", "field": "embedding", "values": [0.1, 0.2, 0.3]}
    ]


@respx.mock
def test_search_mixed_typed_and_typed_score_by(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[
            PreviewTextQuery(field="body", query="test"),
            PreviewDenseVectorQuery(field="vec", values=[0.5, 0.6]),
        ],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert len(body["score_by"]) == 2
    assert body["score_by"][0]["type"] == "text"
    assert body["score_by"][1]["type"] == "dense_vector"


# ---------------------------------------------------------------------------
# Sync — dict passthrough (scenario b)
# ---------------------------------------------------------------------------


@respx.mock
def test_search_dict_score_by_passes_through_verbatim(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    score_by_dict = {"type": "text", "field": "title", "query": "pinecone"}
    docs.search(namespace="my-ns", top_k=5, score_by=[score_by_dict])

    body = orjson.loads(route.calls.last.request.content)
    assert body["score_by"] == [score_by_dict]


# ---------------------------------------------------------------------------
# Sync — validation failures (scenario c)
# ---------------------------------------------------------------------------


def test_search_empty_namespace_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        docs.search(
            namespace="",
            top_k=5,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )


def test_search_top_k_zero_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="top_k"):
        docs.search(
            namespace="ns",
            top_k=0,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )


def test_search_top_k_10001_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="top_k"):
        docs.search(
            namespace="ns",
            top_k=10001,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )


def test_search_top_k_1_accepted(docs: PreviewDocuments) -> None:
    with respx.mock:
        respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        result = docs.search(
            namespace="my-ns",
            top_k=1,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )
    assert isinstance(result, PreviewDocumentSearchResponse)


def test_search_top_k_10000_accepted(docs: PreviewDocuments) -> None:
    with respx.mock:
        respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        result = docs.search(
            namespace="my-ns",
            top_k=10000,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )
    assert isinstance(result, PreviewDocumentSearchResponse)


def test_search_empty_score_by_raises(docs: PreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="score_by"):
        docs.search(namespace="ns", top_k=5, score_by=[])


# ---------------------------------------------------------------------------
# Sync — include_fields semantics (scenarios d, e)
# ---------------------------------------------------------------------------


@respx.mock
def test_search_include_fields_none_sends_empty_list(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        include_fields=None,
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == []


@respx.mock
def test_search_include_fields_wildcard_passes_through(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        include_fields=["*"],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["*"]


@respx.mock
def test_search_include_fields_named_fields_passes_through(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        include_fields=["title", "body"],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["title", "body"]


# ---------------------------------------------------------------------------
# Sync — response deserialization with dynamic attribute access (scenario f)
# ---------------------------------------------------------------------------


@respx.mock
def test_search_response_has_preview_documents_with_dynamic_attributes(
    docs: PreviewDocuments,
) -> None:
    respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    result = docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "title", "query": "hello"}],
    )

    assert isinstance(result, PreviewDocumentSearchResponse)
    assert result.namespace == "my-ns"
    assert result.usage is not None
    assert result.usage.read_units == 5
    assert len(result.matches) == 2

    first = result.matches[0]
    assert isinstance(first, PreviewDocument)
    assert first._id == "doc-1"
    assert first.score == pytest.approx(0.95)
    assert first.title == "hello"  # type: ignore[attr-defined]

    second = result.matches[1]
    assert second._id == "doc-2"
    assert second.score == pytest.approx(0.80)


@respx.mock
def test_search_sends_correct_api_version_header(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
    )

    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@respx.mock
def test_search_filter_included_in_request_when_provided(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        filter={"category": {"$eq": "news"}},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["filter"] == {"category": {"$eq": "news"}}


@respx.mock
def test_search_filter_omitted_when_not_provided(docs: PreviewDocuments) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert "filter" not in body


# ---------------------------------------------------------------------------
# Async — mirrors sync (scenarios a-f)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_search_typed_text_query_serializes_to_wire_shape(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[PreviewTextQuery(field="title", query="hello world")],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["score_by"] == [{"type": "text", "field": "title", "query": "hello world"}]


@pytest.mark.asyncio
@respx.mock
async def test_async_search_dict_score_by_passes_through(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    score_by_dict = {"type": "text", "field": "f", "query": "q"}
    await async_docs.search(namespace="my-ns", top_k=5, score_by=[score_by_dict])

    body = orjson.loads(route.calls.last.request.content)
    assert body["score_by"] == [score_by_dict]


@pytest.mark.asyncio
async def test_async_search_empty_namespace_raises(async_docs: AsyncPreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="namespace"):
        await async_docs.search(
            namespace="",
            top_k=5,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )


@pytest.mark.asyncio
async def test_async_search_top_k_zero_raises(async_docs: AsyncPreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="top_k"):
        await async_docs.search(
            namespace="ns",
            top_k=0,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )


@pytest.mark.asyncio
async def test_async_search_top_k_10001_raises(async_docs: AsyncPreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="top_k"):
        await async_docs.search(
            namespace="ns",
            top_k=10001,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )


@pytest.mark.asyncio
async def test_async_search_empty_score_by_raises(async_docs: AsyncPreviewDocuments) -> None:
    with pytest.raises(ValidationError, match="score_by"):
        await async_docs.search(namespace="ns", top_k=5, score_by=[])


@pytest.mark.asyncio
@respx.mock
async def test_async_search_include_fields_none_sends_empty_list(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        include_fields=None,
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == []


@pytest.mark.asyncio
@respx.mock
async def test_async_search_include_fields_wildcard_passes_through(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        include_fields=["*"],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["*"]


@pytest.mark.asyncio
@respx.mock
async def test_async_search_response_has_preview_documents_with_dynamic_attributes(
    async_docs: AsyncPreviewDocuments,
) -> None:
    respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    result = await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "title", "query": "hello"}],
    )

    assert isinstance(result, PreviewDocumentSearchResponse)
    assert result.namespace == "my-ns"
    assert len(result.matches) == 2
    first = result.matches[0]
    assert isinstance(first, PreviewDocument)
    assert first._id == "doc-1"
    assert first.title == "hello"  # type: ignore[attr-defined]
    assert first.score == pytest.approx(0.95)


@pytest.mark.asyncio
@respx.mock
async def test_async_search_typed_dense_vector_query_serializes_to_wire_shape(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=3,
        score_by=[PreviewDenseVectorQuery(field="embedding", values=[0.1, 0.2, 0.3])],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["score_by"] == [
        {"type": "dense_vector", "field": "embedding", "values": [0.1, 0.2, 0.3]}
    ]


@pytest.mark.asyncio
@respx.mock
async def test_async_search_mixed_typed_and_typed_score_by(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[
            PreviewTextQuery(field="body", query="test"),
            PreviewDenseVectorQuery(field="vec", values=[0.5, 0.6]),
        ],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert len(body["score_by"]) == 2
    assert body["score_by"][0]["type"] == "text"
    assert body["score_by"][1]["type"] == "dense_vector"


@pytest.mark.asyncio
async def test_async_search_top_k_1_accepted(async_docs: AsyncPreviewDocuments) -> None:
    with respx.mock:
        respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        result = await async_docs.search(
            namespace="my-ns",
            top_k=1,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )
    assert isinstance(result, PreviewDocumentSearchResponse)


@pytest.mark.asyncio
async def test_async_search_top_k_10000_accepted(async_docs: AsyncPreviewDocuments) -> None:
    with respx.mock:
        respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))
        result = await async_docs.search(
            namespace="my-ns",
            top_k=10000,
            score_by=[{"type": "text", "field": "f", "query": "q"}],
        )
    assert isinstance(result, PreviewDocumentSearchResponse)


@pytest.mark.asyncio
@respx.mock
async def test_async_search_include_fields_named_fields_passes_through(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        include_fields=["title", "body"],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["include_fields"] == ["title", "body"]


@pytest.mark.asyncio
@respx.mock
async def test_async_search_sends_correct_api_version_header(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
    )

    assert route.calls.last.request.headers.get("X-Pinecone-Api-Version") == INDEXES_API_VERSION


@pytest.mark.asyncio
@respx.mock
async def test_async_search_filter_included_in_request_when_provided(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
        filter={"category": {"$eq": "news"}},
    )

    body = orjson.loads(route.calls.last.request.content)
    assert body["filter"] == {"category": {"$eq": "news"}}


@pytest.mark.asyncio
@respx.mock
async def test_async_search_filter_omitted_when_not_provided(
    async_docs: AsyncPreviewDocuments,
) -> None:
    route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json=_SEARCH_RESPONSE))

    await async_docs.search(
        namespace="my-ns",
        top_k=5,
        score_by=[{"type": "text", "field": "f", "query": "q"}],
    )

    body = orjson.loads(route.calls.last.request.content)
    assert "filter" not in body
