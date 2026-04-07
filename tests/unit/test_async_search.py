"""Unit tests for AsyncIndex.search() method."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.search import SearchRecordsResponse

INDEX_HOST = "my-index-abc123.svc.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
SEARCH_URL_NS = f"{INDEX_HOST_HTTPS}/records/namespaces/test-ns/search"

SEARCH_RESPONSE: dict[str, object] = {
    "result": {
        "hits": [
            {"_id": "r1", "_score": 0.95, "fields": {"chunk_text": "hello world"}},
            {"_id": "r2", "_score": 0.82, "fields": {"chunk_text": "foo bar"}},
        ]
    },
    "usage": {"read_units": 5, "embed_total_tokens": 10},
}


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


class TestAsyncSearch:
    """AsyncIndex.search() — happy paths and request body verification."""

    @respx.mock
    @pytest.mark.anyio
    async def test_async_search_with_text_inputs(self) -> None:
        respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_async_index()
        response = await idx.search(namespace="test-ns", top_k=10, inputs={"text": "hello"})

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 2
        assert response.result.hits[0].id == "r1"
        assert response.result.hits[0].score == pytest.approx(0.95)
        assert response.usage.read_units == 5

    @respx.mock
    @pytest.mark.anyio
    async def test_async_search_with_vector(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_async_index()
        await idx.search(namespace="test-ns", top_k=10, vector=[0.1, 0.2, 0.3])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == [0.1, 0.2, 0.3]

    @respx.mock
    @pytest.mark.anyio
    async def test_async_search_with_id(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_async_index()
        await idx.search(namespace="test-ns", top_k=10, id="vec1")

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["id"] == "vec1"

    @respx.mock
    @pytest.mark.anyio
    async def test_async_search_with_filter(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_async_index()
        await idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            filter={"genre": {"$eq": "sci-fi"}},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["filter"] == {"genre": {"$eq": "sci-fi"}}

    @respx.mock
    @pytest.mark.anyio
    async def test_async_search_with_fields(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_async_index()
        await idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            fields=["chunk_text", "title"],
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["fields"] == ["chunk_text", "title"]

    @respx.mock
    @pytest.mark.anyio
    async def test_async_search_with_rerank(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_async_index()
        await idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            rerank={
                "model": "bge-reranker-v2-m3",
                "rank_fields": ["chunk_text"],
                "top_n": 5,
            },
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["rerank"]["model"] == "bge-reranker-v2-m3"
        assert body["rerank"]["rank_fields"] == ["chunk_text"]
        assert body["rerank"]["top_n"] == 5

    @pytest.mark.anyio
    async def test_async_search_top_k_validation(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            await idx.search(namespace="test-ns", top_k=0, inputs={"text": "hello"})

    @pytest.mark.anyio
    async def test_async_search_rerank_missing_model(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="model"):
            await idx.search(
                namespace="test-ns",
                top_k=10,
                inputs={"text": "hello"},
                rerank={"rank_fields": ["text"]},
            )

    @pytest.mark.anyio
    async def test_async_search_rerank_missing_rank_fields(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="rank_fields"):
            await idx.search(
                namespace="test-ns",
                top_k=10,
                inputs={"text": "hello"},
                rerank={"model": "bge-reranker-v2-m3"},
            )

    @pytest.mark.anyio
    async def test_async_search_no_query_source_raises(self) -> None:
        idx = _make_async_index()
        with pytest.raises(ValidationError, match="At least one of inputs, vector, or id"):
            await idx.search(namespace="test-ns", top_k=5)

    def test_async_search_keyword_only(self) -> None:
        idx = _make_async_index()
        with pytest.raises(TypeError):
            idx.search("ns", 10)  # type: ignore[misc]
