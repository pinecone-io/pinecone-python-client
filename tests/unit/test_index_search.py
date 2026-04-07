"""Unit tests for Index.search() method."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import Index
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


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


class TestSearch:
    """Index.search() — happy paths and request body verification."""

    @respx.mock
    def test_search_with_text_inputs(self) -> None:
        respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        response = idx.search(namespace="test-ns", top_k=10, inputs={"text": "hello"})

        assert isinstance(response, SearchRecordsResponse)
        assert len(response.result.hits) == 2
        assert response.result.hits[0].id == "r1"
        assert response.result.hits[0].score == pytest.approx(0.95)
        assert response.usage.read_units == 5

    @respx.mock
    def test_search_with_vector(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, vector=[0.1, 0.2, 0.3])

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["vector"] == [0.1, 0.2, 0.3]

    @respx.mock
    def test_search_with_id(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, id="vec1")

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["id"] == "vec1"

    @respx.mock
    def test_search_with_filter(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            filter={"genre": {"$eq": "sci-fi"}},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["filter"] == {"genre": {"$eq": "sci-fi"}}

    @respx.mock
    def test_search_with_fields(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            fields=["chunk_text", "title"],
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["fields"] == ["chunk_text", "title"]

    @respx.mock
    def test_search_with_rerank(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
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

    def test_search_top_k_validation(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="top_k must be a positive integer"):
            idx.search(namespace="test-ns", top_k=0, inputs={"text": "hello"})

    def test_search_rerank_missing_model(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="model"):
            idx.search(
                namespace="test-ns",
                top_k=10,
                inputs={"text": "hello"},
                rerank={"rank_fields": ["text"]},
            )

    def test_search_rerank_missing_rank_fields(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="rank_fields"):
            idx.search(
                namespace="test-ns",
                top_k=10,
                inputs={"text": "hello"},
                rerank={"model": "bge-reranker-v2-m3"},
            )

    def test_search_no_query_source_raises(self) -> None:
        idx = _make_index()
        with pytest.raises(ValidationError, match="At least one of inputs, vector, or id"):
            idx.search(namespace="test-ns", top_k=5)

    def test_search_keyword_only(self) -> None:
        idx = _make_index()
        with pytest.raises(TypeError):
            idx.search("ns", 10)  # type: ignore[misc]

    @respx.mock
    def test_search_match_terms_included_in_query_body(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(
            namespace="test-ns",
            top_k=10,
            inputs={"text": "hello"},
            match_terms={"strategy": "all", "terms": ["animal", "duck"]},
        )

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert body["query"]["match_terms"] == {
            "strategy": "all",
            "terms": ["animal", "duck"],
        }

    @respx.mock
    def test_search_match_terms_omitted_when_none(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, inputs={"text": "hello"})

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert "match_terms" not in body["query"]

    @respx.mock
    def test_search_default_fields_none(self) -> None:
        route = respx.post(SEARCH_URL_NS).mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE),
        )
        idx = _make_index()
        idx.search(namespace="test-ns", top_k=10, inputs={"text": "hello"})

        import orjson

        body = orjson.loads(route.calls.last.request.content)
        assert "fields" not in body
