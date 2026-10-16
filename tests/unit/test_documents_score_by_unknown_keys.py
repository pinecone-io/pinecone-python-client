"""A dict ``score_by`` clause with a key its type does not define is rejected, never dropped."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import msgspec
import pytest
import respx

from pinecone._internal.documents_helpers import _build_search_documents_body
from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeValueError
from pinecone.index import Index
from pinecone.models.documents.requests import SearchDocumentsRequest

INDEX_HOST = "score-by-index-abc123.svc.us-east-1-aws.pinecone.io"
SEARCH_URL = f"https://{INDEX_HOST}/namespaces/ns1/documents/search"

CLAUSES: dict[str, dict[str, Any]] = {
    "text": {"query": "dark matter", "fields": ["abstract"]},
    "query_string": {"query": "title:entropy^5"},
    "dense_vector": {"field": "vec", "values": [0.1, 0.2]},
    "sparse_vector": {"field": "sp", "sparse_values": {"indices": [1], "values": [0.5]}},
    "embed": {"field": "body", "query": "fox"},
    "sparse_embed": {"field": "body", "query": "fox"},
}

kinds = pytest.mark.parametrize("kind", list(CLAUSES))


def _clause(kind: str, **extra: Any) -> dict[str, Any]:
    return {"type": kind, **CLAUSES[kind], **extra}


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=INDEX_HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=INDEX_HOST, api_key="test-key")
    yield client
    await client.close()


class TestUnknownKeyOnDictClause:
    @kinds
    def test_request_model_rejects_the_key_by_name(self, kind: str) -> None:
        with pytest.raises(msgspec.ValidationError, match="unknown field `boost`"):
            SearchDocumentsRequest(score_by=[_clause(kind, boost=2)], top_k=3)

    @kinds
    def test_body_builder_raises_pinecone_value_error(self, kind: str) -> None:
        with pytest.raises(PineconeValueError, match="unknown field `boost`"):
            _build_search_documents_body(
                score_by=[_clause(kind, boost=2)], top_k=3, include_fields=None, filter=None
            )

    @kinds
    def test_known_keys_alone_still_build(self, kind: str) -> None:
        body = _build_search_documents_body(
            score_by=[_clause(kind)], top_k=3, include_fields=None, filter=None
        )
        assert body["score_by"][0]["type"] == kind

    def test_unknown_key_among_combined_text_clauses_is_rejected(self) -> None:
        with pytest.raises(PineconeValueError, match="unknown field `weight`"):
            _build_search_documents_body(
                score_by=[_clause("text"), _clause("query_string", weight=3)],
                top_k=3,
                include_fields=None,
                filter=None,
            )

    @kinds
    @respx.mock
    def test_sync_search_raises_before_any_request(self, index: Index, kind: str) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(PineconeValueError, match="unknown field `boost`"):
            index.documents.search(namespace="ns1", top_k=3, score_by=[_clause(kind, boost=2)])
        assert not route.called

    @kinds
    @respx.mock
    async def test_async_search_raises_before_any_request(
        self, async_index: AsyncIndex, kind: str
    ) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(PineconeValueError, match="unknown field `boost`"):
            await async_index.documents.search(
                namespace="ns1", top_k=3, score_by=[_clause(kind, boost=2)]
            )
        assert not route.called

    @respx.mock
    def test_valid_clause_is_sent_unchanged(self, index: Index) -> None:
        route = respx.post(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json={"matches": [], "namespace": "ns1", "usage": {"read_units": 1}}
            )
        )
        index.documents.search(namespace="ns1", top_k=3, score_by=[_clause("text")])
        assert json.loads(route.calls.last.request.content)["score_by"] == [_clause("text")]
