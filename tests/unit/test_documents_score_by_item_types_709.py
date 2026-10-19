"""``score_by`` items that are not a dict or a documents clause are rejected client side (#709)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import (
    DenseVectorQuery,
    EmbedQuery,
    QueryStringQuery,
    SearchQuery,
    SparseEmbedQuery,
    SparseVectorQuery,
    TextQuery,
)
from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeTypeError
from pinecone.index import Index

HOST = "score-by-items-abc123.svc.us-east-1-aws.pinecone.io"
SEARCH_URL = f"https://{HOST}/namespaces/n/documents/search"
OK = httpx.Response(200, json={"namespace": "n", "matches": [], "usage": {"read_units": 1}})

CLAUSE_NAMES = (
    "TextQuery",
    "QueryStringQuery",
    "DenseVectorQuery",
    "SparseVectorQuery",
    "EmbedQuery",
    "SparseEmbedQuery",
)

BAD_ITEMS: list[tuple[str, Any]] = [
    ("SearchQuery", SearchQuery(inputs={"text": "q"}, top_k=1)),
    ("str", "text"),
    ("object", object()),
]

VALID_CLAUSES: list[Any] = [
    TextQuery(fields=["body"], query="q"),
    QueryStringQuery(query="body:q"),
    DenseVectorQuery(field="v", values=[0.1, 0.2]),
    SparseVectorQuery(field="s", sparse_values={"indices": [1], "values": [0.5]}),
    EmbedQuery(field="body", query="q"),
    SparseEmbedQuery(field="body", query="q"),
    {"type": "text", "fields": ["body"], "query": "q"},
]


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def aindex() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=HOST, api_key="test-key")
    yield client
    await client.close()


def _assert_names_clause_types(message: str) -> None:
    for name in CLAUSE_NAMES:
        assert name in message


class TestSync:
    @respx.mock
    @pytest.mark.parametrize("label,item", BAD_ITEMS, ids=[b[0] for b in BAD_ITEMS])
    def test_bad_item_raises_before_any_request(self, index: Index, label: str, item: Any) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=OK)
        with pytest.raises(PineconeTypeError) as excinfo:
            index.documents.search(namespace="n", top_k=1, score_by=[item])
        _assert_names_clause_types(str(excinfo.value))
        assert not route.called

    @respx.mock
    def test_search_query_message_points_at_text_and_embed_query(self, index: Index) -> None:
        respx.post(SEARCH_URL).mock(return_value=OK)
        with pytest.raises(PineconeTypeError) as excinfo:
            index.documents.search(
                namespace="n", top_k=1, score_by=[SearchQuery(inputs={"text": "q"}, top_k=1)]
            )
        message = str(excinfo.value)
        assert "belongs to the legacy records API" in message
        assert "use TextQuery for full-text search or EmbedQuery" in message

    @respx.mock
    def test_bad_item_after_a_valid_clause_is_rejected(self, index: Index) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=OK)
        with pytest.raises(PineconeTypeError):
            index.documents.search(
                namespace="n", top_k=1, score_by=[TextQuery(fields=["body"], query="q"), "text"]
            )
        assert not route.called

    @respx.mock
    @pytest.mark.parametrize("clause", VALID_CLAUSES, ids=[*CLAUSE_NAMES, "dict"])
    def test_every_clause_type_and_dict_is_still_accepted(self, index: Index, clause: Any) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=OK)
        index.documents.search(namespace="n", top_k=1, score_by=[clause])
        assert route.called


class TestAsync:
    @respx.mock
    @pytest.mark.parametrize("label,item", BAD_ITEMS, ids=[b[0] for b in BAD_ITEMS])
    async def test_bad_item_raises_before_any_request(
        self, aindex: AsyncIndex, label: str, item: Any
    ) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=OK)
        with pytest.raises(PineconeTypeError) as excinfo:
            await aindex.documents.search(namespace="n", top_k=1, score_by=[item])
        _assert_names_clause_types(str(excinfo.value))
        assert not route.called

    @respx.mock
    async def test_search_query_message_points_at_text_and_embed_query(
        self, aindex: AsyncIndex
    ) -> None:
        respx.post(SEARCH_URL).mock(return_value=OK)
        with pytest.raises(PineconeTypeError) as excinfo:
            await aindex.documents.search(
                namespace="n", top_k=1, score_by=[SearchQuery(inputs={"text": "q"}, top_k=1)]
            )
        message = str(excinfo.value)
        assert "belongs to the legacy records API" in message
        assert "use TextQuery for full-text search or EmbedQuery" in message

    @respx.mock
    async def test_bad_item_after_a_valid_clause_is_rejected(self, aindex: AsyncIndex) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=OK)
        with pytest.raises(PineconeTypeError):
            await aindex.documents.search(
                namespace="n", top_k=1, score_by=[TextQuery(fields=["body"], query="q"), "text"]
            )
        assert not route.called

    @respx.mock
    @pytest.mark.parametrize("clause", VALID_CLAUSES, ids=[*CLAUSE_NAMES, "dict"])
    async def test_every_clause_type_and_dict_is_still_accepted(
        self, aindex: AsyncIndex, clause: Any
    ) -> None:
        route = respx.post(SEARCH_URL).mock(return_value=OK)
        await aindex.documents.search(namespace="n", top_k=1, score_by=[clause])
        assert route.called
