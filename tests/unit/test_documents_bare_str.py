"""A bare ``str`` for a ``Sequence[str]`` documents argument is rejected, not split into characters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import PineconeTypeError
from pinecone._internal import documents_helpers as h
from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index
from pinecone.models.documents.score_by import (
    DenseVectorQuery,
    EmbedQuery,
    SparseEmbedQuery,
    SparseVectorQuery,
    TextQuery,
)

HOST = "bare-str-abc123.svc.us-east-1-aws.pinecone.io"
NS = "ns"

_TEXT = TextQuery(query="x", fields=["abstract"])

_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("delete", {"namespace": NS, "ids": "abc"}),
    ("fetch", {"namespace": NS, "ids": "abc"}),
    ("fetch", {"namespace": NS, "ids": ["abc"], "include_fields": "title"}),
    ("search", {"namespace": NS, "top_k": 2, "score_by": [_TEXT], "include_fields": "title"}),
    (
        "update",
        {"namespace": NS, "filter": {"a": {"$eq": 1}}, "remove_fields": "title"},
    ),
]
_IDS = ["delete-ids", "fetch-ids", "fetch-include_fields", "search-include_fields", "update-remove"]


@pytest.fixture
def sync_index() -> Iterator[Index]:
    client = Index(host=HOST, api_key="k")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=HOST, api_key="k")
    yield client
    await client.close()


@pytest.mark.parametrize(("method", "kwargs"), _CALLS, ids=_IDS)
@respx.mock
def test_sync_rejects_bare_str_and_sends_nothing(
    method: str, kwargs: dict[str, Any], sync_index: Index
) -> None:
    route = respx.route().mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(PineconeTypeError, match="not a single str") as excinfo:
        getattr(sync_index.documents, method)(**kwargs)
    assert type(excinfo.value) is PineconeTypeError
    assert not route.called


@pytest.mark.parametrize(("method", "kwargs"), _CALLS, ids=_IDS)
@respx.mock
async def test_async_rejects_bare_str_and_sends_nothing(
    method: str, kwargs: dict[str, Any], async_index: AsyncIndex
) -> None:
    route = respx.route().mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(PineconeTypeError, match="not a single str") as excinfo:
        await getattr(async_index.documents, method)(**kwargs)
    assert type(excinfo.value) is PineconeTypeError
    assert not route.called


def test_helpers_reject_bare_str() -> None:
    with pytest.raises(PineconeTypeError):
        h._build_delete_documents_body(ids="abc", filter=None, delete_all=False)
    with pytest.raises(PineconeTypeError):
        h._build_fetch_documents_body(
            ids="abc", filter=None, include_fields=None, pagination_token=None, limit=None
        )
    with pytest.raises(PineconeTypeError):
        h._build_search_documents_body(
            score_by=[_TEXT], top_k=2, include_fields="title", filter=None
        )
    with pytest.raises(PineconeTypeError):
        h._build_update_documents_body(
            documents=None, filter={"a": {"$eq": 1}}, set_fields=None, remove_fields="title"
        )


def test_sequences_still_accepted() -> None:
    assert h._build_delete_documents_body(ids=("a", "b"), filter=None, delete_all=False) == {
        "ids": ["a", "b"]
    }
    assert h._build_search_documents_body(
        score_by=[_TEXT], top_k=2, include_fields=["title"], filter=None
    )["include_fields"] == ["title"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: TextQuery(query="x", fields="title"),
        lambda: DenseVectorQuery(values=[0.1], fields="e"),
        lambda: SparseVectorQuery(sparse_values={"indices": [1], "values": [0.1]}, fields="e"),
        lambda: EmbedQuery(query="x", fields="e"),
        lambda: SparseEmbedQuery(query="x", fields="e"),
    ],
    ids=["text", "dense", "sparse", "embed", "sparse_embed"],
)
def test_clauses_reject_bare_str_fields(build: Any) -> None:
    with pytest.raises(PineconeTypeError, match="not a single str") as excinfo:
        build()
    assert type(excinfo.value) is PineconeTypeError
