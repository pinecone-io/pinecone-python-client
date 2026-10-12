"""SDK struct values such as ``SparseValues`` are accepted as document field values (#698)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from pinecone import SparseValues
from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index
from pinecone.models.documents.document import DocumentRecord

HOST = "sparse-values-abc123.svc.us-east-1-aws.pinecone.io"
URL = f"https://{HOST}/namespaces/n/documents"
SPARSE = SparseValues(indices=[1, 7], values=[1.0, 0.5])
WIRE = {"indices": [1, 7], "values": [1.0, 0.5]}
UPSERT_OK = httpx.Response(202, json={"upserted_count": 1})


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


def _body(route: respx.Route) -> dict[str, Any]:
    return dict(json.loads(route.calls.last.request.content))


DOCS = [{"_id": "x", "emb": [0.1, 0.2], "kw": SPARSE}]
EXPECTED = {"documents": [{"_id": "x", "emb": [0.1, 0.2], "kw": WIRE}]}


class TestSync:
    @respx.mock
    def test_upsert_sends_sparse_values_as_indices_and_values(self, index: Index) -> None:
        route = respx.post(f"{URL}/upsert").mock(return_value=UPSERT_OK)
        index.documents.upsert(namespace="n", documents=DOCS)
        assert _body(route) == EXPECTED

    @respx.mock
    def test_upsert_accepts_document_record(self, index: Index) -> None:
        route = respx.post(f"{URL}/upsert").mock(return_value=UPSERT_OK)
        index.documents.upsert(namespace="n", documents=[DocumentRecord(DOCS[0])])
        assert _body(route) == EXPECTED

    @respx.mock
    def test_batch_upsert_sends_sparse_values(self, index: Index) -> None:
        route = respx.post(f"{URL}/upsert").mock(return_value=UPSERT_OK)
        index.documents.batch_upsert(namespace="n", documents=DOCS)
        assert _body(route) == EXPECTED

    @respx.mock
    def test_update_sends_sparse_values(self, index: Index) -> None:
        route = respx.post(f"{URL}/update").mock(return_value=httpx.Response(202, json={}))
        index.documents.update(namespace="n", documents=[{"_id": "x", "kw": SPARSE}])
        assert _body(route) == {"documents": [{"_id": "x", "kw": WIRE}]}

    @respx.mock
    def test_update_by_filter_sends_sparse_values(self, index: Index) -> None:
        route = respx.post(f"{URL}/update").mock(return_value=httpx.Response(202, json={}))
        index.documents.update(namespace="n", filter={"a": 1}, set_fields={"kw": SPARSE})
        assert _body(route)["set_fields"] == {"kw": WIRE}

    @respx.mock
    def test_plain_dict_value_is_unchanged(self, index: Index) -> None:
        route = respx.post(f"{URL}/upsert").mock(return_value=UPSERT_OK)
        index.documents.upsert(namespace="n", documents=[{"_id": "x", "kw": WIRE}])
        assert _body(route) == {"documents": [{"_id": "x", "kw": WIRE}]}


class TestAsync:
    @respx.mock
    async def test_upsert_sends_sparse_values(self, aindex: AsyncIndex) -> None:
        route = respx.post(f"{URL}/upsert").mock(return_value=UPSERT_OK)
        await aindex.documents.upsert(namespace="n", documents=DOCS)
        assert _body(route) == EXPECTED

    @respx.mock
    async def test_batch_upsert_sends_sparse_values(self, aindex: AsyncIndex) -> None:
        route = respx.post(f"{URL}/upsert").mock(return_value=UPSERT_OK)
        await aindex.documents.batch_upsert(namespace="n", documents=DOCS)
        assert _body(route) == EXPECTED

    @respx.mock
    async def test_update_sends_sparse_values(self, aindex: AsyncIndex) -> None:
        route = respx.post(f"{URL}/update").mock(return_value=httpx.Response(202, json={}))
        await aindex.documents.update(namespace="n", documents=[{"_id": "x", "kw": SPARSE}])
        assert _body(route) == {"documents": [{"_id": "x", "kw": WIRE}]}
