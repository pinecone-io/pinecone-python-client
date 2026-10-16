"""``batch_upsert`` against the documents service's request-size cap.

The server rejects an encoded request body over 2 MiB with 413. A count-only
batcher sends ``batch_size=1000`` documents of ~20 KB each as one 20 MB request
and gives up; ``batch_upsert`` must instead halve a rejected batch and resend,
and must not keep resending a single document the server will never accept.
The fake server below enforces the cap on the real encoded body, so the tests
do not depend on how the SDK picks its split points.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import orjson
import pytest
import respx

from pinecone._internal.bulk import get_registry
from pinecone.async_client.async_index import AsyncIndex
from pinecone.async_client.documents import AsyncDocuments
from pinecone.client.documents import Documents
from pinecone.index import Index

HOST = "documents-cap-idx.svc.us-east-1-aws.pinecone.io"
NS = "articles-en"
UPSERT_URL = f"https://{HOST}/namespaces/{NS}/documents/upsert"
CAP_BYTES = 2 * 1024 * 1024
VECTOR = [0.0123456789] * 1024


@pytest.fixture(autouse=True)
def _fresh_registry() -> Iterator[None]:
    get_registry()._reset()
    yield
    get_registry()._reset()


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=HOST, api_key="test-key")
    yield client
    await client.close()


def _docs(n: int, **extra: Any) -> list[dict[str, Any]]:
    return [{"_id": f"doc-{i}", "embedding": VECTOR, **extra} for i in range(n)]


class _CappedServer:
    """Mimics the documents service: 413 over the cap, 202 otherwise."""

    def __init__(self) -> None:
        self.received: list[str] = []
        self.request_sizes: list[int] = []
        self.rejected: list[list[str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_sizes.append(len(request.content))
        if len(request.content) > CAP_BYTES:
            self.rejected.append([d["_id"] for d in orjson.loads(request.content)["documents"]])
            return httpx.Response(
                413, json={"error": {"code": "PAYLOAD_TOO_LARGE", "message": "Payload Too Large"}}
            )
        docs = orjson.loads(request.content)["documents"]
        self.received.extend(d["_id"] for d in docs)
        return httpx.Response(202, json={"upserted_count": len(docs)})


def test_fixture_documents_really_exceed_the_cap() -> None:
    """Guards the premise: if the fixture shrank, every test below would pass
    without exercising a split."""
    assert len(orjson.dumps({"documents": _docs(1000)})) > 5 * CAP_BYTES


class TestSyncPayloadCap:
    @respx.mock
    def test_oversize_batch_is_split_until_every_document_lands(self, index: Index) -> None:
        server = _CappedServer()
        respx.post(UPSERT_URL).mock(side_effect=server)

        result = index.documents.batch_upsert(
            namespace=NS, documents=_docs(1000), batch_size=1000, show_progress=False
        )

        assert result.successful_item_count == 1000
        assert result.failed_item_count == 0
        assert result.errors == []
        assert sorted(server.received) == sorted(f"doc-{i}" for i in range(1000))
        assert max(s for s in server.request_sizes if s <= CAP_BYTES) > CAP_BYTES // 2

    @respx.mock
    def test_batch_under_the_cap_is_sent_once(self, index: Index) -> None:
        server = _CappedServer()
        respx.post(UPSERT_URL).mock(side_effect=server)

        result = index.documents.batch_upsert(
            namespace=NS, documents=_docs(50), batch_size=50, show_progress=False
        )

        assert result.successful_item_count == 50
        assert len(server.request_sizes) == 1

    @respx.mock
    def test_single_oversize_document_fails_without_resending(self, index: Index) -> None:
        server = _CappedServer()
        respx.post(UPSERT_URL).mock(side_effect=server)
        docs = _docs(10)
        docs[3] = {"_id": "huge", "blob": "x" * (CAP_BYTES + 1)}

        result = index.documents.batch_upsert(
            namespace=NS, documents=docs, batch_size=10, show_progress=False
        )

        assert result.successful_item_count == 9
        assert result.failed_item_count == 1
        assert [d["_id"] for d in result.failed_items] == ["huge"]
        assert [e.retryable for e in result.errors] == [False]
        assert sorted(server.received) == sorted(d["_id"] for d in docs if d["_id"] != "huge")
        assert server.rejected[-1] == ["huge"]
        assert sum(r == ["huge"] for r in server.rejected) == 1, "resent a lone oversize document"

    @respx.mock
    def test_non_413_failure_is_not_split(self, index: Index) -> None:
        calls = {"n": 0}

        def bad_request(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": {"message": "bad"}})

        respx.post(UPSERT_URL).mock(side_effect=bad_request)

        result = index.documents.batch_upsert(
            namespace=NS, documents=_docs(8), batch_size=8, show_progress=False
        )

        assert calls["n"] == 1
        assert result.failed_item_count == 8
        assert [e.batch_index for e in result.errors] == [0]


class TestAsyncPayloadCap:
    @respx.mock
    async def test_oversize_batch_is_split_until_every_document_lands(
        self, async_index: AsyncIndex
    ) -> None:
        server = _CappedServer()
        respx.post(UPSERT_URL).mock(side_effect=server)

        result = await async_index.documents.batch_upsert(
            namespace=NS, documents=_docs(1000), batch_size=1000, show_progress=False
        )

        assert result.successful_item_count == 1000
        assert result.failed_item_count == 0
        assert sorted(server.received) == sorted(f"doc-{i}" for i in range(1000))

    @respx.mock
    async def test_batch_under_the_cap_is_sent_once(self, async_index: AsyncIndex) -> None:
        server = _CappedServer()
        respx.post(UPSERT_URL).mock(side_effect=server)

        result = await async_index.documents.batch_upsert(
            namespace=NS, documents=_docs(50), batch_size=50, show_progress=False
        )

        assert result.successful_item_count == 50
        assert len(server.request_sizes) == 1

    @respx.mock
    async def test_single_oversize_document_fails_without_resending(
        self, async_index: AsyncIndex
    ) -> None:
        server = _CappedServer()
        respx.post(UPSERT_URL).mock(side_effect=server)
        docs = _docs(10)
        docs[3] = {"_id": "huge", "blob": "x" * (CAP_BYTES + 1)}

        result = await async_index.documents.batch_upsert(
            namespace=NS, documents=docs, batch_size=10, show_progress=False
        )

        assert result.successful_item_count == 9
        assert [d["_id"] for d in result.failed_items] == ["huge"]
        assert [e.retryable for e in result.errors] == [False]
        assert server.rejected[-1] == ["huge"]
        assert sum(r == ["huge"] for r in server.rejected) == 1

    @respx.mock
    async def test_non_413_failure_is_not_split(self, async_index: AsyncIndex) -> None:
        calls = {"n": 0}

        def bad_request(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": {"message": "bad"}})

        respx.post(UPSERT_URL).mock(side_effect=bad_request)

        result = await async_index.documents.batch_upsert(
            namespace=NS, documents=_docs(8), batch_size=8, show_progress=False
        )

        assert calls["n"] == 1
        assert result.failed_item_count == 8


@pytest.mark.parametrize("cls", [Documents, AsyncDocuments])
class TestDocstrings:
    def test_upsert_says_requests_are_capped_by_size(self, cls: type) -> None:
        doc = inspect.getdoc(cls.upsert) or ""
        assert "encoded size" in doc
        assert "2 MiB" not in doc and "2097152" not in doc

    def test_batch_upsert_says_it_handles_the_size_cap(self, cls: type) -> None:
        doc = inspect.getdoc(cls.batch_upsert) or ""
        assert "encoded size" in doc
        assert "halv" in doc
        assert "2 MiB" not in doc and "2097152" not in doc
