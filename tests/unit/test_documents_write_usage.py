"""``usage.embed_total_tokens`` on documents upsert and update responses.

The server attaches ``usage`` only when the write ran at least one embedding call, so
each response is exercised with and without it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import msgspec
import pytest
import respx

from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import ResponseParsingError
from pinecone.index import Index
from pinecone.models.documents.responses import (
    DocumentUpdateUsage,
    DocumentUpsertUsage,
    UpdateDocumentsResponse,
    UpsertDocumentsResponse,
)

INDEX_HOST = "embed-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{INDEX_HOST}"
NS = "ns1"
UPSERT_URL = f"{BASE_URL}/namespaces/{NS}/documents/upsert"
UPDATE_URL = f"{BASE_URL}/namespaces/{NS}/documents/update"

DOCS = [{"_id": "d1", "body": "The quick brown fox"}]


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


class TestDecode:
    def test_upsert_usage_read_when_present(self) -> None:
        decoded = msgspec.json.decode(
            b'{"upserted_count": 2, "usage": {"embed_total_tokens": 12}}',
            type=UpsertDocumentsResponse,
        )

        assert decoded.upserted_count == 2
        assert decoded.usage == DocumentUpsertUsage(embed_total_tokens=12)

    def test_upsert_usage_absent_is_none(self) -> None:
        decoded = msgspec.json.decode(b'{"upserted_count": 2}', type=UpsertDocumentsResponse)

        assert decoded.usage is None

    def test_update_usage_read_when_present(self) -> None:
        decoded = msgspec.json.decode(
            b'{"usage": {"embed_total_tokens": 7}}', type=UpdateDocumentsResponse
        )

        assert decoded.matched_records is None
        assert decoded.usage == DocumentUpdateUsage(embed_total_tokens=7)

    def test_update_usage_alongside_matched_records(self) -> None:
        decoded = msgspec.json.decode(
            b'{"matched_records": 3, "usage": {"embed_total_tokens": 9}}',
            type=UpdateDocumentsResponse,
        )

        assert decoded.matched_records == 3
        assert decoded.usage == DocumentUpdateUsage(embed_total_tokens=9)

    def test_update_usage_absent_is_none(self) -> None:
        decoded = msgspec.json.decode(b"{}", type=UpdateDocumentsResponse)

        assert decoded.usage is None


class TestSyncClient:
    @respx.mock
    def test_upsert_with_usage(self, index: Index) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                202, json={"upserted_count": 1, "usage": {"embed_total_tokens": 5}}
            )
        )

        result = index.documents.upsert(namespace=NS, documents=DOCS)

        assert result.upserted_count == 1
        assert result.usage is not None
        assert result.usage.embed_total_tokens == 5

    @respx.mock
    def test_upsert_without_usage(self, index: Index) -> None:
        respx.post(UPSERT_URL).mock(return_value=httpx.Response(202, json={"upserted_count": 1}))

        result = index.documents.upsert(namespace=NS, documents=DOCS)

        assert result.usage is None

    @respx.mock
    def test_upsert_usage_missing_embed_total_tokens_rejected(self, index: Index) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(202, json={"upserted_count": 1, "usage": {}})
        )

        with pytest.raises(ResponseParsingError):
            index.documents.upsert(namespace=NS, documents=DOCS)

    @respx.mock
    def test_update_with_usage(self, index: Index) -> None:
        respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(202, json={"usage": {"embed_total_tokens": 4}})
        )

        result = index.documents.update(namespace=NS, documents=DOCS)

        assert result.usage is not None
        assert result.usage.embed_total_tokens == 4

    @respx.mock
    def test_update_without_usage(self, index: Index) -> None:
        respx.post(UPDATE_URL).mock(return_value=httpx.Response(202, json={}))

        result = index.documents.update(namespace=NS, documents=DOCS)

        assert result.usage is None


class TestAsyncClient:
    @respx.mock
    async def test_upsert_with_usage(self, async_index: AsyncIndex) -> None:
        respx.post(UPSERT_URL).mock(
            return_value=httpx.Response(
                202, json={"upserted_count": 1, "usage": {"embed_total_tokens": 5}}
            )
        )

        result = await async_index.documents.upsert(namespace=NS, documents=DOCS)

        assert result.usage == DocumentUpsertUsage(embed_total_tokens=5)

    @respx.mock
    async def test_upsert_without_usage(self, async_index: AsyncIndex) -> None:
        respx.post(UPSERT_URL).mock(return_value=httpx.Response(202, json={"upserted_count": 1}))

        result = await async_index.documents.upsert(namespace=NS, documents=DOCS)

        assert result.usage is None

    @respx.mock
    async def test_update_with_usage(self, async_index: AsyncIndex) -> None:
        respx.post(UPDATE_URL).mock(
            return_value=httpx.Response(202, json={"usage": {"embed_total_tokens": 4}})
        )

        result = await async_index.documents.update(namespace=NS, documents=DOCS)

        assert result.usage == DocumentUpdateUsage(embed_total_tokens=4)

    @respx.mock
    async def test_update_without_usage(self, async_index: AsyncIndex) -> None:
        respx.post(UPDATE_URL).mock(return_value=httpx.Response(202, json={}))

        result = await async_index.documents.update(namespace=NS, documents=DOCS)

        assert result.usage is None
