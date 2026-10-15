"""A 413 response raises PayloadTooLargeError on every REST path (#701)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import pytest_asyncio
import respx

import pinecone
import pinecone.errors
from pinecone import ApiError, PayloadTooLargeError, PineconeError
from pinecone._internal.http_client import _raise_for_status
from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index

HOST = "payload-too-large-abc123.svc.us-east-1-aws.pinecone.io"
UPSERT_URL = f"https://{HOST}/namespaces/n/documents/upsert"
DOCS = [{"_id": "a"}, {"_id": "b"}]

JSON_413 = httpx.Response(
    413,
    json={
        "error": {
            "code": "PAYLOAD_TOO_LARGE",
            "message": "Request body is too large. Send fewer items per request.",
        }
    },
)
PLAIN_413 = httpx.Response(413, text="Payload Too Large")


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=HOST, api_key="test-key")
    yield client
    client.close()


@pytest_asyncio.fixture
async def aindex() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=HOST, api_key="test-key")
    yield client
    await client.close()


class TestClass:
    def test_is_an_api_error_so_existing_handlers_still_catch_it(self) -> None:
        assert issubclass(PayloadTooLargeError, ApiError)
        assert issubclass(PayloadTooLargeError, PineconeError)

    def test_defaults_to_413(self) -> None:
        assert PayloadTooLargeError().status_code == 413

    def test_exported_from_both_import_paths(self) -> None:
        assert pinecone.PayloadTooLargeError is pinecone.errors.PayloadTooLargeError
        assert "PayloadTooLargeError" in pinecone.__all__
        assert "PayloadTooLargeError" in pinecone.errors.__all__


class TestRaiseForStatus:
    def test_json_body(self) -> None:
        with pytest.raises(PayloadTooLargeError) as exc_info:
            _raise_for_status(JSON_413)
        assert exc_info.value.status_code == 413
        assert exc_info.value.error_code == "PAYLOAD_TOO_LARGE"

    def test_plain_text_body_without_error_code(self) -> None:
        with pytest.raises(PayloadTooLargeError) as exc_info:
            _raise_for_status(PLAIN_413)
        assert exc_info.value.status_code == 413
        assert exc_info.value.error_code is None
        assert exc_info.value.body is None
        assert exc_info.value.message == "Payload Too Large"

    def test_json_body_with_unexpected_shape(self) -> None:
        with pytest.raises(PayloadTooLargeError):
            _raise_for_status(httpx.Response(413, json={"unexpected": True}))

    def test_other_4xx_stay_plain(self) -> None:
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(httpx.Response(414, text="URI Too Long"))
        assert type(exc_info.value) is ApiError


class TestSyncDocuments:
    @pytest.mark.parametrize("response", [JSON_413, PLAIN_413], ids=["json", "plain-text"])
    @respx.mock
    def test_upsert_raises(self, index: Index, response: httpx.Response) -> None:
        respx.post(UPSERT_URL).mock(return_value=response)
        with pytest.raises(PayloadTooLargeError):
            index.documents.upsert(namespace="n", documents=DOCS)

    @pytest.mark.parametrize("response", [JSON_413, PLAIN_413], ids=["json", "plain-text"])
    @respx.mock
    def test_batch_upsert_records_it_in_the_result(
        self, index: Index, response: httpx.Response
    ) -> None:
        respx.post(UPSERT_URL).mock(return_value=response)
        result = index.documents.batch_upsert(
            namespace="n", documents=DOCS, batch_size=2, show_progress=False
        )
        assert len(result.errors) == 1
        assert isinstance(result.errors[0].error, PayloadTooLargeError)
        assert result.errors[0].retryable is False


class TestAsyncDocuments:
    @pytest.mark.parametrize("response", [JSON_413, PLAIN_413], ids=["json", "plain-text"])
    @respx.mock
    @pytest.mark.asyncio
    async def test_upsert_raises(self, aindex: AsyncIndex, response: httpx.Response) -> None:
        respx.post(UPSERT_URL).mock(return_value=response)
        with pytest.raises(PayloadTooLargeError):
            await aindex.documents.upsert(namespace="n", documents=DOCS)

    @pytest.mark.parametrize("response", [JSON_413, PLAIN_413], ids=["json", "plain-text"])
    @respx.mock
    @pytest.mark.asyncio
    async def test_batch_upsert_records_it_in_the_result(
        self, aindex: AsyncIndex, response: httpx.Response
    ) -> None:
        respx.post(UPSERT_URL).mock(return_value=response)
        result = await aindex.documents.batch_upsert(
            namespace="n", documents=DOCS, batch_size=2, show_progress=False
        )
        assert len(result.errors) == 1
        assert isinstance(result.errors[0].error, PayloadTooLargeError)
        assert result.errors[0].retryable is False
