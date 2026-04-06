"""Unit tests for HTTPClient and _raise_for_status."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import respx

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER
from pinecone._internal.http_client import (
    AsyncHTTPClient,
    HTTPClient,
    _build_headers,
    _raise_for_status,
)
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    UnauthorizedError,
)

# ---------------------------------------------------------------------------
# _build_headers
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def test_includes_api_key(self) -> None:
        config = PineconeConfig(api_key="test-key")
        headers = _build_headers(config, "2025-10")
        assert headers["Api-Key"] == "test-key"

    def test_includes_api_version_header(self) -> None:
        config = PineconeConfig(api_key="k")
        headers = _build_headers(config, "2025-10")
        assert headers[API_VERSION_HEADER] == "2025-10"

    def test_includes_user_agent(self) -> None:
        config = PineconeConfig(api_key="k")
        headers = _build_headers(config, "2025-10")
        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("python-client-")

    def test_merges_additional_headers(self) -> None:
        config = PineconeConfig(api_key="k", additional_headers={"X-Custom": "value"})
        headers = _build_headers(config, "2025-10")
        assert headers["X-Custom"] == "value"

    def test_additional_headers_override_defaults(self) -> None:
        """User-supplied additional headers can override built-in headers."""
        config = PineconeConfig(api_key="k", additional_headers={"User-Agent": "custom-agent"})
        headers = _build_headers(config, "2025-10")
        assert headers["User-Agent"] == "custom-agent"

    def test_no_additional_headers(self) -> None:
        config = PineconeConfig(api_key="k")
        headers = _build_headers(config, "2025-10")
        # Should have exactly the three standard headers
        assert set(headers.keys()) == {"Api-Key", API_VERSION_HEADER, "User-Agent"}


# ---------------------------------------------------------------------------
# _raise_for_status
# ---------------------------------------------------------------------------


class TestRaiseForStatus:
    @pytest.mark.parametrize(
        "status_code,exception_type",
        [
            (401, UnauthorizedError),
            (404, NotFoundError),
            (409, ConflictError),
            (500, ApiError),
            (502, ApiError),
            (422, ApiError),
        ],
    )
    def test_status_code_to_exception(
        self, status_code: int, exception_type: type[ApiError]
    ) -> None:
        """Each status code maps to its specific exception type."""
        response = httpx.Response(status_code, json={"message": "oops"})
        with pytest.raises(exception_type) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.status_code == status_code

    def test_extracts_message_from_json_body(self) -> None:
        response = httpx.Response(400, json={"message": "bad input"})
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.message == "bad input"

    def test_fallback_message_when_body_not_json(self) -> None:
        response = httpx.Response(500, text="Internal Server Error")
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert "500" in exc_info.value.message

    def test_fallback_message_when_no_message_key(self) -> None:
        response = httpx.Response(500, json={"error": "something"})
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert "500" in exc_info.value.message

    def test_body_attribute_contains_parsed_json(self) -> None:
        body = {"message": "conflict", "details": "already exists"}
        response = httpx.Response(409, json=body)
        with pytest.raises(ConflictError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.body == body

    def test_body_is_none_when_not_json(self) -> None:
        response = httpx.Response(500, text="oops")
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        assert exc_info.value.body is None

    def test_success_response_does_not_raise(self) -> None:
        response = httpx.Response(200, json={"ok": True})
        _raise_for_status(response)  # Should not raise


# ---------------------------------------------------------------------------
# HTTPClient — sync
# ---------------------------------------------------------------------------

BASE_URL = "https://api.pinecone.io"


def _make_sync_client() -> HTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return HTTPClient(config, api_version="2025-10")


class TestHTTPClientGet:
    @respx.mock
    def test_get_success(self) -> None:
        respx.get(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(200, json={"indexes": []})
        )
        client = _make_sync_client()
        resp = client.get("/indexes")
        assert resp.status_code == 200
        assert resp.json() == {"indexes": []}

    @respx.mock
    def test_get_raises_on_error(self) -> None:
        respx.get(f"{BASE_URL}/indexes/missing").mock(
            return_value=httpx.Response(404, json={"message": "not found"})
        )
        client = _make_sync_client()
        with pytest.raises(NotFoundError):
            client.get("/indexes/missing")


class TestHTTPClientPost:
    @respx.mock
    def test_post_success(self) -> None:
        respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(201, json={"name": "my-index"})
        )
        client = _make_sync_client()
        resp = client.post("/indexes", json={"name": "my-index"})
        assert resp.status_code == 201

    @respx.mock
    def test_post_raises_on_conflict(self) -> None:
        respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(409, json={"message": "already exists"})
        )
        client = _make_sync_client()
        with pytest.raises(ConflictError):
            client.post("/indexes", json={"name": "dup"})


class TestHTTPClientDelete:
    @respx.mock
    def test_delete_success(self) -> None:
        respx.delete(f"{BASE_URL}/indexes/foo").mock(return_value=httpx.Response(202, json={}))
        client = _make_sync_client()
        resp = client.delete("/indexes/foo")
        assert resp.status_code == 202

    @respx.mock
    def test_delete_raises_on_unauthorized(self) -> None:
        respx.delete(f"{BASE_URL}/indexes/foo").mock(
            return_value=httpx.Response(401, json={"message": "bad key"})
        )
        client = _make_sync_client()
        with pytest.raises(UnauthorizedError):
            client.delete("/indexes/foo")


class TestHTTPClientClose:
    def test_close_closes_underlying_client(self) -> None:
        client = _make_sync_client()
        mock_inner = MagicMock()
        client._client = mock_inner
        client.close()
        mock_inner.close.assert_called_once()


# ---------------------------------------------------------------------------
# AsyncHTTPClient
# ---------------------------------------------------------------------------


def _make_async_client() -> AsyncHTTPClient:
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    return AsyncHTTPClient(config, api_version="2025-10")


class TestAsyncHTTPClientGet:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_success(self) -> None:
        respx.get(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(200, json={"indexes": []})
        )
        client = _make_async_client()
        try:
            resp = await client.get("/indexes")
            assert resp.status_code == 200
        finally:
            await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_raises_on_error(self) -> None:
        respx.get(f"{BASE_URL}/indexes/missing").mock(
            return_value=httpx.Response(404, json={"message": "not found"})
        )
        client = _make_async_client()
        try:
            with pytest.raises(NotFoundError):
                await client.get("/indexes/missing")
        finally:
            await client.close()


class TestAsyncHTTPClientPost:
    @respx.mock
    @pytest.mark.asyncio
    async def test_post_raises_on_conflict(self) -> None:
        respx.post(f"{BASE_URL}/indexes").mock(
            return_value=httpx.Response(409, json={"message": "conflict"})
        )
        client = _make_async_client()
        try:
            with pytest.raises(ConflictError):
                await client.post("/indexes", json={"name": "dup"})
        finally:
            await client.close()


class TestAsyncHTTPClientClose:
    @pytest.mark.asyncio
    async def test_close_when_no_client_created(self) -> None:
        """close() should not error when no requests were made."""
        client = _make_async_client()
        await client.close()  # _client is None — should be a no-op

    @respx.mock
    @pytest.mark.asyncio
    async def test_close_closes_underlying_client(self) -> None:
        respx.get(f"{BASE_URL}/ping").mock(return_value=httpx.Response(200))
        client = _make_async_client()
        await client.get("/ping")  # Force client creation
        assert client._client is not None
        await client.close()
