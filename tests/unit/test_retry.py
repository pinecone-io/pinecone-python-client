"""Unit tests for retry transport wrappers."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import (
    HTTPClient,
    _AsyncRetryTransport,
    _RetryTransport,
)
from pinecone.errors.exceptions import ApiError

BASE_URL = "https://api.pinecone.io"


# ---------------------------------------------------------------------------
# Helpers — fake transports that return a sequence of canned responses
# ---------------------------------------------------------------------------


class _FakeTransport(httpx.BaseTransport):
    """Sync transport that returns responses from a pre-configured list."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def close(self) -> None:
        pass

    @property
    def call_count(self) -> int:
        return self._call_count


class _FakeAsyncTransport(httpx.AsyncBaseTransport):
    """Async transport that returns responses from a pre-configured list."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    async def aclose(self) -> None:
        pass

    @property
    def call_count(self) -> int:
        return self._call_count


def _make_request() -> httpx.Request:
    return httpx.Request("GET", "https://api.pinecone.io/indexes")


# ---------------------------------------------------------------------------
# Sync _RetryTransport tests
# ---------------------------------------------------------------------------


class TestSyncRetryTransport:
    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_retries_on_500(self, mock_sleep: Any) -> None:
        fake = _FakeTransport([
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(200, json={"ok": True}),
        ])
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=5,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_retries_on_502(self, mock_sleep: Any) -> None:
        fake = _FakeTransport([
            httpx.Response(502, json={"message": "bad gateway"}),
            httpx.Response(200, json={"ok": True}),
        ])
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=5,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_no_retry_on_429(self, mock_sleep: Any) -> None:
        fake = _FakeTransport([
            httpx.Response(429, json={"message": "rate limited"}),
        ])
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=5,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 429
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_no_retry_on_400(self, mock_sleep: Any) -> None:
        fake = _FakeTransport([
            httpx.Response(400, json={"message": "bad request"}),
        ])
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=5,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 400
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_exhausts_retries(self, mock_sleep: Any) -> None:
        fake = _FakeTransport([
            httpx.Response(503, json={"message": "unavailable"}),
        ])
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=5,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        response = transport.handle_request(_make_request())
        # 1 initial + 5 retries = 6 total attempts
        assert response.status_code == 503
        assert fake.call_count == 6
        assert mock_sleep.call_count == 5

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_retries_all_methods(self, mock_sleep: Any) -> None:
        """POST, DELETE, and PATCH are all retried, not just GET."""
        for method in ("POST", "DELETE", "PATCH"):
            fake = _FakeTransport([
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ])
            transport = _RetryTransport(
                transport=fake,  # type: ignore[arg-type]
                max_retries=5,
                backoff_factor=0.25,
                jitter_max=0.0,
            )
            request = httpx.Request(method, "https://api.pinecone.io/test")
            response = transport.handle_request(request)
            assert response.status_code == 200, f"{method} should be retried"
            assert fake.call_count == 2, f"{method} should have 2 attempts"

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_backoff_delays(self, mock_sleep: Any) -> None:
        """Verify exponential backoff timing with zero jitter."""
        fake = _FakeTransport([
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(200, json={"ok": True}),
        ])
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=5,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        transport.handle_request(_make_request())
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [0.25, 0.5, 1.0]  # 0.25 * 2^0, 0.25 * 2^1, 0.25 * 2^2


# ---------------------------------------------------------------------------
# Async _AsyncRetryTransport tests
# ---------------------------------------------------------------------------


class TestAsyncRetryTransport:
    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_retries_on_500(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport([
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(200, json={"ok": True}),
        ])
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_attempts=5,
            initial_backoff=0.1,
            max_backoff=3.0,
            jitter_max=0.0,
        )
        response = await transport.handle_async_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 3

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_retries_on_503(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport([
            httpx.Response(503, json={"message": "unavailable"}),
            httpx.Response(503, json={"message": "unavailable"}),
            httpx.Response(200, json={"ok": True}),
        ])
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_attempts=5,
            initial_backoff=0.1,
            max_backoff=3.0,
            jitter_max=0.0,
        )
        response = await transport.handle_async_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 3

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_no_retry_on_429(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport([
            httpx.Response(429, json={"message": "rate limited"}),
        ])
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_attempts=5,
            initial_backoff=0.1,
            max_backoff=3.0,
            jitter_max=0.0,
        )
        response = await transport.handle_async_request(_make_request())
        assert response.status_code == 429
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_exhausts_retries(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport([
            httpx.Response(502, json={"message": "bad gateway"}),
        ])
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_attempts=5,
            initial_backoff=0.1,
            max_backoff=3.0,
            jitter_max=0.0,
        )
        response = await transport.handle_async_request(_make_request())
        # max_attempts=5 total, so 5 calls
        assert response.status_code == 502
        assert fake.call_count == 5
        assert mock_sleep.call_count == 4

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_backoff_capped(self, mock_sleep: Any) -> None:
        """Verify async backoff is capped at max_backoff."""
        fake = _FakeAsyncTransport([
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
        ])
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_attempts=5,
            initial_backoff=0.1,
            max_backoff=0.5,
            jitter_max=0.0,
        )
        await transport.handle_async_request(_make_request())
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 0.1*2^0=0.1, 0.1*2^1=0.2, 0.1*2^2=0.4, 0.1*2^3=0.8 -> capped to 0.5
        assert delays == [0.1, 0.2, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Integration: HTTPClient uses retry transport
# ---------------------------------------------------------------------------


class TestHTTPClientRetryIntegration:
    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_client_retries_500_then_raises(self, mock_sleep: Any) -> None:
        """HTTPClient wraps transport with _RetryTransport; 500s are retried then raised."""
        config = PineconeConfig(api_key="test-key", host=BASE_URL)
        client = HTTPClient(config, api_version="2025-10")

        # Replace the inner transport with our fake
        fake = _FakeTransport([
            httpx.Response(500, json={"message": "server error"}),
        ])
        retry_transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=2,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        client._client._transport = retry_transport  # type: ignore[assignment]

        with pytest.raises(ApiError) as exc_info:
            client.get("/indexes")
        assert exc_info.value.status_code == 500
        # 1 initial + 2 retries = 3 total
        assert fake.call_count == 3

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_client_retries_then_succeeds(self, mock_sleep: Any) -> None:
        config = PineconeConfig(api_key="test-key", host=BASE_URL)
        client = HTTPClient(config, api_version="2025-10")

        fake = _FakeTransport([
            httpx.Response(502, json={"message": "bad gateway"}),
            httpx.Response(200, json={"indexes": []}),
        ])
        retry_transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            max_retries=5,
            backoff_factor=0.25,
            jitter_max=0.0,
        )
        client._client._transport = retry_transport  # type: ignore[assignment]

        resp = client.get("/indexes")
        assert resp.status_code == 200
        assert fake.call_count == 2
