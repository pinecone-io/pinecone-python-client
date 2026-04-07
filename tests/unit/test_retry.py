"""Unit tests for retry transport wrappers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pinecone._internal.config import PineconeConfig, RetryConfig
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


def _make_response(status_code: int, **kwargs: Any) -> httpx.Response:
    """Create an httpx.Response with a mock close/aclose for tracking."""
    resp = httpx.Response(status_code, **kwargs)
    resp.close = MagicMock()  # type: ignore[assignment]
    resp.aclose = AsyncMock()  # type: ignore[assignment]
    return resp


class _TrackingTransport(httpx.BaseTransport):
    """Sync transport that returns trackable responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.returned_responses: list[httpx.Response] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        resp = self._responses[idx]
        self.returned_responses.append(resp)
        return resp

    def close(self) -> None:
        pass

    @property
    def call_count(self) -> int:
        return self._call_count


class _TrackingAsyncTransport(httpx.AsyncBaseTransport):
    """Async transport that returns trackable responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.returned_responses: list[httpx.Response] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        resp = self._responses[idx]
        self.returned_responses.append(resp)
        return resp

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
        fake = _FakeTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_retries_on_502(self, mock_sleep: Any) -> None:
        fake = _FakeTransport(
            [
                httpx.Response(502, json={"message": "bad gateway"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_no_retry_on_429(self, mock_sleep: Any) -> None:
        fake = _FakeTransport(
            [
                httpx.Response(429, json={"message": "rate limited"}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 429
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_no_retry_on_400(self, mock_sleep: Any) -> None:
        fake = _FakeTransport(
            [
                httpx.Response(400, json={"message": "bad request"}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = transport.handle_request(_make_request())
        assert response.status_code == 400
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_exhausts_retries(self, mock_sleep: Any) -> None:
        fake = _FakeTransport(
            [
                httpx.Response(503, json={"message": "unavailable"}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = transport.handle_request(_make_request())
        # max_attempts=5 total, so 5 calls
        assert response.status_code == 503
        assert fake.call_count == 5
        assert mock_sleep.call_count == 4

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_no_retry_for_post(self, mock_sleep: Any) -> None:
        """POST requests are NOT retried even on retryable status codes."""
        fake = _FakeTransport(
            [
                httpx.Response(503, json={"message": "unavailable"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        request = httpx.Request("POST", "https://api.pinecone.io/test")
        response = transport.handle_request(request)
        assert response.status_code == 503
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_no_retry_for_delete(self, mock_sleep: Any) -> None:
        """DELETE requests are NOT retried even on retryable status codes."""
        fake = _FakeTransport(
            [
                httpx.Response(503, json={"message": "unavailable"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        request = httpx.Request("DELETE", "https://api.pinecone.io/test")
        response = transport.handle_request(request)
        assert response.status_code == 503
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_no_retry_for_non_idempotent_methods(self, mock_sleep: Any) -> None:
        """PUT and PATCH requests are NOT retried."""
        for method in ("PUT", "PATCH"):
            fake = _FakeTransport(
                [
                    httpx.Response(500, json={"message": "error"}),
                    httpx.Response(200, json={"ok": True}),
                ]
            )
            transport = _RetryTransport(
                transport=fake,  # type: ignore[arg-type]
                retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
            )
            request = httpx.Request(method, "https://api.pinecone.io/test")
            response = transport.handle_request(request)
            assert response.status_code == 500, f"{method} should NOT be retried"
            assert fake.call_count == 1, f"{method} should have only 1 attempt"

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_retries_head(self, mock_sleep: Any) -> None:
        """HEAD requests are retried like GET."""
        fake = _FakeTransport(
            [
                httpx.Response(503, json={"message": "unavailable"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        request = httpx.Request("HEAD", "https://api.pinecone.io/test")
        response = transport.handle_request(request)
        assert response.status_code == 200
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_backoff_delays(self, mock_sleep: Any) -> None:
        """Verify exponential backoff timing with zero jitter."""
        fake = _FakeTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        transport.handle_request(_make_request())
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [0.1, 0.2, 0.4]  # 0.1 * 2^0, 0.1 * 2^1, 0.1 * 2^2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_backoff_capped(self, mock_sleep: Any) -> None:
        """Verify sync backoff is capped at max_backoff, matching async behavior."""
        fake = _FakeTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(
                max_attempts=5,
                initial_backoff=0.1,
                max_backoff=0.5,
                jitter_max=0.0,
            ),
        )
        transport.handle_request(_make_request())
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 0.1*2^0=0.1, 0.1*2^1=0.2, 0.1*2^2=0.4, 0.1*2^3=0.8 -> capped to 0.5
        assert delays == [0.1, 0.2, 0.4, 0.5]

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_closes_discarded_responses(self, mock_sleep: Any) -> None:
        """Discarded responses must be closed to release connections back to the pool."""
        responses = [
            _make_response(500, json={"message": "error"}),
            _make_response(503, json={"message": "unavailable"}),
            _make_response(200, json={"ok": True}),
        ]
        fake = _TrackingTransport(responses)
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        result = transport.handle_request(_make_request())
        assert result.status_code == 200
        # The first two (retried) responses should have been closed
        responses[0].close.assert_called_once()  # type: ignore[union-attr]
        responses[1].close.assert_called_once()  # type: ignore[union-attr]
        # The final (returned) response should NOT be closed
        responses[2].close.assert_not_called()  # type: ignore[union-attr]

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_closes_all_on_exhaustion(self, mock_sleep: Any) -> None:
        """When retries are exhausted, all intermediate responses are closed."""
        responses = [
            _make_response(500, json={"message": "error"}),
            _make_response(500, json={"message": "error"}),
            _make_response(500, json={"message": "error"}),
        ]
        fake = _TrackingTransport(responses)
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=3, initial_backoff=0.1, jitter_max=0.0),
        )
        result = transport.handle_request(_make_request())
        assert result.status_code == 500
        assert fake.call_count == 3
        # First two discarded responses closed; last one returned as-is
        responses[0].close.assert_called_once()  # type: ignore[union-attr]
        responses[1].close.assert_called_once()  # type: ignore[union-attr]
        responses[2].close.assert_not_called()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Async _AsyncRetryTransport tests
# ---------------------------------------------------------------------------


class TestAsyncRetryTransport:
    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_retries_on_500(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = await transport.handle_async_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 3

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_retries_on_503(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport(
            [
                httpx.Response(503, json={"message": "unavailable"}),
                httpx.Response(503, json={"message": "unavailable"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = await transport.handle_async_request(_make_request())
        assert response.status_code == 200
        assert fake.call_count == 3

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_no_retry_on_429(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport(
            [
                httpx.Response(429, json={"message": "rate limited"}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = await transport.handle_async_request(_make_request())
        assert response.status_code == 429
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_exhausts_retries(self, mock_sleep: Any) -> None:
        fake = _FakeAsyncTransport(
            [
                httpx.Response(502, json={"message": "bad gateway"}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        response = await transport.handle_async_request(_make_request())
        # max_attempts=5 total, so 5 calls
        assert response.status_code == 502
        assert fake.call_count == 5
        assert mock_sleep.call_count == 4

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_no_retry_for_post(self, mock_sleep: Any) -> None:
        """POST requests are NOT retried even on retryable status codes."""
        fake = _FakeAsyncTransport(
            [
                httpx.Response(503, json={"message": "unavailable"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        request = httpx.Request("POST", "https://api.pinecone.io/test")
        response = await transport.handle_async_request(request)
        assert response.status_code == 503
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_no_retry_for_delete(self, mock_sleep: Any) -> None:
        """DELETE requests are NOT retried even on retryable status codes."""
        fake = _FakeAsyncTransport(
            [
                httpx.Response(503, json={"message": "unavailable"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        request = httpx.Request("DELETE", "https://api.pinecone.io/test")
        response = await transport.handle_async_request(request)
        assert response.status_code == 503
        assert fake.call_count == 1
        mock_sleep.assert_not_called()

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_backoff_capped(self, mock_sleep: Any) -> None:
        """Verify async backoff is capped at max_backoff."""
        fake = _FakeAsyncTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(
                max_attempts=5,
                initial_backoff=0.1,
                max_backoff=0.5,
                jitter_max=0.0,
            ),
        )
        await transport.handle_async_request(_make_request())
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 0.1*2^0=0.1, 0.1*2^1=0.2, 0.1*2^2=0.4, 0.1*2^3=0.8 -> capped to 0.5
        assert delays == [0.1, 0.2, 0.4, 0.5]

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_closes_discarded_responses(self, mock_sleep: Any) -> None:
        """Discarded responses must be aclosed to release connections back to the pool."""
        responses = [
            _make_response(500, json={"message": "error"}),
            _make_response(502, json={"message": "bad gateway"}),
            _make_response(200, json={"ok": True}),
        ]
        fake = _TrackingAsyncTransport(responses)
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        result = await transport.handle_async_request(_make_request())
        assert result.status_code == 200
        # The first two (retried) responses should have been aclosed
        responses[0].aclose.assert_awaited_once()  # type: ignore[union-attr]
        responses[1].aclose.assert_awaited_once()  # type: ignore[union-attr]
        # The final (returned) response should NOT be aclosed
        responses[2].aclose.assert_not_awaited()  # type: ignore[union-attr]

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_async_closes_all_on_exhaustion(self, mock_sleep: Any) -> None:
        """When retries are exhausted, all intermediate responses are aclosed."""
        responses = [
            _make_response(503, json={"message": "unavailable"}),
            _make_response(503, json={"message": "unavailable"}),
            _make_response(503, json={"message": "unavailable"}),
        ]
        fake = _TrackingAsyncTransport(responses)
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=3, initial_backoff=0.1, jitter_max=0.0),
        )
        result = await transport.handle_async_request(_make_request())
        assert result.status_code == 503
        assert fake.call_count == 3
        # First two discarded responses aclosed; last one returned as-is
        responses[0].aclose.assert_awaited_once()  # type: ignore[union-attr]
        responses[1].aclose.assert_awaited_once()  # type: ignore[union-attr]
        responses[2].aclose.assert_not_awaited()  # type: ignore[union-attr]


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
        fake = _FakeTransport(
            [
                httpx.Response(500, json={"message": "server error"}),
            ]
        )
        retry_transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=3, initial_backoff=0.1, jitter_max=0.0),
        )
        client._client._transport = retry_transport  # type: ignore[assignment]

        with pytest.raises(ApiError) as exc_info:
            client.get("/indexes")
        assert exc_info.value.status_code == 500
        # max_attempts=3 total
        assert fake.call_count == 3

    @patch("pinecone._internal.http_client.time.sleep")
    def test_sync_client_retries_then_succeeds(self, mock_sleep: Any) -> None:
        config = PineconeConfig(api_key="test-key", host=BASE_URL)
        client = HTTPClient(config, api_version="2025-10")

        fake = _FakeTransport(
            [
                httpx.Response(502, json={"message": "bad gateway"}),
                httpx.Response(200, json={"indexes": []}),
            ]
        )
        retry_transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=5, initial_backoff=0.1, jitter_max=0.0),
        )
        client._client._transport = retry_transport  # type: ignore[assignment]

        resp = client.get("/indexes")
        assert resp.status_code == 200
        assert fake.call_count == 2
