"""Unit tests for RetryConfig injection into retry transports."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.http_client import (
    _AsyncRetryTransport,
    _RetryTransport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTransport(httpx.BaseTransport):
    """Sync transport returning canned responses."""

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
    """Async transport returning canned responses."""

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


def _get_request() -> httpx.Request:
    return httpx.Request("GET", "https://api.pinecone.io/indexes")


# ---------------------------------------------------------------------------
# RetryConfig defaults
# ---------------------------------------------------------------------------


class TestRetryConfigDefaults:
    def test_default_retry_config(self) -> None:
        cfg = RetryConfig()
        assert cfg.max_attempts == 5
        assert cfg.initial_backoff == 0.1
        assert cfg.max_backoff == 3.0
        assert cfg.jitter_max == 0.1
        assert cfg.retryable_status_codes == frozenset({500, 502, 503, 504})
        assert cfg.retryable_methods == frozenset({"GET", "HEAD"})

    def test_retry_config_is_frozen(self) -> None:
        cfg = RetryConfig()
        with pytest.raises(AttributeError):
            cfg.max_attempts = 10  # type: ignore[misc]

    def test_pinecone_config_default_retry_config(self) -> None:
        pc_cfg = PineconeConfig(api_key="test-key")
        assert pc_cfg.retry_config == RetryConfig()

    def test_pinecone_config_custom_retry_config(self) -> None:
        custom = RetryConfig(max_attempts=2, initial_backoff=0.5)
        pc_cfg = PineconeConfig(api_key="test-key", retry_config=custom)
        assert pc_cfg.retry_config.max_attempts == 2
        assert pc_cfg.retry_config.initial_backoff == 0.5


# ---------------------------------------------------------------------------
# Custom RetryConfig — sync transport
# ---------------------------------------------------------------------------


class TestCustomRetryConfigSync:
    @patch("pinecone._internal.http_client.time.sleep")
    def test_custom_retry_config_max_attempts(self, mock_sleep: Any) -> None:
        """Custom RetryConfig with max_attempts=2 only retries once."""
        fake = _FakeTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=2, jitter_max=0.0),
        )
        response = transport.handle_request(_get_request())
        assert response.status_code == 500
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_custom_retry_config_status_codes(self, mock_sleep: Any) -> None:
        """Custom retryable_status_codes enables retry on non-default codes."""
        fake = _FakeTransport(
            [
                httpx.Response(429, json={"message": "rate limited"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(
                retryable_status_codes=frozenset({429, 500}),
                jitter_max=0.0,
            ),
        )
        response = transport.handle_request(_get_request())
        assert response.status_code == 200
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_custom_retry_config_methods(self, mock_sleep: Any) -> None:
        """Custom retryable_methods allows POST to be retried."""
        fake = _FakeTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(
                retryable_methods=frozenset({"GET", "HEAD", "POST"}),
                jitter_max=0.0,
            ),
        )
        request = httpx.Request("POST", "https://api.pinecone.io/test")
        response = transport.handle_request(request)
        assert response.status_code == 200
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.time.sleep")
    def test_default_config_used_when_none(self, mock_sleep: Any) -> None:
        """When retry_config=None, default RetryConfig is used."""
        fake = _FakeTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _RetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=None,
        )
        response = transport.handle_request(_get_request())
        assert response.status_code == 200
        assert fake.call_count == 2


# ---------------------------------------------------------------------------
# Custom RetryConfig — async transport
# ---------------------------------------------------------------------------


class TestCustomRetryConfigAsync:
    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_custom_retry_config_max_attempts(self, mock_sleep: Any) -> None:
        """Custom RetryConfig with max_attempts=2 only retries once."""
        fake = _FakeAsyncTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(max_attempts=2, jitter_max=0.0),
        )
        response = await transport.handle_async_request(_get_request())
        assert response.status_code == 500
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_custom_retry_config_status_codes(self, mock_sleep: Any) -> None:
        """Custom retryable_status_codes enables retry on non-default codes."""
        fake = _FakeAsyncTransport(
            [
                httpx.Response(429, json={"message": "rate limited"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(
                retryable_status_codes=frozenset({429, 500}),
                jitter_max=0.0,
            ),
        )
        response = await transport.handle_async_request(_get_request())
        assert response.status_code == 200
        assert fake.call_count == 2

    @patch("pinecone._internal.http_client.asyncio.sleep")
    @pytest.mark.asyncio
    async def test_custom_retry_config_methods(self, mock_sleep: Any) -> None:
        """Custom retryable_methods allows POST to be retried."""
        fake = _FakeAsyncTransport(
            [
                httpx.Response(500, json={"message": "error"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        transport = _AsyncRetryTransport(
            transport=fake,  # type: ignore[arg-type]
            retry_config=RetryConfig(
                retryable_methods=frozenset({"GET", "HEAD", "POST"}),
                jitter_max=0.0,
            ),
        )
        request = httpx.Request("POST", "https://api.pinecone.io/test")
        response = await transport.handle_async_request(request)
        assert response.status_code == 200
        assert fake.call_count == 2
