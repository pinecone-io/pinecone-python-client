"""Tests for HTTPClient.post fast-path retry behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.http_client import HTTPClient


def _make_client(max_retries: int = 2) -> HTTPClient:
    retry_config = RetryConfig(max_retries=max_retries, backoff_factor=0.0, max_wait=0.0)
    config = PineconeConfig(
        api_key="test-key-1234",
        host="https://test.example.com",
        retry_config=retry_config,
    )
    return HTTPClient(config, "2025-10")


def test_fast_path_post_retried_on_connect_error() -> None:
    """Fast-path POST bypasses Client.send() but still goes through _RetryTransport."""
    client = _make_client(max_retries=2)

    mock_inner = MagicMock(spec=httpx.BaseTransport)
    mock_inner.handle_request.side_effect = [
        httpx.ConnectError("connection refused"),
        httpx.Response(200, content=b"{}"),
    ]
    # Swap the inner transport inside _RetryTransport so retry logic wraps the mock.
    client._client._transport._transport = mock_inner  # type: ignore[attr-defined]

    result = client.post("/some/path", json={"foo": "bar"})
    assert result.status_code == 200
    assert mock_inner.handle_request.call_count == 2


def test_fast_path_post_succeeds_on_first_try() -> None:
    client = _make_client()

    mock_inner = MagicMock(spec=httpx.BaseTransport)
    mock_inner.handle_request.return_value = httpx.Response(200, content=b"{}")
    client._client._transport._transport = mock_inner  # type: ignore[attr-defined]

    result = client.post("/vectors/upsert", json={"vectors": []})
    assert result.status_code == 200
    assert mock_inner.handle_request.call_count == 1


@pytest.mark.asyncio
async def test_fast_path_uses_retry_transport_not_client_send() -> None:
    """Regression guard: if _RetryTransport is bypassed, retry count would be 1."""
    client = _make_client(max_retries=3)

    mock_inner = MagicMock(spec=httpx.BaseTransport)
    mock_inner.handle_request.side_effect = [
        httpx.ConnectError("reset by peer"),
        httpx.ConnectError("reset by peer"),
        httpx.Response(200, content=b"{}"),
    ]
    client._client._transport._transport = mock_inner  # type: ignore[attr-defined]

    result = client.post("/query", json={"vector": [0.1, 0.2]})
    assert result.status_code == 200
    assert mock_inner.handle_request.call_count == 3
