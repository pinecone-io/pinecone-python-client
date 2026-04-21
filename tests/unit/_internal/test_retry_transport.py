"""Tests for _RetryTransport retry behavior including connection errors."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from pinecone._internal.config import RetryConfig
from pinecone._internal.http_client import _AsyncRetryTransport, _RetryTransport


def _transport(max_retries: int = 3) -> tuple[_RetryTransport, MagicMock]:
    inner = MagicMock(spec=httpx.BaseTransport)
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.001, max_wait=0.01)
    return _RetryTransport(transport=inner, retry_config=cfg), inner  # type: ignore[arg-type]


def _req() -> httpx.Request:
    return httpx.Request("POST", "https://example.com/test")


def test_connection_error_is_retried_and_succeeds() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.Response(200),
    ]
    result = rt.handle_request(_req())
    assert result.status_code == 200
    assert inner.handle_request.call_count == 2


def test_connection_error_exhausts_retries_and_raises() -> None:
    rt, inner = _transport(max_retries=2)
    inner.handle_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.RemoteProtocolError("peer closed connection"),
    ]
    with pytest.raises(httpx.RemoteProtocolError):
        rt.handle_request(_req())


def test_retryable_status_code_still_retried() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.side_effect = [httpx.Response(503), httpx.Response(200)]
    result = rt.handle_request(_req())
    assert result.status_code == 200


def test_non_retryable_status_returns_immediately() -> None:
    rt, inner = _transport(max_retries=3)
    inner.handle_request.return_value = httpx.Response(400)
    result = rt.handle_request(_req())
    assert result.status_code == 400
    assert inner.handle_request.call_count == 1


# --- async variants ---


def _async_transport(max_retries: int = 3) -> tuple[_AsyncRetryTransport, AsyncMock]:
    inner = AsyncMock(spec=httpx.AsyncBaseTransport)
    cfg = RetryConfig(max_retries=max_retries, backoff_factor=0.001, max_wait=0.01)
    return _AsyncRetryTransport(transport=inner, retry_config=cfg), inner  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_connection_error_is_retried_and_succeeds() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.Response(200),
    ]
    result = await rt.handle_async_request(_req())
    assert result.status_code == 200
    assert inner.handle_async_request.call_count == 2


@pytest.mark.asyncio
async def test_async_connection_error_exhausts_retries_and_raises() -> None:
    rt, inner = _async_transport(max_retries=2)
    inner.handle_async_request.side_effect = [
        httpx.RemoteProtocolError("peer closed connection"),
        httpx.RemoteProtocolError("peer closed connection"),
    ]
    with pytest.raises(httpx.RemoteProtocolError):
        await rt.handle_async_request(_req())


@pytest.mark.asyncio
async def test_async_retryable_status_code_still_retried() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.side_effect = [httpx.Response(503), httpx.Response(200)]
    result = await rt.handle_async_request(_req())
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_async_non_retryable_status_returns_immediately() -> None:
    rt, inner = _async_transport(max_retries=3)
    inner.handle_async_request.return_value = httpx.Response(400)
    result = await rt.handle_async_request(_req())
    assert result.status_code == 400
    assert inner.handle_async_request.call_count == 1
