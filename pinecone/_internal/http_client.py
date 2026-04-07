"""httpx-based HTTP client for sync and async operations."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import socket
import sys
import time
from typing import Any

import httpx
import orjson

from pinecone import __version__
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER, DEFAULT_BASE_URL
from pinecone._internal.user_agent import build_user_agent
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeTimeoutError,
    ServiceError,
    UnauthorizedError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})
_RETRYABLE_METHODS: frozenset[str] = frozenset({"GET", "HEAD"})


def _build_socket_options() -> list[tuple[int, int, int]]:
    """Build platform-specific TCP socket options.

    Enables TCP keep-alive and disables Nagle's algorithm on all platforms.
    Adds platform-specific keep-alive tuning on Linux and macOS.
    """
    opts: list[tuple[int, int, int]] = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
    ]
    if sys.platform == "linux":
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 300))
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60))
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4))
    elif sys.platform == "darwin":
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60))
    return opts


def _default_pool_size() -> int:
    """Return the default connection pool size: 5x CPU count with a floor of 20."""
    return max(5 * (os.cpu_count() or 1), 20)


def _encode_json(body: Any) -> bytes:
    """Serialize *body* to JSON bytes using orjson (2-3x faster than stdlib json)."""
    return orjson.dumps(body)


def _prepare_json_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """If *kwargs* contains ``json=``, replace it with ``content=`` + Content-Type header.

    This bypasses httpx's default stdlib ``json.dumps`` in favour of orjson.
    """
    if "json" in kwargs:
        data = kwargs.pop("json")
        kwargs["content"] = _encode_json(data)
        headers: dict[str, str] = kwargs.pop("headers", {})
        headers["Content-Type"] = "application/json"
        kwargs["headers"] = headers
    return kwargs


def _build_headers(config: PineconeConfig, api_version: str) -> dict[str, str]:
    headers: dict[str, str] = {
        API_VERSION_HEADER: api_version,
        "User-Agent": build_user_agent(__version__, config.source_tag or None),
    }
    if config.api_key:
        headers["Api-Key"] = config.api_key
    if config.additional_headers:
        headers.update(config.additional_headers)
    return headers


def _log_curl(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None = None,
) -> None:
    """Log a curl-equivalent command for debugging when PINECONE_DEBUG_CURL is set."""
    if not os.environ.get("PINECONE_DEBUG_CURL"):
        return
    parts = [f"curl -X {method} '{url}'"]
    for key, value in headers.items():
        parts.append(f"-H '{key}: {value}'")
    if body is not None:
        parts.append(f"-d '{body.decode('utf-8', errors='replace')}'")
    curl_cmd = " ".join(parts)
    logger.debug("curl equivalent: %s", curl_cmd)
    print(curl_cmd)  # noqa: T201


class _RetryTransport(httpx.BaseTransport):
    """Sync transport wrapper that retries on transient server errors."""

    def __init__(
        self,
        *,
        transport: httpx.HTTPTransport,
        max_attempts: int = 5,
        initial_backoff: float = 0.1,
        max_backoff: float = 3.0,
        jitter_max: float = 0.1,
    ) -> None:
        self._transport = transport
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._jitter_max = jitter_max

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._transport.handle_request(request)
        if request.method not in _RETRYABLE_METHODS:
            return response
        for attempt in range(self._max_attempts - 1):
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                return response
            response.close()
            delay = min(self._initial_backoff * (2**attempt), self._max_backoff) + random.uniform(
                0, self._jitter_max
            )
            time.sleep(delay)
            response = self._transport.handle_request(request)
        return response

    def close(self) -> None:
        self._transport.close()


class _AsyncRetryTransport(httpx.AsyncBaseTransport):
    """Async transport wrapper that retries on transient server errors."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncHTTPTransport,
        max_attempts: int = 5,
        initial_backoff: float = 0.1,
        max_backoff: float = 3.0,
        jitter_max: float = 0.1,
    ) -> None:
        self._transport = transport
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._jitter_max = jitter_max

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._transport.handle_async_request(request)
        if request.method not in _RETRYABLE_METHODS:
            return response
        for attempt in range(self._max_attempts - 1):
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                return response
            await response.aclose()
            delay = min(self._initial_backoff * (2**attempt), self._max_backoff) + random.uniform(
                0, self._jitter_max
            )
            await asyncio.sleep(delay)
            response = await self._transport.handle_async_request(request)
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return

    body: dict[str, Any] | None = None
    try:
        body = response.json()
    except Exception:
        body = None

    message = ""
    if body and isinstance(body.get("message"), str):
        message = body["message"]
    else:
        message = f"Request failed with status {response.status_code}"

    status = response.status_code
    reason = response.reason_phrase
    headers = dict(response.headers)
    if status == 401:
        raise UnauthorizedError(message=message, status_code=status, body=body, reason=reason, headers=headers)
    if status == 403:
        raise ForbiddenError(message=message, status_code=status, body=body, reason=reason, headers=headers)
    if status == 404:
        raise NotFoundError(message=message, status_code=status, body=body, reason=reason, headers=headers)
    if status == 409:
        raise ConflictError(message=message, status_code=status, body=body, reason=reason, headers=headers)
    if 500 <= status <= 599:
        raise ServiceError(message=message, status_code=status, body=body, reason=reason, headers=headers)
    raise ApiError(message=message, status_code=status, body=body, reason=reason, headers=headers)


class HTTPClient:
    """Synchronous HTTP client wrapping httpx."""

    def __init__(self, config: PineconeConfig, api_version: str) -> None:
        self._config = config
        self._headers = _build_headers(config, api_version)
        verify: str | bool = config.ssl_ca_certs if config.ssl_ca_certs else config.ssl_verify
        pool_size = (
            config.connection_pool_maxsize
            if config.connection_pool_maxsize > 0
            else _default_pool_size()
        )
        limits = httpx.Limits(
            max_connections=pool_size,
            max_keepalive_connections=pool_size // 2,
        )
        transport = _RetryTransport(
            transport=httpx.HTTPTransport(
                http2=True, limits=limits, socket_options=_build_socket_options()
            ),
        )
        proxy: httpx.Proxy | str | None = None
        if config.proxy_url:
            if config.proxy_headers:
                proxy = httpx.Proxy(url=config.proxy_url, headers=config.proxy_headers)
            else:
                proxy = config.proxy_url
        self._client = httpx.Client(
            base_url=config.host or DEFAULT_BASE_URL,
            headers=self._headers,
            timeout=config.timeout,
            transport=transport,
            proxy=proxy,
            verify=verify,
        )

    def _build_url(self, path: str) -> str:
        return f"{self._client.base_url}{path}"

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        _log_curl("GET", self._build_url(path), dict(self._headers))
        try:
            response = self._client.get(path, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        body: bytes | None = None
        if "content" in kwargs:
            body = kwargs["content"] if isinstance(kwargs["content"], bytes) else None
        elif "json" in kwargs:
            body = _encode_json(kwargs["json"])
        _log_curl("POST", self._build_url(path), dict(self._headers), body=body)
        try:
            response = self._client.post(path, **_prepare_json_kwargs(kwargs))
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    def put(self, path: str, **kwargs: Any) -> httpx.Response:
        body: bytes | None = None
        if "content" in kwargs:
            body = kwargs["content"] if isinstance(kwargs["content"], bytes) else None
        elif "json" in kwargs:
            body = _encode_json(kwargs["json"])
        _log_curl("PUT", self._build_url(path), dict(self._headers), body=body)
        try:
            response = self._client.put(path, **_prepare_json_kwargs(kwargs))
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    def patch(self, path: str, **kwargs: Any) -> httpx.Response:
        body: bytes | None = None
        if "content" in kwargs:
            body = kwargs["content"] if isinstance(kwargs["content"], bytes) else None
        elif "json" in kwargs:
            body = _encode_json(kwargs["json"])
        _log_curl("PATCH", self._build_url(path), dict(self._headers), body=body)
        try:
            response = self._client.patch(path, **_prepare_json_kwargs(kwargs))
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        _log_curl("DELETE", self._build_url(path), dict(self._headers))
        try:
            response = self._client.delete(path, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    def close(self) -> None:
        self._client.close()


class AsyncHTTPClient:
    """Asynchronous HTTP client wrapping httpx.

    The underlying ``httpx.AsyncClient`` is created lazily on the first
    async method call rather than in ``__init__``.  This allows the
    client to be instantiated in a synchronous context (e.g. module
    scope) and used later inside an async event loop.
    """

    def __init__(self, config: PineconeConfig, api_version: str) -> None:
        self._config = config
        self._headers = _build_headers(config, api_version)
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the underlying client, creating it on first use."""
        if self._client is None:
            verify: str | bool = (
                self._config.ssl_ca_certs if self._config.ssl_ca_certs else self._config.ssl_verify
            )
            pool_size = (
                self._config.connection_pool_maxsize
                if self._config.connection_pool_maxsize > 0
                else _default_pool_size()
            )
            limits = httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size // 2,
            )
            transport = _AsyncRetryTransport(
                transport=httpx.AsyncHTTPTransport(
                    http2=True, limits=limits, socket_options=_build_socket_options()
                ),
            )
            proxy: httpx.Proxy | str | None = None
            if self._config.proxy_url:
                if self._config.proxy_headers:
                    proxy = httpx.Proxy(
                        url=self._config.proxy_url, headers=self._config.proxy_headers
                    )
                else:
                    proxy = self._config.proxy_url
            self._client = httpx.AsyncClient(
                base_url=self._config.host or DEFAULT_BASE_URL,
                headers=self._headers,
                timeout=self._config.timeout,
                transport=transport,
                proxy=proxy,
                verify=verify,
            )
        return self._client

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._ensure_client().get(path, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._ensure_client().post(path, **_prepare_json_kwargs(kwargs))
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    async def put(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._ensure_client().put(path, **_prepare_json_kwargs(kwargs))
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    async def patch(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._ensure_client().patch(path, **_prepare_json_kwargs(kwargs))
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    async def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._ensure_client().delete(path, **kwargs)
        except httpx.TimeoutException as exc:
            raise PineconeTimeoutError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise PineconeConnectionError(str(exc)) from exc
        _raise_for_status(response)
        return response

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
