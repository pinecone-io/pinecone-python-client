"""httpx-based HTTP client for sync and async operations."""

from __future__ import annotations

from typing import Any

import httpx

from pinecone import __version__
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import API_VERSION_HEADER, DEFAULT_BASE_URL
from pinecone._internal.user_agent import build_user_agent
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    UnauthorizedError,
)


def _build_headers(config: PineconeConfig, api_version: str) -> dict[str, str]:
    headers: dict[str, str] = {
        "Api-Key": config.api_key,
        API_VERSION_HEADER: api_version,
        "User-Agent": build_user_agent(__version__, config.source_tag or None),
    }
    if config.additional_headers:
        headers.update(config.additional_headers)
    return headers


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
    if status == 401:
        raise UnauthorizedError(message=message, status_code=status, body=body)
    if status == 404:
        raise NotFoundError(message=message, status_code=status, body=body)
    if status == 409:
        raise ConflictError(message=message, status_code=status, body=body)
    raise ApiError(message=message, status_code=status, body=body)


class HTTPClient:
    """Synchronous HTTP client wrapping httpx."""

    def __init__(self, config: PineconeConfig, api_version: str) -> None:
        self._config = config
        self._headers = _build_headers(config, api_version)
        self._client = httpx.Client(
            base_url=config.host or DEFAULT_BASE_URL,
            headers=self._headers,
            timeout=config.timeout,
            http2=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        response = self._client.get(path, **kwargs)
        _raise_for_status(response)
        return response

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        response = self._client.post(path, **kwargs)
        _raise_for_status(response)
        return response

    def put(self, path: str, **kwargs: Any) -> httpx.Response:
        response = self._client.put(path, **kwargs)
        _raise_for_status(response)
        return response

    def patch(self, path: str, **kwargs: Any) -> httpx.Response:
        response = self._client.patch(path, **kwargs)
        _raise_for_status(response)
        return response

    def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        response = self._client.delete(path, **kwargs)
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
            self._client = httpx.AsyncClient(
                base_url=self._config.host or DEFAULT_BASE_URL,
                headers=self._headers,
                timeout=self._config.timeout,
                http2=True,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._ensure_client().get(path, **kwargs)
        _raise_for_status(response)
        return response

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._ensure_client().post(path, **kwargs)
        _raise_for_status(response)
        return response

    async def put(self, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._ensure_client().put(path, **kwargs)
        _raise_for_status(response)
        return response

    async def patch(self, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._ensure_client().patch(path, **kwargs)
        _raise_for_status(response)
        return response

    async def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._ensure_client().delete(path, **kwargs)
        _raise_for_status(response)
        return response

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
