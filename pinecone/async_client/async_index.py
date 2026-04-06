"""Asynchronous data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from typing import Any

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.errors.exceptions import ValidationError
from pinecone.index import _validate_host

logger = logging.getLogger(__name__)


class AsyncIndex:
    """Asynchronous data plane client targeting a specific Pinecone index.

    Can be constructed directly with a host URL, or via the
    :meth:`AsyncPinecone.index` factory method.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        additional_headers (dict[str, str] | None): Extra headers included in every request.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.

    Raises:
        ValidationError: If no API key can be resolved or the host is invalid.

    Examples:

        from pinecone import AsyncIndex

        async with AsyncIndex(host="my-index-abc123.svc.pinecone.io", api_key="...") as idx:
            print(idx.host)
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None = None,
        additional_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        # Resolve API key: explicit arg > env var (check BEFORE host per unified-ord-0001)
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._host = _validate_host(host)

        config = PineconeConfig(
            api_key=resolved_key,
            host=self._host,
            timeout=timeout,
            additional_headers=additional_headers or {},
        )
        self._config = config

        from pinecone._internal.http_client import AsyncHTTPClient

        self._http = AsyncHTTPClient(config, DATA_PLANE_API_VERSION)

        logger.info("AsyncIndex client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        await self._http.close()

    async def __aenter__(self) -> AsyncIndex:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"AsyncIndex(host='{self._host}')"
