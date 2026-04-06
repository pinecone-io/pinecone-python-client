"""Synchronous Pinecone client."""

from __future__ import annotations

from typing import Any

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.http_client import HTTPClient
from pinecone.errors.exceptions import ValidationError

_DEPRECATED_KWARGS: frozenset[str] = frozenset({"openapi_config", "pool_threads", "index_api"})


class Pinecone:
    """Synchronous Pinecone client for control-plane operations.

    Args:
        api_key: Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        host: Control-plane API host. Falls back to ``PINECONE_CONTROLLER_HOST``
            env var, then defaults to ``https://api.pinecone.io``.
        additional_headers: Extra headers included in every request.
        source_tag: Tag appended to the User-Agent string for request attribution.
        proxy_url: HTTP proxy URL for outgoing requests.
        ssl_ca_certs: Path to a CA certificate bundle for SSL verification.
        ssl_verify: Whether to verify SSL certificates. Defaults to ``True``.
        timeout: Request timeout in seconds. Defaults to ``30.0``.

    Raises:
        ValidationError: If no API key can be resolved from arguments or
            environment variables.

    Example::

        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        host: str | None = None,
        additional_headers: dict[str, str] | None = None,
        source_tag: str | None = None,
        proxy_url: str | None = None,
        ssl_ca_certs: str | None = None,
        ssl_verify: bool = True,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        # Check for deprecated kwargs
        deprecated_used = _DEPRECATED_KWARGS & set(kwargs)
        if deprecated_used:
            names = ", ".join(sorted(deprecated_used))
            raise ValidationError(
                f"The following parameters are no longer supported: {names}. "
                "See the migration guide for updated usage."
            )

        config = PineconeConfig(
            api_key=api_key or "",
            host=host or "",
            timeout=timeout,
            additional_headers=additional_headers or {},
            source_tag=source_tag or "",
            proxy_url=proxy_url or "",
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
        )

        if not config.api_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Apply default host if none resolved
        resolved_host = config.host or DEFAULT_BASE_URL
        if resolved_host != config.host:
            config = PineconeConfig(
                api_key=config.api_key,
                host=resolved_host,
                timeout=config.timeout,
                additional_headers=config.additional_headers,
                source_tag=config.source_tag,
                proxy_url=config.proxy_url,
                ssl_ca_certs=config.ssl_ca_certs,
                ssl_verify=config.ssl_verify,
            )

        self._config = config
        self._http = HTTPClient(config, CONTROL_PLANE_API_VERSION)

    @property
    def config(self) -> PineconeConfig:
        """The resolved configuration for this client."""
        return self._config

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> Pinecone:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
