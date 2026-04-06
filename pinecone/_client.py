"""Synchronous Pinecone client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from pinecone.client.indexes import Indexes
    from pinecone.index import Index

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

        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, CONTROL_PLANE_API_VERSION)
        self._indexes: Indexes | None = None
        self._host_cache: dict[str, str] = {}

    @property
    def indexes(self) -> Indexes:
        """Access the Indexes namespace for control-plane index operations.

        Lazily imported and instantiated on first access.

        Returns:
            Indexes namespace instance.

        Example::

            pc = Pinecone(api_key="your-api-key")
            for idx in pc.indexes.list():
                print(idx.name)
        """
        if self._indexes is None:
            from pinecone.client.indexes import Indexes as _Indexes

            self._indexes = _Indexes(http=self._http)
        return self._indexes

    def index(
        self,
        name: str = "",
        *,
        host: str = "",
    ) -> Index:
        """Create a data plane client targeting a specific index.

        Can target by host URL directly (skips the describe call) or by
        index name (triggers a describe-index lookup to resolve the host).

        Args:
            name: Name of the index. Triggers a describe call to resolve host.
            host: Direct host URL of the index. Skips the describe call.

        Returns:
            A sync :class:`Index` data plane client.

        Raises:
            ValidationError: If neither *name* nor *host* is provided.

        Example::

            pc = Pinecone(api_key="...")
            idx = pc.index(host="my-index-abc123.svc.pinecone.io")
            # or
            idx = pc.index(name="my-index")
        """
        if host:
            from pinecone.index import Index as _Index

            return _Index(
                host=host,
                api_key=self._config.api_key,
                additional_headers=dict(self._config.additional_headers),
                timeout=self._config.timeout,
            )

        if name:
            # Check cache first
            cached_host = self._host_cache.get(name)
            if cached_host:
                from pinecone.index import Index as _Index

                return _Index(
                    host=cached_host,
                    api_key=self._config.api_key,
                    additional_headers=dict(self._config.additional_headers),
                    timeout=self._config.timeout,
                )

            # Resolve host via describe
            desc = self.indexes.describe(name)
            self._host_cache[name] = desc.host

            from pinecone.index import Index as _Index

            return _Index(
                host=desc.host,
                api_key=self._config.api_key,
                additional_headers=dict(self._config.additional_headers),
                timeout=self._config.timeout,
            )

        raise ValidationError("Either name or host must be provided to create an Index client.")

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
