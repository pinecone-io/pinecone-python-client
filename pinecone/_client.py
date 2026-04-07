"""Synchronous Pinecone client."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.indexes_helpers import IndexKwargs, poll_index_until_ready
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from pinecone.client.assistants import Assistants
    from pinecone.client.backups import Backups
    from pinecone.client.collections import Collections
    from pinecone.client.indexes import Indexes
    from pinecone.client.inference import Inference
    from pinecone.client.restore_jobs import RestoreJobs
    from pinecone.grpc import GrpcIndex
    from pinecone.index import Index
    from pinecone.models.assistant.model import AssistantModel
    from pinecone.models.enums import DeletionProtection
    from pinecone.models.indexes.index import IndexModel


class Pinecone:
    """Synchronous Pinecone client for control-plane operations.

    Args:
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        host (str | None): Control-plane API host. Falls back to ``PINECONE_CONTROLLER_HOST``
            env var, then defaults to ``https://api.pinecone.io``.
        additional_headers (dict[str, str] | None): Extra headers included in every request.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        proxy_headers (dict[str, str] | None): Custom headers for proxy authentication.
        ssl_ca_certs (str | None): Path to a CA certificate bundle for SSL verification.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.
        connection_pool_maxsize (int): Maximum number of connections to keep in the
            pool. ``0`` (default) uses httpx defaults.
        retry_config (RetryConfig | None): Custom retry configuration. When ``None``
            (default), uses built-in defaults (5 attempts, exponential backoff, retries
            on 500/502/503/504 for GET/HEAD).

    Raises:
        ValidationError: If no API key can be resolved from arguments or
            environment variables.

    Examples:

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
        proxy_headers: dict[str, str] | None = None,
        ssl_ca_certs: str | None = None,
        ssl_verify: bool = True,
        timeout: float = 30.0,
        connection_pool_maxsize: int = 0,
        retry_config: RetryConfig | None = None,
    ) -> None:
        config = PineconeConfig(
            api_key=api_key or "",
            host=host or "",
            timeout=timeout,
            additional_headers=additional_headers or {},
            source_tag=source_tag or "",
            proxy_url=proxy_url or "",
            proxy_headers=proxy_headers or {},
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            connection_pool_maxsize=connection_pool_maxsize,
            retry_config=retry_config or RetryConfig(),
        )

        if not config.api_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Apply default host if none resolved
        resolved_host = config.host or DEFAULT_BASE_URL
        if resolved_host != config.host:
            config = replace(config, host=resolved_host)

        self._config = config

        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, CONTROL_PLANE_API_VERSION)
        self._indexes: Indexes | None = None
        self._collections: Collections | None = None
        self._backups: Backups | None = None
        self._restore_jobs: RestoreJobs | None = None
        self._inference: Inference | None = None
        self._assistants: Assistants | None = None
        self._host_cache: dict[str, str] = {}

    def __repr__(self) -> str:
        masked = f"...{self._config.api_key[-4:]}" if len(self._config.api_key) >= 4 else "***"
        return f"Pinecone(api_key='{masked}', host='{self._config.host}')"

    @property
    def indexes(self) -> Indexes:
        """Access the Indexes namespace for control-plane index operations.

        Lazily imported and instantiated on first access.

        Returns:
            Indexes namespace instance.

        Examples:

            pc = Pinecone(api_key="your-api-key")
            for idx in pc.indexes.list():
                print(idx.name)
        """
        if self._indexes is None:
            from pinecone.client.indexes import Indexes as _Indexes

            self._indexes = _Indexes(http=self._http, host_cache=self._host_cache)
        return self._indexes

    @property
    def collections(self) -> Collections:
        """Access the Collections namespace for collection operations.

        Lazily imported and instantiated on first access.

        Returns:
            Collections namespace instance.

        Examples:

            pc = Pinecone(api_key="your-api-key")
            for col in pc.collections.list():
                print(col.name)
        """
        if self._collections is None:
            from pinecone.client.collections import Collections as _Collections

            self._collections = _Collections(http=self._http)
        return self._collections

    @property
    def backups(self) -> Backups:
        """Access the Backups namespace for backup operations.

        Lazily imported and instantiated on first access.

        Returns:
            Backups namespace instance.

        Examples:

            pc = Pinecone(api_key="your-api-key")
            for backup in pc.backups.list():
                print(backup.backup_id)
        """
        if self._backups is None:
            from pinecone.client.backups import Backups as _Backups

            self._backups = _Backups(http=self._http)
        return self._backups

    @property
    def restore_jobs(self) -> RestoreJobs:
        """Access the RestoreJobs namespace for restore job operations.

        Lazily imported and instantiated on first access.

        Returns:
            RestoreJobs namespace instance.

        Examples:

            pc = Pinecone(api_key="your-api-key")
            for job in pc.restore_jobs.list():
                print(job.restore_job_id)
        """
        if self._restore_jobs is None:
            from pinecone.client.restore_jobs import RestoreJobs as _RestoreJobs

            self._restore_jobs = _RestoreJobs(http=self._http)
        return self._restore_jobs

    @property
    def inference(self) -> Inference:
        """Access the Inference namespace for embed and rerank operations.

        Lazily imported and instantiated on first access.

        Returns:
            Inference namespace instance.

        Examples:

            pc = Pinecone(api_key="your-api-key")
            embeddings = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=["Hello, world!"],
            )
        """
        if self._inference is None:
            from pinecone.client.inference import Inference as _Inference

            self._inference = _Inference(config=self._config)
        return self._inference

    @property
    def assistants(self) -> Assistants:
        """Access the Assistants namespace for assistant operations.

        Lazily imported and instantiated on first access.

        Returns:
            Assistants namespace instance.

        Examples:

            pc = Pinecone(api_key="your-api-key")
            assistants = pc.assistants
        """
        if self._assistants is None:
            from pinecone.client.assistants import Assistants as _Assistants

            self._assistants = _Assistants(config=self._config)
        return self._assistants

    def assistant(self, name: str) -> AssistantModel:
        """Convenience method to retrieve an existing assistant by name.

        This is a shorthand for ``pc.assistants.describe(name=name)``.

        Args:
            name (str): The name of the assistant to retrieve.

        Returns:
            :class:`AssistantModel` describing the assistant.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> assistant = pc.assistant("research-assistant")
            >>> print(assistant.status)
        """
        return self.assistants.describe(name=name)

    def index(
        self,
        name: str = "",
        *,
        host: str = "",
        grpc: bool = False,
    ) -> Index | GrpcIndex:
        """Create a data plane client targeting a specific index.

        Can target by host URL directly (skips the describe call) or by
        index name (triggers a describe-index lookup to resolve the host).

        .. seealso::
           Use ``pc.indexes`` for control-plane operations (create, list,
           describe, delete, configure). To create an index from a backup,
           use :meth:`Pinecone.create_index_from_backup`.

        Args:
            name (str): Name of the index. Triggers a describe call to resolve host.
            host (str): Direct host URL of the index. Skips the describe call.
            grpc (bool): If ``True``, return a :class:`~pinecone.grpc.GrpcIndex`
                that routes data-plane operations over gRPC instead of HTTP.
                Defaults to ``False``.

        Returns:
            A sync :class:`Index` (HTTP) or :class:`~pinecone.grpc.GrpcIndex`
            (gRPC) data plane client.

        Raises:
            ValidationError: If neither *name* nor *host* is provided.

        Examples:

            pc = Pinecone(api_key="...")
            idx = pc.index(host="my-index-abc123.svc.pinecone.io")
            # or
            idx = pc.index(name="my-index")
            # gRPC transport
            idx = pc.index(name="my-index", grpc=True)
        """
        resolved_host = self._resolve_index_host(name=name, host=host)

        if grpc:
            from pinecone.grpc import GrpcIndex as _GrpcIndex

            return _GrpcIndex(
                host=resolved_host,
                api_key=self._config.api_key,
                source_tag=self._config.source_tag or None,
            )

        from pinecone.index import Index as _Index

        return _Index(**self._build_index_kwargs(resolved_host))

    def _build_index_kwargs(self, host: str) -> IndexKwargs:
        """Return the kwargs dict for constructing an Index or AsyncIndex."""
        return IndexKwargs(
            host=host,
            api_key=self._config.api_key,
            additional_headers=dict(self._config.additional_headers),
            timeout=self._config.timeout,
            proxy_url=self._config.proxy_url,
            ssl_ca_certs=self._config.ssl_ca_certs,
            ssl_verify=self._config.ssl_verify,
            source_tag=self._config.source_tag,
            connection_pool_maxsize=self._config.connection_pool_maxsize,
        )

    def _resolve_index_host(self, *, name: str, host: str) -> str:
        """Resolve the data plane host from explicit host, cache, or describe call.

        Args:
            name: Index name (triggers describe if not cached).
            host: Direct host URL (returned as-is if provided).

        Returns:
            The resolved host string.

        Raises:
            ValidationError: If neither *name* nor *host* is provided.
        """
        if host:
            return host

        if name:
            cached_host = self._host_cache.get(name)
            if cached_host:
                return cached_host

            desc = self.indexes.describe(name)
            self._host_cache[name] = desc.host
            return desc.host

        raise ValidationError("Either name or host must be provided to create an Index client.")

    def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: DeletionProtection | str | None = None,
        tags: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """Create a new index by restoring from a backup.

        Sends a POST to ``/backups/{backup_id}/create-index`` and then
        polls until the index is ready (unless *timeout* is ``-1``).

        Args:
            name (str): Name for the new index.
            backup_id (str): Identifier of the backup to restore from.
            deletion_protection (DeletionProtection | str | None): ``"enabled"`` or
                ``"disabled"``. Defaults to ``"disabled"`` server-side when omitted.
            tags (dict[str, str] | None): Optional key-value tags for the new index.
            timeout (int | None): Seconds to wait for readiness. ``None`` (default)
                blocks up to 300 s. ``-1`` returns immediately without polling.

        Returns:
            An :class:`IndexModel` describing the restored index.

        Raises:
            :exc:`ValidationError`: If *name* or *backup_id* is empty.
            :exc:`PineconeTimeoutError`: If the index is not ready within the timeout.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            Restore an index from a backup:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.create_index_from_backup(
            ...     name="product-search-restored",
            ...     backup_id="bk-daily-20240115",
            ... )

            Restore with tags and deletion protection:

            >>> index = pc.create_index_from_backup(
            ...     name="product-search-restored",
            ...     backup_id="bk-daily-20240115",
            ...     deletion_protection="enabled",
            ...     tags={"env": "production", "team": "search"},
            ... )
        """
        require_non_empty("name", name)
        require_non_empty("backup_id", backup_id)

        body: dict[str, Any] = {"name": name}
        if deletion_protection is not None:
            dp_val = (
                deletion_protection.value
                if hasattr(deletion_protection, "value")
                else deletion_protection
            )
            body["deletion_protection"] = dp_val
        if tags is not None:
            body["tags"] = tags

        from pinecone._internal.adapters.backups_adapter import BackupsAdapter

        response = self._http.post(f"/backups/{backup_id}/create-index", json=body)
        BackupsAdapter.to_create_index_from_backup_response(response.content)

        if timeout == -1:
            return self.indexes.describe(name)

        effective_timeout = timeout if timeout is not None else 300
        return poll_index_until_ready(self.indexes.describe, name, effective_timeout)

    @property
    def config(self) -> PineconeConfig:
        """The resolved configuration for this client."""
        return self._config

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
        if self._inference is not None:
            self._inference.close()
        if self._assistants is not None:
            self._assistants.close()

    def __enter__(self) -> Pinecone:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
