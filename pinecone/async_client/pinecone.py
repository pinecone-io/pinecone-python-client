"""Asynchronous Pinecone client."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.indexes_helpers import IndexKwargs, async_poll_index_until_ready
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from pinecone.async_client.assistants import AsyncAssistants
    from pinecone.async_client.async_index import AsyncIndex
    from pinecone.async_client.backups import AsyncBackups
    from pinecone.async_client.collections import AsyncCollections
    from pinecone.async_client.indexes import AsyncIndexes
    from pinecone.async_client.inference import AsyncInference
    from pinecone.async_client.restore_jobs import AsyncRestoreJobs
    from pinecone.models.assistant.model import AssistantModel
    from pinecone.models.enums import DeletionProtection
    from pinecone.models.indexes.index import IndexModel


class AsyncPinecone:
    """Asynchronous Pinecone client for control-plane operations.

    Args:
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        host (str | None): Control-plane API host. Falls back to ``PINECONE_CONTROLLER_HOST``
            env var, then defaults to ``https://api.pinecone.io``.
        additional_headers (dict[str, str] | None): Extra headers included in every request.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        proxy_headers (dict[str, str] | None): Not yet supported. Raises
            ``NotImplementedError`` if provided.
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

        from pinecone import AsyncPinecone

        async with AsyncPinecone(api_key="your-api-key") as pc:
            indexes = await pc.indexes.list()

    .. note:: **Differences from sync Pinecone**

        1. **index(name=...) requires a cached host.** Unlike the sync
           ``Pinecone`` client, ``AsyncPinecone.index()`` is a synchronous
           factory and cannot auto-resolve an index host from its name.
           Call ``await pc.indexes.describe(name)`` first to populate the
           cache, then create the data-plane client::

               desc = await pc.indexes.describe("my-index")
               idx = pc.index("my-index")          # uses cached host
               # — or —
               idx = pc.index(host=desc.host)       # explicit host

        2. **upsert_from_dataframe() is not supported.** ``AsyncIndex``
           raises ``NotImplementedError`` for this method. Use batched
           ``upsert()`` calls instead.

        3. **No grpc parameter on index().** Async gRPC transport is not
           yet available, so the ``grpc`` option accepted by the sync
           client is absent here.
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
        if proxy_headers:
            raise NotImplementedError("proxy_headers is not yet supported for the async client")

        config = PineconeConfig(
            api_key=api_key or "",
            host=host or "",
            timeout=timeout,
            additional_headers=additional_headers or {},
            source_tag=source_tag or "",
            proxy_url=proxy_url or "",
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

        from pinecone._internal.http_client import AsyncHTTPClient

        self._http = AsyncHTTPClient(config, CONTROL_PLANE_API_VERSION)
        self._indexes: AsyncIndexes | None = None
        self._collections: AsyncCollections | None = None
        self._assistants: AsyncAssistants | None = None
        self._backups: AsyncBackups | None = None
        self._restore_jobs: AsyncRestoreJobs | None = None
        self._inference: AsyncInference | None = None
        self._host_cache: dict[str, str] = {}

    def __repr__(self) -> str:
        masked = f"...{self._config.api_key[-4:]}" if len(self._config.api_key) >= 4 else "***"
        return f"AsyncPinecone(api_key='{masked}', host='{self._config.host}')"

    @property
    def indexes(self) -> AsyncIndexes:
        """Access the AsyncIndexes namespace for control-plane index operations.

        Lazily imported and instantiated on first access.

        Returns:
            AsyncIndexes namespace instance.

        Examples:

            async with AsyncPinecone(api_key="your-api-key") as pc:
                for idx in await pc.indexes.list():
                    print(idx.name)
        """
        if self._indexes is None:
            from pinecone.async_client.indexes import AsyncIndexes as _AsyncIndexes

            self._indexes = _AsyncIndexes(http=self._http, host_cache=self._host_cache)
        return self._indexes

    @property
    def collections(self) -> AsyncCollections:
        """Access the AsyncCollections namespace for control-plane collection operations.

        Lazily imported and instantiated on first access.

        Returns:
            AsyncCollections namespace instance.

        Examples:

            async with AsyncPinecone(api_key="your-api-key") as pc:
                for col in await pc.collections.list():
                    print(col.name)
        """
        if self._collections is None:
            from pinecone.async_client.collections import AsyncCollections as _AsyncCollections

            self._collections = _AsyncCollections(http=self._http)
        return self._collections

    @property
    def assistants(self) -> AsyncAssistants:
        """Access the AsyncAssistants namespace for assistant operations.

        Lazily imported and instantiated on first access.

        Returns:
            AsyncAssistants namespace instance.

        Examples:

            async with AsyncPinecone(api_key="your-api-key") as pc:
                assistants = await pc.assistants.list()
        """
        if self._assistants is None:
            from pinecone.async_client.assistants import AsyncAssistants as _AsyncAssistants

            self._assistants = _AsyncAssistants(config=self._config)
        return self._assistants

    async def assistant(self, name: str) -> AssistantModel:
        """Convenience method to describe a single assistant by name.

        This is a shorthand for ``await pc.assistants.describe(name=name)``.

        Args:
            name (str): The name of the assistant to describe.

        Returns:
            :class:`AssistantModel` describing the assistant.

        Raises:
            :exc:`ApiError`: If the API returns an error response (e.g. 404
                when the assistant does not exist).

        Examples:

            assistant = await pc.assistant("my-assistant")
        """
        return await self.assistants.describe(name=name)

    @property
    def backups(self) -> AsyncBackups:
        """Access the AsyncBackups namespace for control-plane backup operations.

        Lazily imported and instantiated on first access.

        Returns:
            AsyncBackups namespace instance.

        Examples:

            async with AsyncPinecone(api_key="your-api-key") as pc:
                for backup in await pc.backups.list():
                    print(backup.backup_id)
        """
        if self._backups is None:
            from pinecone.async_client.backups import AsyncBackups as _AsyncBackups

            self._backups = _AsyncBackups(http=self._http)
        return self._backups

    @property
    def restore_jobs(self) -> AsyncRestoreJobs:
        """Access the AsyncRestoreJobs namespace for restore job operations.

        Lazily imported and instantiated on first access.

        Returns:
            AsyncRestoreJobs namespace instance.

        Examples:

            async with AsyncPinecone(api_key="your-api-key") as pc:
                for job in await pc.restore_jobs.list():
                    print(job.restore_job_id)
        """
        if self._restore_jobs is None:
            from pinecone.async_client.restore_jobs import AsyncRestoreJobs as _AsyncRestoreJobs

            self._restore_jobs = _AsyncRestoreJobs(http=self._http)
        return self._restore_jobs

    @property
    def inference(self) -> AsyncInference:
        """Access the AsyncInference namespace for inference operations.

        Lazily imported and instantiated on first access.

        Returns:
            AsyncInference namespace instance.

        Examples:

            async with AsyncPinecone(api_key="your-api-key") as pc:
                embeddings = await pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=["Hello, world!"],
                )
        """
        if self._inference is None:
            from pinecone.async_client.inference import AsyncInference as _AsyncInference

            self._inference = _AsyncInference(config=self._config)
        return self._inference

    async def create_index_from_backup(
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

            >>> from pinecone import AsyncPinecone
            >>> async with AsyncPinecone(api_key="your-api-key") as pc:
            ...     index = await pc.create_index_from_backup(
            ...         name="product-search-restored",
            ...         backup_id="bk-daily-20240115",
            ...     )

            Restore with tags and deletion protection:

            >>> async with AsyncPinecone(api_key="your-api-key") as pc:
            ...     index = await pc.create_index_from_backup(
            ...         name="product-search-restored",
            ...         backup_id="bk-daily-20240115",
            ...         deletion_protection="enabled",
            ...         tags={"env": "production", "team": "search"},
            ...     )
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

        response = await self._http.post(f"/backups/{backup_id}/create-index", json=body)
        BackupsAdapter.to_create_index_from_backup_response(response.content)

        if timeout == -1:
            return await self.indexes.describe(name)

        effective_timeout = timeout if timeout is not None else 300
        return await async_poll_index_until_ready(self.indexes.describe, name, effective_timeout)

    @property
    def config(self) -> PineconeConfig:
        """The resolved configuration for this client."""
        return self._config

    def _build_index_kwargs(self, host: str) -> IndexKwargs:
        """Return the kwargs dict for constructing an AsyncIndex."""
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

    def index(
        self,
        name: str = "",
        *,
        host: str = "",
    ) -> AsyncIndex:
        """Create an async data plane client targeting a specific index.

        This is a synchronous factory method (not a coroutine). It can target
        an index by host URL directly, or by name if the host has already been
        cached from a prior call or an explicit describe lookup.

        To resolve an index host from its name, use
        ``await pc.indexes.describe(name)`` first, then pass the ``host``.

        Args:
            name (str): Name of the index. Uses cached host from a prior
                describe call; raises if the host is not yet cached.
            host (str): Direct host URL of the index. Preferred path — no
                describe call needed.

        Returns:
            An async :class:`AsyncIndex` data plane client.

        Raises:
            ValidationError: If neither *name* nor *host* is provided, or if
                *name* is given but the host has not been cached yet.

        Examples:

            async with AsyncPinecone(api_key="...") as pc:
                idx = pc.index(host="my-index-abc123.svc.pinecone.io")
                # or resolve name first, then use cached host:
                await pc.indexes.describe("my-index")
                idx = pc.index(name="my-index")
        """
        from pinecone.async_client.async_index import AsyncIndex as _AsyncIndex

        if host:
            return _AsyncIndex(**self._build_index_kwargs(host))

        if name:
            # Check cache first
            cached_host = self._host_cache.get(name)
            if cached_host:
                return _AsyncIndex(**self._build_index_kwargs(cached_host))

            raise ValidationError(
                f"Host for index '{name}' is not cached. Resolve the host first with "
                "'desc = await pc.indexes.describe(name)', then call "
                "'pc.index(host=desc.host)' or 'pc.index(name=name)' (cached after describe)."
            )

        raise ValidationError("Either name or host must be provided to create an Index client.")

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.close()
        if self._assistants is not None:
            await self._assistants.close()
        if self._inference is not None:
            await self._inference.close()

    async def __aenter__(self) -> AsyncPinecone:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
