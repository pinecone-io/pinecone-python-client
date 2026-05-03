"""Synchronous Pinecone client."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.indexes_helpers import IndexKwargs, poll_index_until_ready
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from pinecone.client._assistant_namespace_proxy import _AssistantNamespaceProxy
    from pinecone.client.assistants import Assistants
    from pinecone.client.backups import Backups
    from pinecone.client.collections import Collections
    from pinecone.client.indexes import Indexes
    from pinecone.client.inference import Inference
    from pinecone.client.restore_jobs import RestoreJobs
    from pinecone.grpc import GrpcIndex
    from pinecone.index import Index
    from pinecone.inference.models.index_embed import IndexEmbed
    from pinecone.models.backups.list import BackupList, RestoreJobList
    from pinecone.models.backups.model import BackupModel, RestoreJobModel
    from pinecone.models.collections.list import CollectionList
    from pinecone.models.collections.model import CollectionModel
    from pinecone.models.enums import (
        AwsRegion,
        AzureRegion,
        CloudProvider,
        DeletionProtection,
        GcpRegion,
        Metric,
        VectorType,
    )
    from pinecone.models.indexes.index import IndexModel
    from pinecone.models.indexes.list import IndexList
    from pinecone.models.indexes.specs import (
        ByocSpec,
        EmbedConfig,
        IntegratedSpec,
        PodSpec,
        ServerlessSpec,
    )
    from pinecone.preview import Preview


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
        :exc:`PineconeValueError`: If no API key can be resolved from arguments or
            environment variables.

    Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone(api_key="your-api-key")  # or set PINECONE_API_KEY env var

            # Control plane: manage indexes
            indexes = pc.indexes.list()

            # Data plane: operate on vectors
            index = pc.index("my-index")
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
        self._preview: Preview | None = None

    def __repr__(self) -> str:
        masked = f"...{self._config.api_key[-4:]}" if len(self._config.api_key) >= 4 else "***"
        return f"Pinecone(api_key='{masked}', host='{self._config.host}')"

    @property
    def indexes(self) -> Indexes:
        """Access the Indexes namespace for control-plane index operations.

        Lazily imported and instantiated on first access.

        Returns:
            :class:`Indexes` namespace instance.

        Examples:

            >>> names = [idx.name for idx in pc.indexes.list()]  # doctest: +SKIP
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
            :class:`Collections` namespace instance.

        Examples:

            >>> names = [col.name for col in pc.collections.list()]  # doctest: +SKIP
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
            :class:`Backups` namespace instance.

        Examples:

            >>> ids = [b.backup_id for b in pc.backups.list()]  # doctest: +SKIP
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
            :class:`RestoreJobs` namespace instance.

        Examples:

            >>> ids = [job.restore_job_id for job in pc.restore_jobs.list()]  # doctest: +SKIP
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
            :class:`Inference` namespace instance.

        Examples:

            >>> embeddings = pc.inference.embed(  # doctest: +SKIP
            ...     model="multilingual-e5-large", inputs=["Hello, world!"]
            ... )
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
            :class:`Assistants` namespace instance.

        Examples:

            >>> names = [a.name for a in pc.assistants.list()]  # doctest: +SKIP
        """
        if self._assistants is None:
            from pinecone.client.assistants import Assistants as _Assistants

            self._assistants = _Assistants(config=self._config)
        return self._assistants

    @property
    def assistant(self) -> _AssistantNamespaceProxy:
        """Convenience alias for :attr:`Pinecone.assistants`.

        Returns a proxy that supports both namespace-style access
        (``pc.assistant.create_assistant(...)``) and the convenience call
        form (``pc.assistant("my-name")`` — shortcut for
        ``pc.assistants.describe(name="my-name")``).

        The canonical entry point is :attr:`Pinecone.assistants`; this
        alias is provided for ergonomic singular-form access and is not
        deprecated.
        """
        from pinecone.client._assistant_namespace_proxy import _AssistantNamespaceProxy

        return _AssistantNamespaceProxy(self.assistants)

    @property
    def preview(self) -> Preview:
        """Access the Preview namespace for pre-release API features.

        Lazily imported and instantiated on first access. Preview surface is
        not covered by SemVer — signatures and behavior may change in any
        minor SDK release.

        Returns:
            :class:`~pinecone.preview.Preview` namespace instance.

        Examples:

            .. code-block:: python

                pc = Pinecone(api_key="your-api-key")
                pc.preview.indexes.create(...)  # when a preview area exists
        """
        if self._preview is None:
            from pinecone.preview import Preview as _Preview

            self._preview = _Preview(http=self._http, config=self._config)
        return self._preview

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
            :exc:`PineconeValueError`: If neither *name* nor *host* is provided.

        Examples:

            .. code-block:: python

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
            proxy_headers=dict(self._config.proxy_headers),
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
            :exc:`PineconeValueError`: If *name* or *backup_id* is empty.
            :exc:`PineconeTimeoutError`: If the index is not ready within the timeout.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.create_index_from_backup(  # doctest: +SKIP
            ...     name="product-search-restored",
            ...     backup_id="bk-daily-20240115",
            ... )

            >>> index = pc.create_index_from_backup(  # doctest: +SKIP
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

    # ---- Backcompat flat-method delegates (:meta private:) ----

    def create_index(
        self,
        name: str,
        spec: ServerlessSpec | PodSpec | ByocSpec | IntegratedSpec | dict[str, Any],
        dimension: int | None = None,
        metric: Metric | str | None = "cosine",
        timeout: int | None = None,
        deletion_protection: DeletionProtection | str | None = "disabled",
        vector_type: VectorType | str = "dense",
        tags: dict[str, str] | None = None,
    ) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.create()`` instead of ``pc.create_index()``.
        """
        resolved_dp = deletion_protection if deletion_protection is not None else "disabled"
        return self.indexes.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric if metric is not None else "cosine",
            vector_type=vector_type,
            deletion_protection=resolved_dp,
            tags=tags,
            timeout=timeout,
        )

    def create_index_for_model(
        self,
        name: str,
        cloud: CloudProvider | str,
        region: AwsRegion | GcpRegion | AzureRegion | str,
        embed: IndexEmbed | EmbedConfig | dict[str, Any],
        tags: dict[str, str] | None = None,
        deletion_protection: DeletionProtection | str | None = "disabled",
        read_capacity: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> IndexModel:
        """Backwards-compatibility shim for creating an integrated (model-backed) index.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.create()`` with an ``IntegratedSpec``
        (``IntegratedSpec(cloud=..., region=..., embed=EmbedConfig(...))``)
        instead of ``pc.create_index_for_model()``.
        """
        from pinecone.inference.models.index_embed import IndexEmbed as _IndexEmbed
        from pinecone.models.indexes.specs import EmbedConfig as _EmbedConfig
        from pinecone.models.indexes.specs import IntegratedSpec as _IntegratedSpec

        if isinstance(embed, _IndexEmbed):
            embed_config: EmbedConfig = _EmbedConfig(
                model=embed.model,
                field_map={k: str(v) for k, v in embed.field_map.items()},
                metric=embed.metric,
                read_parameters=embed.read_parameters or None,
                write_parameters=embed.write_parameters or None,
            )
        elif isinstance(embed, _EmbedConfig):
            embed_config = embed
        else:
            embed_config = _EmbedConfig(**embed)

        cloud_str = cloud.value if hasattr(cloud, "value") else str(cloud)
        region_str = region.value if hasattr(region, "value") else str(region)
        spec = _IntegratedSpec(cloud=cloud_str, region=region_str, embed=embed_config)
        resolved_dp = deletion_protection if deletion_protection is not None else "disabled"
        return self.indexes.create(
            name=name,
            spec=spec,
            tags=tags,
            deletion_protection=resolved_dp,
            schema=schema,
            timeout=timeout,
        )

    def describe_index(self, name: str) -> IndexModel:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.describe()`` instead of ``pc.describe_index()``.
        """
        return self.indexes.describe(name)

    def list_indexes(self) -> IndexList:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should use ``pc.indexes.list()`` instead of ``pc.list_indexes()``.
        """
        return self.indexes.list()

    def has_index(self, name: str) -> bool:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.exists`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.exists()`` instead of ``pc.has_index()``.
        """
        return self.indexes.exists(name)

    def configure_index(
        self,
        name: str,
        replicas: int | None = None,
        pod_type: str | None = None,
        deletion_protection: DeletionProtection | str | None = None,
        tags: dict[str, str] | None = None,
        embed: dict[str, Any] | None = None,
        read_capacity: dict[str, Any] | None = None,
    ) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.configure`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.configure()`` instead of ``pc.configure_index()``.
        """
        self.indexes.configure(
            name=name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
            embed=embed,
            read_capacity=read_capacity,
        )

    def delete_index(self, name: str, timeout: int | None = None) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.indexes.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.indexes.delete()`` instead of ``pc.delete_index()``.
        """
        self.indexes.delete(name, timeout=timeout)

    def create_collection(self, name: str, source: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.create()`` instead of ``pc.create_collection()``.
        """
        return self.collections.create(name=name, source=source)

    def list_collections(self) -> CollectionList:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.list()`` instead of ``pc.list_collections()``.
        """
        return self.collections.list()

    def describe_collection(self, name: str) -> CollectionModel:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.describe()`` instead of ``pc.describe_collection()``.
        """
        return self.collections.describe(name)

    def delete_collection(self, name: str) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.collections.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.collections.delete()`` instead of ``pc.delete_collection()``.
        """
        self.collections.delete(name)

    def create_backup(
        self,
        *,
        index_name: str,
        backup_name: str | None = None,
        description: str = "",
    ) -> BackupModel:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.create`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.create()`` instead of ``pc.create_backup()``.
        """
        return self.backups.create(
            index_name=index_name,
            name=backup_name,
            description=description,
        )

    def list_backups(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = 10,
        pagination_token: str | None = None,
    ) -> BackupList:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.list()`` instead of ``pc.list_backups()``.
        """
        return self.backups.list(
            index_name=index_name,
            limit=limit if limit is not None else 10,
            pagination_token=pagination_token,
        )

    def describe_backup(self, *, backup_id: str) -> BackupModel:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.describe()`` instead of ``pc.describe_backup()``.
        """
        return self.backups.describe(backup_id=backup_id)

    def delete_backup(self, *, backup_id: str) -> None:
        """Backwards-compatibility shim for :meth:`Pinecone.backups.delete`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.backups.delete()`` instead of ``pc.delete_backup()``.
        """
        self.backups.delete(backup_id=backup_id)

    def list_restore_jobs(
        self,
        *,
        limit: int | None = 10,
        pagination_token: str | None = None,
    ) -> RestoreJobList:
        """Backwards-compatibility shim for :meth:`Pinecone.restore_jobs.list`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.restore_jobs.list()`` instead of ``pc.list_restore_jobs()``.
        """
        return self.restore_jobs.list(
            limit=limit if limit is not None else 10,
            pagination_token=pagination_token,
        )

    def describe_restore_job(self, *, job_id: str) -> RestoreJobModel:
        """Backwards-compatibility shim for :meth:`Pinecone.restore_jobs.describe`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.restore_jobs.describe()`` instead of ``pc.describe_restore_job()``.
        """
        return self.restore_jobs.describe(job_id=job_id)

    def Index(self, name: str = "", host: str = "", **kwargs: Any) -> Index:  # noqa: N802
        """Backwards-compatibility shim for :meth:`Pinecone.index`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New code
        should use ``pc.index(name=..., host=...)`` instead of ``pc.Index(...)``.
        """
        from pinecone.index import Index as _Index

        return cast(_Index, self.index(name=name, host=host))

    def IndexAsyncio(self, host: str, **kwargs: Any) -> Any:  # noqa: N802
        """Backwards-compatibility shim that returns an :class:`AsyncIndex`.

        Preserved to ease migration from the legacy Pinecone Python SDK. New
        code should construct an :class:`AsyncPinecone` and call ``.index(host=...)``
        on it (or instantiate :class:`AsyncIndex` directly) instead of
        ``Pinecone.IndexAsyncio(...)``.
        """
        from pinecone.async_client.async_index import AsyncIndex as _AsyncIndex

        return _AsyncIndex(
            host=host,
            api_key=self._config.api_key,
            additional_headers=dict(self._config.additional_headers),
            timeout=self._config.timeout,
            proxy_url=self._config.proxy_url,
            proxy_headers=dict(self._config.proxy_headers),
            ssl_ca_certs=self._config.ssl_ca_certs,
            ssl_verify=self._config.ssl_verify,
            source_tag=self._config.source_tag,
            connection_pool_maxsize=self._config.connection_pool_maxsize,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
        if self._inference is not None:
            self._inference.close()
        if self._assistants is not None:
            self._assistants.close()
        if self._preview is not None:
            self._preview.close()

    def __enter__(self) -> Pinecone:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
