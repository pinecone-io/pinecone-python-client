"""Synchronous Pinecone client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import CONTROL_PLANE_API_VERSION, DEFAULT_BASE_URL
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from pinecone.client.backups import Backups
    from pinecone.client.collections import Collections
    from pinecone.client.indexes import Indexes
    from pinecone.client.restore_jobs import RestoreJobs
    from pinecone.index import Index
    from pinecone.models.enums import DeletionProtection
    from pinecone.models.indexes.index import IndexModel

_DEPRECATED_KWARGS: frozenset[str] = frozenset({"openapi_config", "pool_threads", "index_api"})


class Pinecone:
    """Synchronous Pinecone client for control-plane operations.

    Args:
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        host (str | None): Control-plane API host. Falls back to ``PINECONE_CONTROLLER_HOST``
            env var, then defaults to ``https://api.pinecone.io``.
        additional_headers (dict[str, str] | None): Extra headers included in every request.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        proxy_url (str | None): HTTP proxy URL for outgoing requests.
        ssl_ca_certs (str | None): Path to a CA certificate bundle for SSL verification.
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to ``True``.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.

    Raises:
        ValidationError: If no API key can be resolved from arguments or
            environment variables, or if deprecated keyword arguments
            (``openapi_config``, ``pool_threads``, ``index_api``) are passed.
            These parameters are no longer supported; see the migration guide
            for updated usage.

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
        self._collections: Collections | None = None
        self._backups: Backups | None = None
        self._restore_jobs: RestoreJobs | None = None
        self._host_cache: dict[str, str] = {}

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

            self._indexes = _Indexes(http=self._http)
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
            name (str): Name of the index. Triggers a describe call to resolve host.
            host (str): Direct host URL of the index. Skips the describe call.

        Returns:
            A sync :class:`Index` data plane client.

        Raises:
            ValidationError: If neither *name* nor *host* is provided.

        Examples:

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
            An IndexModel describing the restored index.

        Raises:
            ValidationError: If *name* or *backup_id* is empty.
            PineconeTimeoutError: If the index is not ready within the timeout.
            ApiError: If the API returns an error response.

        Examples:

            pc = Pinecone(api_key="...")
            index = pc.create_index_from_backup(
                name="restored-index",
                backup_id="bk-123",
            )
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
        return self.indexes._poll_until_ready(name, effective_timeout)

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
