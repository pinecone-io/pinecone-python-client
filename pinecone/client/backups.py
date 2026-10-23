"""Backups namespace: create, list, describe, and delete backups."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.backups_adapter import BackupsAdapter
from pinecone._internal.backups_helpers import (
    backup_list_params,
    require_index_scope_for_include_deleted,
)
from pinecone._internal.validation import require_non_empty, require_str
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class Backups:
    """Stored, point-in-time snapshots of a serverless or BYOC index.

    A backup captures an index's records and schema so that a new index can
    be created from it later with
    :meth:`~pinecone.Pinecone.create_index_from_backup`. Backups are
    identified by a ``backup_id`` and outlive the index they were taken
    from. Reached as ``pc.backups``; not constructed directly.

    Pod-based indexes cannot be backed up; their snapshot mechanism is
    :class:`~pinecone.client.collections.Collections`.

    Examples:
        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> page = pc.backups.list(limit=100)
        >>> [b.backup_id for b in page]
        ['bk-abc123', 'bk-def456']

    .. seealso::
       - :meth:`~pinecone.client.indexes.Indexes.list_backups`, the
         index-scoped listing, which walks every page for you.
       - `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__,
         for the exceptions these methods raise and which ones to retry.
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http
        self._adapter = BackupsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Backups()"

    def create(
        self,
        *,
        index_name: str,
        name: str | None = None,
        description: str | None = None,
    ) -> BackupModel:
        """Create a backup of an existing index.

        Only serverless and BYOC indexes can be backed up. The call returns as
        soon as the server accepts the request, not when the backup is ready.

        Args:
            index_name (str): Name of the index to back up.
            name (str | None): Name for the backup, e.g. ``"daily-20240115"``.
                When omitted, the backup has no name and is identified only
                by its ``backup_id``.
            description (str | None): Description for the backup.

        Returns:
            A :class:`BackupModel` describing the new backup, with ``status``
            ``"Initializing"``. Poll :meth:`describe` to follow it.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty.
            :exc:`PineconeTypeError`: If *index_name* is not a ``str``.
            :exc:`ForbiddenError`: If the organization's plan does not
                include backups.
            :exc:`NotFoundError`: If *index_name* does not resolve to an
                index in this project.
            :exc:`ApiError`: If *index_name* names a pod-based index, which
                is snapshotted into a collection rather than a backup.

        Examples:
            Poll :meth:`describe` until the status *leaves*
            ``"Initializing"``. A backup that fails settles on
            ``"InitializationFailed"``, so waiting for ``"Ready"`` alone could
            loop forever.

            >>> import time
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> backup = pc.backups.create(index_name="product-search")
            >>> while backup.status == "Initializing":
            ...     time.sleep(10)
            ...     backup = pc.backups.describe(backup_id=backup.backup_id)
            >>> backup.backup_id
            'bk-abc123'
            >>> backup.status
            'Ready'

            Give the backup a name and description so a later listing
            identifies it by more than its ``backup_id``:

            >>> backup = pc.backups.create(
            ...     index_name="product-search",
            ...     name="daily-20240115",
            ...     description="Scheduled daily backup before reindexing",
            ... )
            >>> backup.name
            'daily-20240115'

        .. seealso::
           - :meth:`~pinecone.client.backup_schedules.BackupSchedules.create`,
             for backups on a recurring cadence.
           - :meth:`~pinecone.Pinecone.create_index_from_backup`, to restore
             a backup into a new index.
        """
        require_non_empty("index_name", index_name)
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        logger.info("Creating backup for index %r", index_name)
        response = self._http.post(f"/indexes/{quote(index_name, safe='')}/backups", json=body)
        result = self._adapter.to_backup(response.content)
        logger.debug("Created backup %r", result.backup_id)
        return result

    def list(
        self,
        *,
        index_name: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        include_deleted: bool | None = None,
    ) -> BackupList:
        """List one page of backups.

        When *index_name* is given, lists backups of that index only;
        otherwise lists every backup in the project. One call returns one
        page, and iterating the result does not fetch the next one. Follow
        ``pagination.next`` yourself (see
        `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__),
        or use :meth:`~pinecone.client.indexes.Indexes.list_backups`, which
        walks every page of one index's backups for you.

        Args:
            index_name (str | None): Index name to scope the listing to, or
                ``None`` for every backup in the project.
            limit (int | None): Maximum number of results per page, e.g.
                ``100``. When ``None``, the server applies its default. The
                SDK drops *limit* when *pagination_token* is given, because
                the token already carries the page size.
            pagination_token (str | None): Token for the next page, taken
                from ``BackupList.pagination.next``.
            include_deleted (bool | None): When ``True``, include backups of
                every index that has ever used *index_name*, including
                deleted ones. Defaults to the server's behavior, which
                excludes them. Only valid together with *index_name*.

        Returns:
            A :class:`BackupList` holding one page, supporting iteration,
            ``len()``, and index access. ``pagination`` is ``None`` on the
            final page. Backups created or deleted during a walk can shift
            later pages, so de-duplicate by ``backup_id``.

        Raises:
            :exc:`PineconeTypeError`: If *index_name* is given and is not a ``str``.
            :exc:`PineconeValueError`: If *include_deleted* is given without
                *index_name*.
            :exc:`NotFoundError`: If *index_name* does not name an active
                index and *include_deleted* is not ``True``.

        Examples:
            Passing *index_name* scopes the listing to one index:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> for backup in pc.backups.list(index_name="product-search"):
            ...     print(backup.name, backup.status)
            daily-20240115 Ready

            Walk every page of the project-wide listing:

            >>> page = pc.backups.list(limit=100)
            >>> backups = list(page)
            >>> while page.pagination and page.pagination.next:
            ...     page = pc.backups.list(pagination_token=page.pagination.next)
            ...     backups.extend(page)
            >>> [b.backup_id for b in backups]
            ['bk-abc123', 'bk-def456', 'bk-ghi789']

            Pass ``include_deleted=True`` to reach the backups of an index
            you have already deleted:

            >>> orphaned = pc.backups.list(
            ...     index_name="legacy-catalog", include_deleted=True
            ... )
            >>> [b.backup_id for b in orphaned if b.source_index_deleted_at]
            ['bk-old111']

        .. note::
           If every index that used *index_name* has been deleted, listing
           without *include_deleted* raises :exc:`NotFoundError` rather than
           returning an empty list. Pass ``include_deleted=True`` to see
           their backups.

        .. versionchanged:: 10.0
           Added *include_deleted*. :class:`BackupModel` now carries
           :attr:`~pinecone.models.backups.model.BackupModel.source_index_deleted_at`
           instead of ``dimension``/``metric``.
        """
        require_index_scope_for_include_deleted(index_name, include_deleted)
        params: dict[str, Any] = backup_list_params(
            limit=limit,
            pagination_token=pagination_token,
            include_deleted=include_deleted,
        )

        if index_name is not None:
            require_str("index_name", index_name)
            path = f"/indexes/{quote(index_name, safe='')}/backups"
        else:
            path = "/backups"

        logger.info("Listing backups (path=%s)", path)
        response = self._http.get(path, params=params)
        result = self._adapter.to_backup_list(response.content)
        logger.debug("Listed %d backups", len(result))
        return result

    def describe(self, *, backup_id: str) -> BackupModel:
        """Get the current state of one backup.

        Args:
            backup_id (str): The identifier of the backup to describe.

        Returns:
            A :class:`BackupModel` whose ``status`` is ``"Initializing"``,
            ``"Ready"``, or ``"InitializationFailed"``, alongside the
            ``source_index_name`` it was taken from, the captured ``schema``,
            and the ``record_count`` and ``size_bytes`` of the snapshot.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`PineconeTypeError`: If *backup_id* is not a ``str``.
            :exc:`NotFoundError`: If the backup does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> backup = pc.backups.describe(backup_id="bk-abc123")
            >>> backup.status
            'Ready'
            >>> backup.source_index_name
            'product-search'

        .. seealso::
           :meth:`~pinecone.client.indexes.Indexes.describe_backup`, the
           same call under the ``indexes`` namespace.
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Describing backup %r", backup_id)
        response = self._http.get(f"/backups/{quote(backup_id, safe='')}")
        result = self._adapter.to_backup(response.content)
        logger.debug("Described backup %r", backup_id)
        return result

    def get(self, *, backup_id: str) -> BackupModel:
        """Get detailed information about a backup (alias for :meth:`describe`).

        Args:
            backup_id (str): The identifier of the backup.

        Returns:
            A :class:`BackupModel` whose ``status`` is ``"Initializing"``,
            ``"Ready"``, or ``"InitializationFailed"``, alongside the
            ``source_index_name`` it was taken from, the captured ``schema``,
            and the ``record_count`` and ``size_bytes`` of the snapshot.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`PineconeTypeError`: If *backup_id* is not a ``str``.
            :exc:`NotFoundError`: If the backup does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> backup = pc.backups.get(backup_id="bk-abc123")
            >>> backup.status
            'Ready'
            >>> backup.source_index_name
            'product-search'
        """
        return self.describe(backup_id=backup_id)

    def delete(self, *, backup_id: str) -> None:
        """Delete a backup.

        Args:
            backup_id (str): The identifier of the backup to delete.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`PineconeTypeError`: If *backup_id* is not a ``str``.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`FailedPreconditionError`: If a restore job from this backup
                is still running. Wait for it to finish, then delete.

        Examples:
            Deleting a backup leaves its source index, other backups of that
            index, and any index restored from it untouched:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> pc.backups.delete(backup_id="bk-abc123")
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Deleting backup %r", backup_id)
        self._http.delete(f"/backups/{quote(backup_id, safe='')}")
        logger.debug("Deleted backup %r", backup_id)
