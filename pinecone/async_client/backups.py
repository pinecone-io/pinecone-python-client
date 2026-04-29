"""Async Backups namespace — create, list, describe, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.backups_adapter import BackupsAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncBackups:
    """Async control-plane operations for Pinecone backups.

    Provides methods to create, list, describe, and delete backups.

    Args:
        http (AsyncHTTPClient): Async HTTP client for making API requests.

    Examples:

        from pinecone import AsyncPinecone

        async with AsyncPinecone(api_key="your-api-key") as pc:
            for backup in await pc.backups.list():
                print(backup.backup_id)
    """

    def __init__(self, http: AsyncHTTPClient) -> None:
        self._http = http
        self._adapter = BackupsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncBackups()"

    async def create(
        self,
        *,
        index_name: str,
        name: str | None = None,
        description: str = "",
    ) -> BackupModel:
        """Create a backup of an existing index.

        Args:
            index_name (str): Name of the index to back up.
            name (str | None): Optional name for the backup.
            description (str): Description for the backup (defaults to empty string).

        Returns:
            A :class:`BackupModel` describing the created backup.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty.
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            Create a backup of an index:

            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.create(
                        index_name="product-search",
                    )
                    print(backup.backup_id)

            Create a backup with a name and description:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.create(
                        index_name="product-search",
                        name="daily-20240115",
                        description="Scheduled daily backup before reindexing",
                    )
        """
        require_non_empty("index_name", index_name)
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        body["description"] = description
        logger.info("Creating backup for index %r", index_name)
        response = await self._http.post(f"/indexes/{index_name}/backups", json=body)
        result = self._adapter.to_backup(response.content)
        logger.debug("Created backup %r", result.backup_id)
        return result

    async def list(
        self,
        *,
        index_name: str | None = None,
        limit: int = 10,
        pagination_token: str | None = None,
    ) -> BackupList:
        """List backups.

        When *index_name* is provided, lists backups for that index only.
        Otherwise lists all backups in the project.

        Args:
            index_name (str | None): Index name to filter by, or ``None`` for all.
            limit (int): Maximum number of results per page. Defaults to 10.
            pagination_token (str | None): Token for cursor-based pagination.

        Returns:
            A :class:`BackupList` supporting iteration, len(), and index access.

        Raises:
            :exc:`ApiError`: If the API returns an error response.

        Examples:
            List all backups in the project:

            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list():
                        print(backup.backup_id, backup.name)

            List backups for a specific index:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    for backup in await pc.backups.list(
                        index_name="product-search",
                    ):
                        print(backup.name)
        """
        params: dict[str, Any] = {"limit": limit}
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        if index_name is not None:
            path = f"/indexes/{index_name}/backups"
        else:
            path = "/backups"

        logger.info("Listing backups (path=%s)", path)
        response = await self._http.get(path, params=params)
        result = self._adapter.to_backup_list(response.content)
        logger.debug("Listed %d backups", len(result))
        return result

    async def describe(self, *, backup_id: str) -> BackupModel:
        """Get detailed information about a backup.

        Args:
            backup_id (str): The identifier of the backup to describe.

        Returns:
            A :class:`BackupModel` with full backup details.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.describe(
                        backup_id="bk-daily-20240115",
                    )
                    print(backup.status)
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Describing backup %r", backup_id)
        response = await self._http.get(f"/backups/{backup_id}")
        result = self._adapter.to_backup(response.content)
        logger.debug("Described backup %r", backup_id)
        return result

    async def get(self, *, backup_id: str) -> BackupModel:
        """Get detailed information about a backup (alias for :meth:`describe`).

        Args:
            backup_id (str): The identifier of the backup.

        Returns:
            A :class:`BackupModel` with full backup details.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    backup = await pc.backups.get(
                        backup_id="bk-daily-20240115",
                    )
                    print(backup.status)
        """
        return await self.describe(backup_id=backup_id)

    async def delete(self, *, backup_id: str) -> None:
        """Delete a backup.

        Args:
            backup_id (str): The identifier of the backup to delete.

        Raises:
            :exc:`PineconeValueError`: If *backup_id* is empty.
            :exc:`NotFoundError`: If the backup does not exist.
            :exc:`ApiError`: If the API returns another error response.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone
                async with AsyncPinecone(api_key="your-api-key") as pc:
                    await pc.backups.delete(backup_id="bk-daily-20240115")
        """
        require_non_empty("backup_id", backup_id)
        logger.info("Deleting backup %r", backup_id)
        await self._http.delete(f"/backups/{backup_id}")
        logger.debug("Deleted backup %r", backup_id)
