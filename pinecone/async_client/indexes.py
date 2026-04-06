"""Async Indexes namespace — list, describe, and exists operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pinecone._internal.adapters.indexes_adapter import IndexesAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.errors.exceptions import NotFoundError
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncIndexes:
    """Async control-plane operations for Pinecone indexes.

    Provides methods to list, describe, and check existence of indexes.

    Args:
        http: Async HTTP client for making API requests.

    Example::

        from pinecone import AsyncPinecone

        async with AsyncPinecone(api_key="your-api-key") as pc:
            for idx in await pc.indexes.list():
                print(idx.name)
    """

    def __init__(self, http: AsyncHTTPClient) -> None:
        self._http = http
        self._adapter = IndexesAdapter()
        self._host_cache: dict[str, str] = {}

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncIndexes()"

    async def list(self) -> IndexList:
        """List all indexes in the project.

        Returns all indexes in a single response without filtering,
        sorting, or pagination.

        Returns:
            An IndexList supporting iteration, len(), index access,
            and a names() convenience method.

        Example::

            async with AsyncPinecone(api_key="your-api-key") as pc:
                indexes = await pc.indexes.list()
                print(indexes.names())
        """
        logger.info("Listing indexes")
        response = await self._http.get("/indexes")
        result = self._adapter.to_index_list(response.json())
        logger.debug("Listed %d indexes", len(result))
        return result

    async def describe(self, name: str) -> IndexModel:
        """Get detailed information about a named index.

        After a successful call the host URL is cached internally for
        later data-plane client construction.

        Args:
            name: The name of the index to describe.

        Returns:
            An IndexModel with name, dimension, metric, host, spec,
            status, deletion_protection, vector_type, and tags.

        Raises:
            ValidationError: If *name* is empty.
            NotFoundError: If the index does not exist.

        Example::

            async with AsyncPinecone(api_key="your-api-key") as pc:
                desc = await pc.indexes.describe("my-index")
                print(desc.host)
        """
        require_non_empty("name", name)
        logger.info("Describing index %r", name)
        response = await self._http.get(f"/indexes/{name}")
        model = self._adapter.to_index_model(response.json())
        self._host_cache[name] = model.host
        logger.debug("Described index %r (host=%s)", name, model.host)
        return model

    async def exists(self, name: str) -> bool:
        """Check whether a named index exists.

        Uses describe internally; returns ``True`` on success and
        ``False`` when a 404 is returned.

        Args:
            name: The name of the index to check.

        Returns:
            True if the index exists, False otherwise.

        Raises:
            ValidationError: If *name* is empty.

        Example::

            async with AsyncPinecone(api_key="your-api-key") as pc:
                if await pc.indexes.exists("my-index"):
                    print("Index found")
        """
        require_non_empty("name", name)
        try:
            await self.describe(name)
            return True
        except NotFoundError:
            return False
