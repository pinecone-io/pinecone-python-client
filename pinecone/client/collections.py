"""Collections namespace — create, list, describe, and delete operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pinecone._internal.adapters.collections_adapter import CollectionsAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class Collections:
    """Control-plane operations for Pinecone collections.

    Provides methods to create, list, describe, and delete collections.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Example::

        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        for col in pc.collections.list():
            print(col.name)
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http
        self._adapter = CollectionsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "Collections()"

    def create(self, *, name: str, source: str) -> CollectionModel:
        """Create a collection from an existing index.

        Returns immediately after the API call without polling for
        readiness.

        Args:
            name (str): Name for the new collection.
            source (str): Name of the source index.

        Returns:
            A CollectionModel describing the created collection.

        Raises:
            ValidationError: If *name* or *source* is empty.

        Example::

            col = pc.collections.create(name="my-collection", source="my-index")
            print(col.status)
        """
        require_non_empty("name", name)
        require_non_empty("source", source)
        logger.info("Creating collection %r from source %r", name, source)
        response = self._http.post("/collections", json={"name": name, "source": source})
        result = self._adapter.to_collection(response.content)
        logger.debug("Created collection %r", name)
        return result

    def list(self) -> CollectionList:
        """List all collections in the project.

        Returns all collections in a single response without filtering,
        sorting, or pagination.

        Returns:
            A CollectionList supporting iteration, len(), index access,
            and a names() convenience method.

        Example::

            collections = pc.collections.list()
            print(collections.names())
            for col in collections:
                print(col.name, col.status)
        """
        logger.info("Listing collections")
        response = self._http.get("/collections")
        result = self._adapter.to_collection_list(response.content)
        logger.debug("Listed %d collections", len(result))
        return result

    def describe(self, name: str) -> CollectionModel:
        """Get detailed information about a named collection.

        Args:
            name (str): The name of the collection to describe.

        Returns:
            A CollectionModel with name, status, size, dimension,
            vector_count, and environment.

        Raises:
            ValidationError: If *name* is empty.
            NotFoundError: If the collection does not exist.

        Example::

            desc = pc.collections.describe("my-collection")
            print(desc.size)
        """
        require_non_empty("name", name)
        logger.info("Describing collection %r", name)
        response = self._http.get(f"/collections/{name}")
        result = self._adapter.to_collection(response.content)
        logger.debug("Described collection %r", name)
        return result

    def delete(self, name: str) -> None:
        """Delete a collection by name.

        Args:
            name (str): The name of the collection to delete.

        Raises:
            ValidationError: If *name* is empty.
            NotFoundError: If the collection does not exist.

        Example::

            pc.collections.delete("my-collection")
        """
        require_non_empty("name", name)
        logger.info("Deleting collection %r", name)
        self._http.delete(f"/collections/{name}")
        logger.debug("Deleted collection %r", name)
