"""Synchronous data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone._internal.config import PineconeConfig, normalize_host
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import (
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
)

logger = logging.getLogger(__name__)


def _validate_host(host: str) -> str:
    """Validate and normalize an index host URL.

    Raises:
        ValidationError: If the host is empty or does not look like a real hostname.
    """
    if not host or not host.strip():
        raise ValidationError("host must be a non-empty string")
    normalized = normalize_host(host.strip())
    # Strip scheme for the dot/localhost check
    bare = normalized
    for prefix in ("https://", "http://"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    if "." not in bare and "localhost" not in bare.lower():
        raise ValidationError(
            f"host {host!r} does not appear to be a valid URL (must contain a dot or 'localhost')"
        )
    return normalized


class Index:
    """Synchronous data plane client targeting a specific Pinecone index.

    Can be constructed directly with a host URL, or via the
    :meth:`Pinecone.index` factory method.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        additional_headers (dict[str, str] | None): Extra headers included in every request.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.

    Raises:
        ValidationError: If no API key can be resolved or the host is invalid.

    Examples:

        from pinecone import Index

        idx = Index(host="my-index-abc123.svc.pinecone.io", api_key="...")
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None = None,
        additional_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        # Resolve API key: explicit arg > env var (check BEFORE host per unified-ord-0001)
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._host = _validate_host(host)

        config = PineconeConfig(
            api_key=resolved_key,
            host=self._host,
            timeout=timeout,
            additional_headers=additional_headers or {},
        )
        self._config = config

        from pinecone._internal.http_client import HTTPClient

        self._http = HTTPClient(config, DATA_PLANE_API_VERSION)
        self._adapter = VectorsAdapter()

        logger.info("Index client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    def query(
        self,
        *,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """Query a namespace for the nearest neighbors of a vector.

        Args:
            top_k (int): Number of results to return (must be >= 1).
            vector (list[float] | None): Dense query vector values.
            id (str | None): ID of a stored vector to use as the query.
            namespace (str): Namespace to query. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression.
            include_values (bool): Whether to include vector values in results.
            include_metadata (bool): Whether to include metadata in results.
            sparse_vector (dict[str, Any] | None): Sparse query vector with indices and values.

        Returns:
            QueryResponse with matches, namespace, and usage info.

        Raises:
            ValidationError: If top_k < 1, or both/neither vector and id provided.

        Examples:

            response = idx.query(top_k=10, vector=[0.1, 0.2, 0.3])
            for match in response.matches:
                print(match.id, match.score)
        """
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")

        has_vector = vector is not None
        has_id = id is not None
        if has_vector and has_id:
            raise ValidationError("Exactly one of vector or id must be provided, not both")
        if not has_vector and not has_id:
            raise ValidationError("Exactly one of vector or id must be provided, got neither")

        body: dict[str, Any] = {
            "topK": top_k,
            "includeValues": include_values,
            "includeMetadata": include_metadata,
        }
        if namespace:
            body["namespace"] = namespace
        if vector is not None:
            body["vector"] = vector
        if id is not None:
            body["id"] = id
        if filter is not None:
            body["filter"] = filter
        if sparse_vector is not None:
            body["sparseVector"] = sparse_vector

        logger.info("Querying index with top_k=%d", top_k)
        response = self._http.post("/query", json=body)
        result = self._adapter.to_query_response(response.content)
        logger.debug("Query returned %d matches", len(result.matches))
        return result

    def fetch(
        self,
        *,
        ids: list[str],
        namespace: str = "",
    ) -> FetchResponse:
        """Fetch vectors by their IDs from a namespace.

        Args:
            ids (list[str]): List of vector IDs to fetch (must be non-empty).
            namespace (str): Namespace to fetch from. Defaults to the default namespace.

        Returns:
            FetchResponse with a map of vector IDs to Vector objects, namespace,
            and usage info. IDs that do not exist are omitted from the map rather
            than raising an error.

        Raises:
            ValidationError: If ids is empty.

        Examples:

            response = idx.fetch(ids=["vec1", "vec2"])
            for vid, vec in response.vectors.items():
                print(vid, vec.values)
        """
        if not ids:
            raise ValidationError("ids must be a non-empty list")

        params: dict[str, Any] = {"ids": ids}
        if namespace:
            params["namespace"] = namespace

        logger.info("Fetching %d vectors", len(ids))
        response = self._http.get("/vectors/fetch", params=params)
        result = self._adapter.to_fetch_response(response.content)
        logger.debug("Fetched %d vectors", len(result.vectors))
        return result

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        delete_all: bool = False,
        filter: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> None:
        """Delete vectors from a namespace by ID, filter, or delete-all flag.

        Exactly one of ``ids``, ``delete_all``, or ``filter`` must be specified.
        Deleting IDs that do not exist does not raise an error.

        Args:
            ids (list[str] | None): List of vector IDs to delete.
            delete_all (bool): If True, delete all vectors in the namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to delete.
            namespace (str): Namespace to delete from. Defaults to the default namespace.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            ValidationError: If zero or more than one deletion mode is specified.

        Examples:

            # Delete by IDs
            idx.delete(ids=["vec1", "vec2"])

            # Delete all vectors in a namespace
            idx.delete(delete_all=True, namespace="old-data")

            # Delete by metadata filter
            idx.delete(filter={"category": {"$eq": "obsolete"}})
        """
        mode_count = sum([ids is not None, delete_all, filter is not None])
        if mode_count == 0:
            raise ValidationError("Must specify one of ids, delete_all, or filter")
        if mode_count > 1:
            raise ValidationError(
                "Cannot combine ids, delete_all, and filter — specify exactly one"
            )

        body: dict[str, Any] = {"namespace": namespace}
        if ids is not None:
            body["ids"] = ids
        if delete_all:
            body["deleteAll"] = True
        if filter is not None:
            body["filter"] = filter

        logger.info("Deleting vectors from namespace %r", namespace)
        self._http.post("/vectors/delete", json=body)

    def update(
        self,
        *,
        id: str | None = None,
        values: list[float] | None = None,
        sparse_values: dict[str, Any] | None = None,
        set_metadata: dict[str, Any] | None = None,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> UpdateResponse:
        """Update vectors by ID or metadata filter.

        Updates a single vector's dense values, sparse values, or metadata by
        identifier, or bulk-updates metadata on all vectors matching a filter.

        Exactly one of ``id`` or ``filter`` must be specified.

        Args:
            id (str | None): ID of the vector to update.
            values (list[float] | None): New dense vector values.
            sparse_values (dict[str, Any] | None): New sparse vector with ``indices``
                and ``values`` keys.
            set_metadata (dict[str, Any] | None): Metadata fields to set or overwrite.
            namespace (str): Namespace to target. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to update.
            dry_run (bool): If True, return the count of records that would be
                affected without applying changes. Only applies to filter-based
                updates.

        Returns:
            UpdateResponse with matched_records count (when available).

        Raises:
            ValidationError: If both or neither of id and filter are provided.

        Examples:

            # Update by ID
            idx.update(id="vec1", values=[0.1, 0.2, 0.3])

            # Bulk-update metadata by filter
            idx.update(
                filter={"genre": {"$eq": "drama"}},
                set_metadata={"year": 2020},
            )
        """
        has_id = id is not None
        has_filter = filter is not None
        if has_id and has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, not both")
        if not has_id and not has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, got neither")

        body: dict[str, Any] = {"namespace": namespace}
        if id is not None:
            body["id"] = id
        if values is not None:
            body["values"] = values
        if sparse_values is not None:
            body["sparseValues"] = sparse_values
        if set_metadata is not None:
            body["setMetadata"] = set_metadata
        if filter is not None:
            body["filter"] = filter
        if dry_run:
            body["dryRun"] = True

        logger.info("Updating vectors in namespace %r", namespace)
        response = self._http.post("/vectors/update", json=body)
        return self._adapter.to_update_response(response.content)

    def list_paginated(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str = "",
    ) -> ListResponse:
        """Fetch a single page of vector IDs from a namespace.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return in this page.
            pagination_token (str | None): Token from a previous response to fetch the next page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Returns:
            ListResponse with vector IDs, pagination info, namespace, and usage.

        Raises:
            ValidationError: If inputs are invalid.

        Examples:

            response = idx.list_paginated(prefix="doc1#", limit=50)
            for item in response.vectors:
                print(item.id)
        """
        params: dict[str, Any] = {"namespace": namespace}
        if prefix is not None:
            params["prefix"] = prefix
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing vectors in namespace %r", namespace)
        response = self._http.get("/vectors/list", params=params)
        return self._adapter.to_list_response(response.content)

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
    ) -> Iterator[ListResponse]:
        """List vector IDs in a namespace, automatically following pagination.

        Yields one ``ListResponse`` per page. The generator automatically
        follows pagination tokens until all pages have been retrieved.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return per page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Yields:
            ListResponse for each page of results.

        Examples:

            for page in idx.list(prefix="doc1#"):
                for item in page.vectors:
                    print(item.id)
        """
        pagination_token: str | None = None
        while True:
            page = self.list_paginated(
                prefix=prefix,
                limit=limit,
                pagination_token=pagination_token,
                namespace=namespace,
            )
            yield page
            if page.pagination is not None and page.pagination.next is not None:
                pagination_token = page.pagination.next
            else:
                break

    def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        self._http.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"Index(host='{self._host}')"
