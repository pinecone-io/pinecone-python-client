"""Asynchronous data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from typing import Any

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.errors.exceptions import ValidationError
from pinecone.index import _validate_host
from pinecone.models.vectors.responses import FetchResponse, QueryResponse, UpdateResponse
from pinecone.models.vectors.sparse import SparseValues

logger = logging.getLogger(__name__)


class AsyncIndex:
    """Asynchronous data plane client targeting a specific Pinecone index.

    Can be constructed directly with a host URL, or via the
    :meth:`AsyncPinecone.index` factory method.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        additional_headers (dict[str, str] | None): Extra headers included in every request.
        timeout (float): Request timeout in seconds. Defaults to ``30.0``.

    Raises:
        ValidationError: If no API key can be resolved or the host is invalid.

    Examples:

        from pinecone import AsyncIndex

        async with AsyncIndex(host="my-index-abc123.svc.pinecone.io", api_key="...") as idx:
            print(idx.host)
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

        from pinecone._internal.http_client import AsyncHTTPClient

        self._http = AsyncHTTPClient(config, DATA_PLANE_API_VERSION)
        self._adapter = VectorsAdapter()

        logger.info("AsyncIndex client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    async def query(
        self,
        *,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        sparse_vector: SparseValues | dict[str, Any] | None = None,
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
            sparse_vector (SparseValues | dict[str, Any] | None): Sparse query vector
                with indices and values.

        Returns:
            QueryResponse with matches, namespace, and usage info.

        Raises:
            ValidationError: If top_k < 1, or both/neither vector and id provided.
            ApiError: If the API returns an error response.
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
            if isinstance(sparse_vector, SparseValues):
                body["sparseVector"] = {
                    "indices": sparse_vector.indices,
                    "values": sparse_vector.values,
                }
            else:
                body["sparseVector"] = sparse_vector

        logger.info("Querying index with top_k=%d", top_k)
        response = await self._http.post("/query", json=body)
        result = self._adapter.to_query_response(response.content)
        logger.debug("Query returned %d matches", len(result.matches))
        return result

    async def fetch(
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
            ApiError: If the API returns an error response.
        """
        if not ids:
            raise ValidationError("ids must be a non-empty list")

        params: dict[str, Any] = {"ids": ids}
        if namespace:
            params["namespace"] = namespace

        logger.info("Fetching %d vectors", len(ids))
        response = await self._http.get("/vectors/fetch", params=params)
        result = self._adapter.to_fetch_response(response.content)
        logger.debug("Fetched %d vectors", len(result.vectors))
        return result

    async def delete(
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
            ApiError: If the API returns an error response.
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
        await self._http.post("/vectors/delete", json=body)

    async def update(
        self,
        *,
        id: str | None = None,
        values: list[float] | None = None,
        sparse_values: SparseValues | dict[str, Any] | None = None,
        set_metadata: dict[str, Any] | None = None,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> UpdateResponse:
        """Update vectors by ID or metadata filter.

        Exactly one of ``id`` or ``filter`` must be specified.

        Args:
            id (str | None): ID of the vector to update.
            values (list[float] | None): New dense vector values.
            sparse_values (SparseValues | dict[str, Any] | None): New sparse vector.
            set_metadata (dict[str, Any] | None): Metadata fields to set or overwrite.
            namespace (str): Namespace to target. Defaults to the default namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to update.
            dry_run (bool): If True, return the count of records that would be
                affected without applying changes.

        Returns:
            UpdateResponse with matched_records count (when available).

        Raises:
            ValidationError: If both or neither of id and filter are provided.
            ApiError: If the API returns an error response.
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
            if isinstance(sparse_values, SparseValues):
                body["sparseValues"] = {
                    "indices": sparse_values.indices,
                    "values": sparse_values.values,
                }
            else:
                body["sparseValues"] = sparse_values
        if set_metadata is not None:
            body["setMetadata"] = set_metadata
        if filter is not None:
            body["filter"] = filter
        if dry_run:
            body["dryRun"] = True

        logger.info("Updating vectors in namespace %r", namespace)
        response = await self._http.post("/vectors/update", json=body)
        return self._adapter.to_update_response(response.content)

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        await self._http.close()

    async def __aenter__(self) -> AsyncIndex:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"AsyncIndex(host='{self._host}')"
