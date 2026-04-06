"""Asynchronous data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone._internal.config import PineconeConfig
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone._internal.vector_factory import VectorFactory
from pinecone.errors.exceptions import ValidationError
from pinecone.index import _validate_host, _vector_to_dict
from pinecone.models.namespaces.models import NamespaceDescription
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchRecordsResponse
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

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

    async def upsert_records(
        self,
        *,
        records: list[dict[str, Any]],
        namespace: str,
    ) -> UpsertRecordsResponse:
        """Upsert records for indexes with integrated inference.

        Records are sent as newline-delimited JSON (NDJSON). Embeddings are
        generated server-side.

        Args:
            records: List of record dicts. Each must contain an ``_id`` or
                ``id`` field. Additional fields are passed through for
                server-side embedding.
            namespace (str): Target namespace (required).

        Returns:
            UpsertRecordsResponse with the count of records submitted.

        Raises:
            ValidationError: If records is empty or a record is missing an
                identifier field.
            ApiError: If the API returns an error response.

        Examples:

            response = await idx.upsert_records(
                namespace="my-ns",
                records=[
                    {"_id": "rec1", "text": "hello world"},
                    {"_id": "rec2", "text": "goodbye world"},
                ],
            )
            print(response.record_count)
        """
        if not records:
            raise ValidationError("records must be a non-empty list")

        for i, record in enumerate(records):
            if "_id" not in record and "id" not in record:
                raise ValidationError(f"Record at index {i} must contain an '_id' or 'id' field")

        import json as _json

        normalized: list[dict[str, Any]] = []
        for record in records:
            r = dict(record)  # shallow copy
            if "_id" not in r and "id" in r:
                r["_id"] = r.pop("id")
            normalized.append(r)

        ndjson_lines = [_json.dumps(r, separators=(",", ":")) for r in normalized]
        ndjson_body = "\n".join(ndjson_lines) + "\n"

        logger.info("Upserting %d records into namespace %r (NDJSON)", len(records), namespace)
        await self._http.post(
            f"/records/namespaces/{namespace}/upsert",
            content=ndjson_body.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        return UpsertRecordsResponse(record_count=len(records))

    async def upsert(
        self,
        *,
        vectors: Sequence[
            Vector
            | tuple[str, list[float]]
            | tuple[str, list[float], dict[str, Any]]
            | dict[str, Any]
        ],
        namespace: str = "",
    ) -> UpsertResponse:
        """Upsert a batch of vectors into a namespace.

        If a vector with the same ID already exists in the namespace, it is
        overwritten.

        Args:
            vectors: Sequence of vectors to upsert. Each element can be a
                ``Vector`` instance, a tuple of ``(id, values)`` or
                ``(id, values, metadata)``, or a dict with ``id``, ``values``,
                and optional ``sparse_values`` / ``metadata`` keys.
            namespace (str): Target namespace. Defaults to the default
                (empty-string) namespace.

        Returns:
            UpsertResponse with the count of vectors upserted.

        Raises:
            TypeError: If a vector element is not a recognized format.
            ValueError: If a vector element is malformed.
            ApiError: If the API returns an error response.

        Examples:

            response = await idx.upsert(
                vectors=[
                    Vector(id="vec1", values=[0.1, 0.2, 0.3]),
                    ("vec2", [0.4, 0.5, 0.6]),
                    {"id": "vec3", "values": [0.7, 0.8, 0.9]},
                ],
                namespace="my-ns",
            )
            print(response.upserted_count)
        """
        built = [VectorFactory.build(v) for v in vectors]
        body: dict[str, Any] = {
            "vectors": [_vector_to_dict(v) for v in built],
        }
        if namespace:
            body["namespace"] = namespace

        logger.info("Upserting %d vectors into namespace %r", len(built), namespace)
        response = await self._http.post("/vectors/upsert", json=body)
        result = self._adapter.to_upsert_response(response.content)
        logger.debug("Upserted %d vectors", result.upserted_count)
        return result

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

    async def search(
        self,
        *,
        namespace: str,
        top_k: int,
        inputs: dict[str, Any] | None = None,
        vector: list[float] | None = None,
        id: str | None = None,
        filter: dict[str, Any] | None = None,
        fields: list[str] | None = None,
        rerank: dict[str, Any] | None = None,
    ) -> SearchRecordsResponse:
        """Search records by text, vector, or ID with optional reranking.

        Searches a namespace using integrated inference (text inputs embedded
        server-side), a raw vector, or an existing record ID as the query.

        Args:
            namespace (str): Namespace to search in (required).
            top_k (int): Number of results to return (must be >= 1).
            inputs (dict[str, Any] | None): Inputs for server-side embedding
                (e.g. ``{"text": "query text"}``).
            vector (list[float] | None): Dense query vector values.
            id (str | None): ID of an existing record to use as the query.
            filter (dict[str, Any] | None): Metadata filter expression.
            fields (list[str] | None): Field names to include in results.
                When ``None``, the server returns all available fields.
            rerank (dict[str, Any] | None): Reranking configuration with
                ``model`` (required), ``rank_fields`` (required), and optional
                ``top_n``, ``parameters``, ``query`` keys.

        Returns:
            SearchRecordsResponse with hits and usage statistics.

        Raises:
            ValidationError: If ``namespace`` is not a string, ``top_k < 1``,
                or ``rerank`` is missing required keys.
            ApiError: If the API returns an error response.

        Examples:

            response = await idx.search(
                namespace="my-ns",
                top_k=10,
                inputs={"text": "semantic search query"},
            )
            for hit in response.result.hits:
                print(hit.id, hit.score)
        """
        if not isinstance(namespace, str):
            raise ValidationError("namespace must be a string")
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")
        if rerank is not None:
            if "model" not in rerank:
                raise ValidationError("rerank requires 'model' to be specified")
            if "rank_fields" not in rerank:
                raise ValidationError("rerank requires 'rank_fields' to be specified")

        query_body: dict[str, Any] = {"top_k": top_k}
        if inputs is not None:
            query_body["inputs"] = inputs
        if vector is not None:
            query_body["vector"] = vector
        if id is not None:
            query_body["id"] = id
        if filter is not None:
            query_body["filter"] = filter

        body: dict[str, Any] = {"query": query_body}
        if fields is not None:
            body["fields"] = fields
        if rerank is not None:
            body["rerank"] = rerank

        logger.info("Searching namespace %r with top_k=%d", namespace, top_k)
        response = await self._http.post(f"/records/namespaces/{namespace}/search", json=body)
        return self._adapter.to_search_response(response.content)

    async def list_paginated(
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
            ApiError: If the API returns an error response.
        """
        params: dict[str, Any] = {"namespace": namespace}
        if prefix is not None:
            params["prefix"] = prefix
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing vectors in namespace %r", namespace)
        response = await self._http.get("/vectors/list", params=params)
        return self._adapter.to_list_response(response.content)

    async def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
    ) -> AsyncIterator[ListResponse]:
        """List vector IDs in a namespace, automatically following pagination.

        Yields one ``ListResponse`` per page. The generator automatically
        follows pagination tokens until all pages have been retrieved.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return per page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Yields:
            ListResponse for each page of results.
        """
        pagination_token: str | None = None
        while True:
            page = await self.list_paginated(
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

    async def describe_index_stats(
        self,
        *,
        filter: dict[str, Any] | None = None,
    ) -> DescribeIndexStatsResponse:
        """Return statistics for this index.

        Returns aggregate statistics including total vector count,
        per-namespace vector counts, dimension, and index fullness.

        Args:
            filter (dict[str, Any] | None): Metadata filter expression. When
                provided, only vectors matching the filter are counted.

        Returns:
            DescribeIndexStatsResponse with namespace summaries, dimension,
            total vector count, and fullness metrics.

        Raises:
            ApiError: If the API returns an error response.
        """
        body: dict[str, Any] = {}
        if filter is not None:
            body["filter"] = filter

        logger.info("Describing index stats")
        response = await self._http.post("/describe_index_stats", json=body)
        return self._adapter.to_stats_response(response.content)

    async def create_namespace(
        self,
        *,
        name: str,
        schema: dict[str, Any] | None = None,
    ) -> NamespaceDescription:
        """Create a named namespace in the index.

        Args:
            name (str): Name for the new namespace (must be non-empty).
            schema (dict[str, Any] | None): Optional schema configuration
                with metadata field indexing settings.

        Returns:
            NamespaceDescription with the namespace name and record count.

        Raises:
            ValidationError: If the name is not a string or is empty/whitespace.
            ApiError: If the API returns an error response (e.g. 409 conflict
                when namespace already exists).

        Examples:

            ns = await idx.create_namespace(name="my-ns")
            print(ns.name, ns.record_count)
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        body: dict[str, Any] = {"name": name}
        if schema is not None:
            body["schema"] = schema

        logger.info("Creating namespace %r", name)
        response = await self._http.post("/namespaces", json=body)
        return self._adapter.to_namespace_description(response.content)

    async def describe_namespace(
        self,
        *,
        name: str,
    ) -> NamespaceDescription:
        """Describe a namespace by name.

        Args:
            name (str): Name of the namespace to describe.

        Returns:
            NamespaceDescription with the namespace name, record count,
            and schema information.

        Raises:
            ValidationError: If the name is not a string or is empty/whitespace.
            ApiError: If the API returns an error response.

        Examples:

            ns = await idx.describe_namespace(name="my-ns")
            print(ns.name, ns.record_count)
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        logger.info("Describing namespace %r", name)
        response = await self._http.get(f"/namespaces/{name}")
        return self._adapter.to_namespace_description(response.content)

    async def delete_namespace(
        self,
        *,
        name: str,
    ) -> None:
        """Delete a namespace by name, removing all its vectors.

        Args:
            name (str): Name of the namespace to delete.

        Returns:
            None — a successful delete returns no payload.

        Raises:
            ValidationError: If the name is not a string or is empty/whitespace.
            ApiError: If the API returns an error response.

        Examples:

            await idx.delete_namespace(name="old-data")
        """
        if not isinstance(name, str):
            raise ValidationError("namespace name must be a string")
        if not name or not name.strip():
            raise ValidationError("namespace name must be a non-empty string")

        logger.info("Deleting namespace %r", name)
        await self._http.delete(f"/namespaces/{name}")

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
