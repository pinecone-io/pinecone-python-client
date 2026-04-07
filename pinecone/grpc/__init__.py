"""Synchronous gRPC data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any

from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone._internal.data_plane_helpers import _validate_host
from pinecone._internal.vector_factory import VectorFactory
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListItem,
    ListResponse,
    NamespaceSummary,
    Pagination,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector

logger = logging.getLogger(__name__)


def _build_grpc_endpoint(host: str, secure: bool) -> str:
    """Build a gRPC endpoint URL from a host string.

    Strips any existing scheme and applies the correct one for gRPC.
    """
    bare = host
    for prefix in ("https://", "http://"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break

    scheme = "https" if secure else "http"
    return f"{scheme}://{bare}"


def _vector_to_grpc_dict(v: Vector) -> dict[str, Any]:
    """Serialize a Vector to a dict matching GrpcChannel's expected input format."""
    d: dict[str, Any] = {"id": v.id, "values": v.values}
    if v.sparse_values is not None:
        d["sparse_values"] = {
            "indices": v.sparse_values.indices,
            "values": v.sparse_values.values,
        }
    if v.metadata is not None:
        d["metadata"] = v.metadata
    return d


def _dict_to_vector(vid: str, data: dict[str, Any]) -> Vector:
    """Convert a GrpcChannel vector dict to a Vector model."""
    sparse = None
    sv = data.get("sparse_values")
    if sv is not None:
        sparse = SparseValues(indices=sv["indices"], values=sv["values"])
    return Vector(
        id=vid,
        values=data.get("values", []),
        sparse_values=sparse,
        metadata=data.get("metadata"),
    )


def _dict_to_scored_vector(data: dict[str, Any]) -> ScoredVector:
    """Convert a GrpcChannel scored vector dict to a ScoredVector model."""
    sparse = None
    sv = data.get("sparse_values")
    if sv is not None:
        sparse = SparseValues(indices=sv["indices"], values=sv["values"])
    return ScoredVector(
        id=data["id"],
        score=data.get("score", 0.0),
        values=data.get("values", []),
        sparse_values=sparse,
        metadata=data.get("metadata"),
    )


def _dict_to_usage(data: dict[str, Any] | None) -> Usage | None:
    """Convert a usage dict to a Usage model, or None."""
    if data is None:
        return None
    return Usage(read_units=data.get("read_units", 0))


class GrpcIndex:
    """Synchronous gRPC data plane client targeting a specific Pinecone index.

    Provides the same interface as :class:`~pinecone.index.Index` but routes
    data-plane operations through a gRPC transport (via the Rust-backed
    :class:`~pinecone._grpc.GrpcChannel`) instead of HTTP/REST.

    Args:
        host (str): The index-specific data plane host URL.
        api_key (str | None): Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        api_version (str): API version string. Defaults to the current data plane version.
        source_tag (str | None): Tag appended to the User-Agent string for request attribution.
        secure (bool): Whether to use TLS encryption. Defaults to ``True``.
        timeout (float): Request timeout in seconds. Defaults to ``20.0``.
        connect_timeout (float): Connection timeout in seconds. Defaults to ``1.0``.

    Raises:
        ValidationError: If no API key can be resolved or the host is invalid.

    Examples:

        from pinecone.grpc import GrpcIndex

        idx = GrpcIndex(host="movie-recs-abc123.svc.pinecone.io", api_key="...")
    """

    def __init__(
        self,
        *,
        host: str,
        api_key: str | None = None,
        api_version: str = DATA_PLANE_API_VERSION,
        source_tag: str | None = None,
        secure: bool = True,
        timeout: float = 20.0,
        connect_timeout: float = 1.0,
    ) -> None:
        # Resolve API key: explicit arg > env var
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not resolved_key:
            raise ValidationError(
                "No API key provided. Pass api_key='...' or set the "
                "PINECONE_API_KEY environment variable."
            )

        # Validate and normalize host
        self._host = _validate_host(host)
        self._source_tag = source_tag

        # Build gRPC endpoint and create the Rust-backed channel
        endpoint = _build_grpc_endpoint(self._host, secure)

        from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]

        self._channel = GrpcChannel(
            endpoint,
            resolved_key,
            api_version,
            secure,
            timeout,
            connect_timeout,
        )

        logger.info("GrpcIndex client created for host %s", self._host)

    @property
    def host(self) -> str:
        """The data plane host URL for this index."""
        return self._host

    def upsert(
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
        """
        built = [VectorFactory.build(v) for v in vectors]
        grpc_vectors = [_vector_to_grpc_dict(v) for v in built]

        logger.info("Upserting %d vectors via gRPC into namespace %r", len(built), namespace)
        result = self._channel.upsert(grpc_vectors, namespace or None)
        return UpsertResponse(upserted_count=result.get("upserted_count", 0))

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

        Returns:
            QueryResponse with matches, namespace, and usage info.

        Raises:
            ValidationError: If top_k < 1, or both/neither vector and id provided.
        """
        if top_k < 1:
            raise ValidationError(f"top_k must be a positive integer, got {top_k}")

        has_vector = vector is not None
        has_id = id is not None
        if has_vector and has_id:
            raise ValidationError("Exactly one of vector or id must be provided, not both")
        if not has_vector and not has_id:
            raise ValidationError("Exactly one of vector or id must be provided, got neither")

        logger.info("Querying index via gRPC with top_k=%d", top_k)
        result = self._channel.query(
            top_k,
            vector=vector,
            id=id,
            namespace=namespace or None,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )

        matches = [_dict_to_scored_vector(m) for m in result.get("matches", [])]
        usage = _dict_to_usage(result.get("usage"))
        return QueryResponse(
            matches=matches,
            namespace=result.get("namespace", ""),
            usage=usage,
        )

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
            and usage info.

        Raises:
            ValidationError: If ids is empty.
        """
        if not ids:
            raise ValidationError("ids must be a non-empty list")

        logger.info("Fetching %d vectors via gRPC", len(ids))
        result = self._channel.fetch(ids, namespace=namespace or None)

        vectors: dict[str, Vector] = {}
        for vid, vdata in result.get("vectors", {}).items():
            vectors[vid] = _dict_to_vector(vid, vdata)

        usage = _dict_to_usage(result.get("usage"))
        return FetchResponse(
            vectors=vectors,
            namespace=result.get("namespace", ""),
            usage=usage,
        )

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

        Args:
            ids (list[str] | None): List of vector IDs to delete.
            delete_all (bool): If True, delete all vectors in the namespace.
            filter (dict[str, Any] | None): Metadata filter expression selecting vectors to delete.
            namespace (str): Namespace to delete from. Defaults to the default namespace.

        Raises:
            ValidationError: If zero or more than one deletion mode is specified.
        """
        mode_count = sum([ids is not None, delete_all, filter is not None])
        if mode_count == 0:
            raise ValidationError("Must specify one of ids, delete_all, or filter")
        if mode_count > 1:
            raise ValidationError(
                "Cannot combine ids, delete_all, and filter — specify exactly one"
            )

        logger.info("Deleting vectors via gRPC from namespace %r", namespace)
        self._channel.delete(
            ids=ids,
            delete_all=delete_all,
            namespace=namespace or None,
            filter=filter,
        )

    def update(
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
        """
        has_id = id is not None
        has_filter = filter is not None
        if has_id and has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, not both")
        if not has_id and not has_filter:
            raise ValidationError("Exactly one of id or filter must be provided, got neither")

        # Convert SparseValues model to dict for GrpcChannel
        sv_dict: dict[str, Any] | None = None
        if sparse_values is not None:
            if isinstance(sparse_values, SparseValues):
                sv_dict = {
                    "indices": sparse_values.indices,
                    "values": sparse_values.values,
                }
            else:
                sv_dict = sparse_values

        logger.info("Updating vectors via gRPC in namespace %r", namespace)
        result = self._channel.update(
            id or "",
            values=values,
            sparse_values=sv_dict,
            set_metadata=set_metadata,
            namespace=namespace or None,
            filter=filter,
            dry_run=dry_run if dry_run else None,
        )

        return UpdateResponse(matched_records=result.get("matched_records"))

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
        """
        logger.info("Listing vectors via gRPC in namespace %r", namespace)
        result = self._channel.list(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            namespace=namespace or None,
        )

        vectors = [ListItem(id=v.get("id")) for v in result.get("vectors", [])]
        pagination_data = result.get("pagination")
        pagination = None
        if pagination_data is not None:
            pagination = Pagination(next=pagination_data.get("next"))
        usage = _dict_to_usage(result.get("usage"))

        return ListResponse(
            vectors=vectors,
            pagination=pagination,
            namespace=result.get("namespace", ""),
            usage=usage,
        )

    def list(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str = "",
    ) -> Iterator[ListResponse]:
        """List vector IDs in a namespace, automatically following pagination.

        Yields one ``ListResponse`` per page.

        Args:
            prefix (str | None): Return only IDs starting with this prefix.
            limit (int | None): Maximum number of IDs to return per page.
            namespace (str): Namespace to list from. Defaults to the default namespace.

        Yields:
            ListResponse for each page of results.
        """
        pagination_token: str | None = None
        while True:
            page = self.list_paginated(
                prefix=prefix,
                limit=limit,
                pagination_token=pagination_token,
                namespace=namespace,
            )
            if page.vectors:
                yield page
            if page.pagination is not None and page.pagination.next is not None:
                pagination_token = page.pagination.next
            else:
                break

    def describe_index_stats(
        self,
        *,
        filter: dict[str, Any] | None = None,
    ) -> DescribeIndexStatsResponse:
        """Return statistics for this index.

        Args:
            filter (dict[str, Any] | None): Metadata filter expression. When
                provided, only vectors matching the filter are counted.

        Returns:
            DescribeIndexStatsResponse with namespace summaries, dimension,
            total vector count, and fullness metrics.
        """
        logger.info("Describing index stats via gRPC")
        result = self._channel.describe_index_stats(filter=filter)

        namespaces: dict[str, NamespaceSummary] = {}
        for ns_name, ns_data in result.get("namespaces", {}).items():
            namespaces[ns_name] = NamespaceSummary(
                vector_count=ns_data.get("vector_count", 0),
            )

        return DescribeIndexStatsResponse(
            namespaces=namespaces,
            dimension=result.get("dimension"),
            index_fullness=result.get("index_fullness", 0.0),
            total_vector_count=result.get("total_vector_count", 0),
            metric=result.get("metric"),
            vector_type=result.get("vector_type"),
            memory_fullness=result.get("memory_fullness"),
            storage_fullness=result.get("storage_fullness"),
        )
