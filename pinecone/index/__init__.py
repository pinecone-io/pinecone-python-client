"""Synchronous data plane client for a Pinecone index."""

from __future__ import annotations

import logging
import os
from typing import Any

from pinecone._internal.adapters.vectors_adapter import VectorsAdapter
from pinecone._internal.config import PineconeConfig, normalize_host
from pinecone._internal.constants import DATA_PLANE_API_VERSION
from pinecone.errors.exceptions import ValidationError
from pinecone.models.vectors.responses import FetchResponse, QueryResponse

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
        host: The index-specific data plane host URL.
        api_key: Pinecone API key. Falls back to ``PINECONE_API_KEY`` env var.
        additional_headers: Extra headers included in every request.
        timeout: Request timeout in seconds. Defaults to ``30.0``.

    Raises:
        ValidationError: If no API key can be resolved or the host is invalid.

    Example::

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
            top_k: Number of results to return (must be >= 1).
            vector: Dense query vector values.
            id: ID of a stored vector to use as the query.
            namespace: Namespace to query. Defaults to the default namespace.
            filter: Metadata filter expression.
            include_values: Whether to include vector values in results.
            include_metadata: Whether to include metadata in results.
            sparse_vector: Sparse query vector with indices and values.

        Returns:
            QueryResponse with matches, namespace, and usage info.

        Raises:
            ValidationError: If top_k < 1, or both/neither vector and id provided.

        Example::

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
            ids: List of vector IDs to fetch (must be non-empty).
            namespace: Namespace to fetch from. Defaults to the default namespace.

        Returns:
            FetchResponse with a map of vector IDs to Vector objects, namespace,
            and usage info. IDs that do not exist are omitted from the map rather
            than raising an error.

        Raises:
            ValidationError: If ids is empty.

        Example::

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
