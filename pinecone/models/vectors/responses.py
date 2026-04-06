"""Data plane response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector


class UpsertResponse(Struct, rename="camel", kw_only=True):
    """Response from an upsert operation.

    Attributes:
        upserted_count: Number of vectors successfully upserted.
    """

    upserted_count: int

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['upserted_count'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class QueryResponse(Struct, rename="camel", kw_only=True):
    """Response from a query operation.

    Attributes:
        matches: List of scored vectors sorted by descending similarity.
        namespace: Namespace that was queried. Defaults to ``""`` (the
            default namespace).
        usage: Read unit usage for this query, or ``None`` if not reported.
    """

    matches: list[ScoredVector] = []
    namespace: str | None = ""
    usage: Usage | None = None

    def __post_init__(self) -> None:
        """Normalize null namespace to empty string (claim unified-rs-0013)."""
        if self.namespace is None:
            self.namespace = ""

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['matches'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class FetchResponse(Struct, rename="camel", kw_only=True):
    """Response from a fetch operation.

    Attributes:
        vectors: Mapping of vector ID to ``Vector`` for each fetched vector.
        namespace: Namespace the vectors were fetched from.
        usage: Read unit usage for this fetch, or ``None`` if not reported.
    """

    vectors: dict[str, Vector] = {}
    namespace: str = ""
    usage: Usage | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['vectors'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class NamespaceSummary(Struct, rename="camel", kw_only=True):
    """Summary statistics for a single namespace.

    Attributes:
        vector_count: Number of vectors in this namespace.
    """

    vector_count: int = 0


class DescribeIndexStatsResponse(Struct, rename="camel", kw_only=True):
    """Response from a describe index stats operation.

    Attributes:
        namespaces: Mapping of namespace name to ``NamespaceSummary`` for
            each namespace in the index.
        dimension: Dimensionality of vectors in the index, or ``None`` if
            not yet determined.
        index_fullness: Fraction of the index capacity used, from 0.0 to 1.0.
        total_vector_count: Total number of vectors across all namespaces.
        metric: Distance metric of the index (e.g. ``"cosine"``), or
            ``None`` if not reported.
        vector_type: Type of vectors stored (e.g. ``"dense"``), or ``None``
            if not reported.
        memory_fullness: Fraction of memory capacity used, or ``None`` if
            not reported.
        storage_fullness: Fraction of storage capacity used, or ``None``
            if not reported.
    """

    namespaces: dict[str, NamespaceSummary] = {}
    dimension: int | None = None
    index_fullness: float = 0.0
    total_vector_count: int = 0
    metric: str | None = None
    vector_type: str | None = None
    memory_fullness: float | None = None
    storage_fullness: float | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['dimension'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class ResponseInfo(Struct, kw_only=True):
    """HTTP response metadata carrier.

    Attributes:
        request_id: Server-assigned request identifier, or ``None`` if not
            present in the response.
    """

    request_id: str | None = None


class Pagination(Struct, kw_only=True):
    """Pagination token for continued listing.

    Attributes:
        next: Opaque token to pass to the next list call to retrieve the
            next page, or ``None`` if there are no more results.
    """

    next: str | None = None


class ListItem(Struct, kw_only=True):
    """A single vector ID entry in a list response.

    Attributes:
        id: The vector identifier, or ``None`` if not present.
    """

    id: str | None = None


class ListResponse(Struct, rename="camel", kw_only=True):
    """Response from a list vectors operation.

    Attributes:
        vectors: List of vector ID entries in this page.
        pagination: Pagination token for fetching the next page, or ``None``
            if there are no more results.
        namespace: Namespace the vectors were listed from.
        usage: Read unit usage for this list call, or ``None`` if not
            reported.
    """

    vectors: list[ListItem] = []
    pagination: Pagination | None = None
    namespace: str = ""
    usage: Usage | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['vectors'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class UpdateResponse(Struct, rename="camel", kw_only=True):
    """Response from an update operation.

    Attributes:
        matched_records: Number of records matched by the update, or ``None``
            if not reported by the server.
    """

    matched_records: int | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['matched_records'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None
