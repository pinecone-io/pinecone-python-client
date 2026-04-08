"""Data plane response models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

from msgspec import Struct

from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector


class UpsertResponse(Struct, rename="camel", kw_only=True):
    """Response from an upsert operation.

    Attributes:
        upserted_count (int): Number of vectors successfully upserted.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    upserted_count: int
    response_info: ResponseInfo | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['upserted_count'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'upserted_count' in response``)."""
        return key in self.__struct_fields__


class QueryResponse(Struct, rename="camel", kw_only=True):
    """Response from a query operation.

    Attributes:
        matches (list[ScoredVector]): List of scored vectors sorted by descending similarity.
        namespace (str): Namespace that was queried. Defaults to ``""`` (the
            default namespace).
        usage (Usage | None): Read unit usage for this query, or ``None`` if not reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    matches: list[ScoredVector] = []
    namespace: str | None = ""
    usage: Usage | None = None
    response_info: ResponseInfo | None = None

    def __post_init__(self) -> None:
        """Normalize null namespace to empty string (claim unified-rs-0013)."""
        if self.namespace is None:
            self.namespace = ""

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['matches'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'matches' in response``)."""
        return key in self.__struct_fields__


class FetchResponse(Struct, rename="camel", kw_only=True):
    """Response from a fetch operation.

    Attributes:
        vectors (dict[str, Vector]): Mapping of vector ID to ``Vector`` for each fetched vector.
        namespace (str): Namespace the vectors were fetched from.
        usage (Usage | None): Read unit usage for this fetch, or ``None`` if not reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    vectors: dict[str, Vector] = {}
    namespace: str = ""
    usage: Usage | None = None
    response_info: ResponseInfo | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['vectors'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'vectors' in response``)."""
        return key in self.__struct_fields__


class FetchByMetadataResponse(Struct, rename="camel", kw_only=True):
    """Response from a fetch-by-metadata operation.

    Attributes:
        vectors (dict[str, Vector]): Mapping of vector ID to Vector for each fetched vector.
        namespace (str): Namespace the vectors were fetched from.
        usage (Usage | None): Read unit usage, or None if not reported.
        pagination (Pagination | None): Pagination token for the next page, or None if
            this is the last page.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    vectors: dict[str, Vector] = {}
    namespace: str = ""
    usage: Usage | None = None
    pagination: Pagination | None = None
    response_info: ResponseInfo | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator."""
        return key in self.__struct_fields__


class NamespaceSummary(Struct, rename="camel", kw_only=True):
    """Summary statistics for a single namespace.

    Attributes:
        vector_count (int): Number of vectors in this namespace.
    """

    vector_count: int = 0


class DescribeIndexStatsResponse(Struct, rename="camel", kw_only=True):
    """Response from a describe index stats operation.

    Attributes:
        namespaces (dict[str, NamespaceSummary]): Mapping of namespace name to
            ``NamespaceSummary`` for each namespace in the index.
        dimension (int | None): Dimensionality of vectors in the index, or ``None`` if
            not yet determined.
        index_fullness (float): Fraction of the index capacity used, from 0.0 to 1.0.
        total_vector_count (int): Total number of vectors across all namespaces.
        metric (str | None): Distance metric of the index (e.g. ``"cosine"``), or
            ``None`` if not reported.
        vector_type (str | None): Type of vectors stored (e.g. ``"dense"``), or ``None``
            if not reported.
        memory_fullness (float | None): Fraction of memory capacity used, or ``None`` if
            not reported.
        storage_fullness (float | None): Fraction of storage capacity used, or ``None``
            if not reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    namespaces: dict[str, NamespaceSummary] = {}
    dimension: int | None = None
    index_fullness: float = 0.0
    total_vector_count: int = 0
    metric: str | None = None
    vector_type: str | None = None
    memory_fullness: float | None = None
    storage_fullness: float | None = None
    response_info: ResponseInfo | None = None

    def __repr__(self) -> str:
        parts = []
        if self.dimension is not None:
            parts.append(f"dimension={self.dimension!r}")
        parts.append(f"total_vector_count={self.total_vector_count!r}")
        if self.metric is not None:
            parts.append(f"metric={self.metric!r}")
        parts.append(f"namespaces={len(self.namespaces)!r}")
        return f"DescribeIndexStatsResponse({', '.join(parts)})"

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['dimension'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'dimension' in response``)."""
        return key in self.__struct_fields__


class ResponseInfo(Struct, kw_only=True):
    """HTTP response metadata carrier.

    Attributes:
        request_id (str | None): Server-assigned request identifier, or ``None`` if not
            present in the response.
        lsn_reconciled (int | None): Log sequence number indicating how far the index has
            reconciled, or ``None`` if not present in the response headers.
        lsn_committed (int | None): Log sequence number of the last committed write, or
            ``None`` if not present in the response headers.
    """

    request_id: str | None = None
    lsn_reconciled: int | None = None
    lsn_committed: int | None = None

    def is_reconciled(self, target: int) -> bool:
        """Return True when the reconciled LSN meets or exceeds *target*."""
        return self.lsn_reconciled is not None and self.lsn_reconciled >= target


class Pagination(Struct, kw_only=True):
    """Pagination token for continued listing.

    Attributes:
        next (str | None): Opaque token to pass to the next list call to retrieve the
            next page, or ``None`` if there are no more results.
    """

    next: str | None = None


class ListItem(Struct, kw_only=True):
    """A single vector ID entry in a list response.

    Attributes:
        id (str | None): The vector identifier, or ``None`` if not present.
    """

    id: str | None = None


class ListResponse(Struct, rename="camel", kw_only=True):
    """Response from a list vectors operation.

    Attributes:
        vectors (list[ListItem]): List of vector ID entries in this page.
        pagination (Pagination | None): Pagination token for fetching the next page, or ``None``
            if there are no more results.
        namespace (str): Namespace the vectors were listed from.
        usage (Usage | None): Read unit usage for this list call, or ``None`` if not
            reported.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    vectors: list[ListItem] = []
    pagination: Pagination | None = None
    namespace: str = ""
    usage: Usage | None = None
    response_info: ResponseInfo | None = None

    @overload
    def __getitem__(self, key: int) -> ListItem: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | str) -> Any:
        """Support integer indexing into vectors and string bracket access.

        Args:
            key: An integer index into ``vectors``, or a string field name.

        Returns:
            The list item at the given index, or the field value.
        """
        if isinstance(key, int):
            return self.vectors[key]
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'vectors' in response``)."""
        return key in self.__struct_fields__

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self) -> Iterator[ListItem]:
        return iter(self.vectors)


class UpsertRecordsResponse(Struct, kw_only=True):
    """Response from an upsert_records operation.

    Attributes:
        record_count (int): Number of records submitted by the caller. This is a
            client-side count, not a server-confirmed count.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    record_count: int
    response_info: ResponseInfo | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['record_count'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'record_count' in response``)."""
        return key in self.__struct_fields__


class UpdateResponse(Struct, rename="camel", kw_only=True):
    """Response from an update operation.

    Attributes:
        matched_records (int | None): Number of records matched by the update, or ``None``
            if not reported by the server.
        response_info (ResponseInfo | None): HTTP response metadata (request ID, LSN values), or
            ``None`` if not populated.
    """

    matched_records: int | None = None
    response_info: ResponseInfo | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['matched_records'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'matched_records' in response``)."""
        return key in self.__struct_fields__
