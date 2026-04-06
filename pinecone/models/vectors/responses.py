"""Data plane response models."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector


class UpsertResponse(Struct, rename="camel", kw_only=True):
    """Response from an upsert operation."""

    upserted_count: int


class QueryResponse(Struct, rename="camel", kw_only=True):
    """Response from a query operation."""

    matches: list[ScoredVector] = []
    namespace: str = ""
    usage: Usage | None = None


class FetchResponse(Struct, rename="camel", kw_only=True):
    """Response from a fetch operation."""

    vectors: dict[str, Vector] = {}
    namespace: str = ""
    usage: Usage | None = None


class NamespaceSummary(Struct, rename="camel", kw_only=True):
    """Summary statistics for a single namespace."""

    vector_count: int = 0


class DescribeIndexStatsResponse(Struct, rename="camel", kw_only=True):
    """Response from a describe index stats operation."""

    namespaces: dict[str, NamespaceSummary] = {}
    dimension: int | None = None
    index_fullness: float = 0.0
    total_vector_count: int = 0
    metric: str | None = None
    vector_type: str | None = None
    memory_fullness: float | None = None
    storage_fullness: float | None = None


class ResponseInfo(Struct, kw_only=True):
    """HTTP response metadata carrier."""

    request_id: str | None = None


class Pagination(Struct, kw_only=True):
    """Pagination token for continued listing."""

    next: str | None = None


class ListItem(Struct, kw_only=True):
    """A single vector ID entry in a list response."""

    id: str | None = None


class ListResponse(Struct, rename="camel", kw_only=True):
    """Response from a list vectors operation."""

    vectors: list[ListItem] = []
    pagination: Pagination | None = None
    namespace: str = ""
    usage: Usage | None = None


class UpdateResponse(Struct, rename="camel", kw_only=True):
    """Response from an update operation."""

    matched_records: int | None = None
