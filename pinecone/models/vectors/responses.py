"""Data plane response models."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector


class UpsertResponse(Struct, kw_only=True):
    """Response from an upsert operation."""

    upserted_count: int


class QueryResponse(Struct, kw_only=True):
    """Response from a query operation."""

    matches: list[ScoredVector] = []
    namespace: str = ""
    usage: Usage | None = None


class FetchResponse(Struct, kw_only=True):
    """Response from a fetch operation."""

    vectors: dict[str, Vector] = {}
    namespace: str = ""
    usage: Usage | None = None


class NamespaceSummary(Struct, kw_only=True):
    """Summary statistics for a single namespace."""

    vector_count: int = 0


class DescribeIndexStatsResponse(Struct, kw_only=True):
    """Response from a describe index stats operation."""

    namespaces: dict[str, NamespaceSummary] = {}
    dimension: int | None = None
    index_fullness: float = 0.0
    total_vector_count: int = 0


class ResponseInfo(Struct, kw_only=True):
    """HTTP response metadata carrier."""

    request_id: str | None = None
