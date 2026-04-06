"""Vector and ScoredVector response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models.vectors.sparse import SparseValues


class Vector(Struct, rename="camel", kw_only=True):
    """A stored vector with optional sparse values and metadata."""

    id: str
    values: list[float] = []
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None


class ScoredVector(Struct, rename="camel", kw_only=True):
    """A vector match with similarity score from a query operation."""

    id: str
    score: float
    values: list[float] = []
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None
