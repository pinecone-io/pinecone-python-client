"""Vector and ScoredVector response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models.vectors.sparse import SparseValues


class Vector(Struct, rename="camel", kw_only=True):
    """A stored vector with optional sparse values and metadata.

    Attributes:
        id (str): Unique identifier for the vector.
        values (list[float]): Dense vector values as a list of floats.
        sparse_values (SparseValues | None): Sparse vector component, or ``None`` if the vector
            has no sparse values.
        metadata (dict[str, Any] | None): User-defined metadata key-value pairs, or ``None`` if
            no metadata is attached.
    """

    id: str
    values: list[float] = []
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None


class ScoredVector(Struct, rename="camel", kw_only=True):
    """A vector match with similarity score from a query operation.

    Attributes:
        id (str): Unique identifier of the matched vector.
        score (float): Similarity score for this match.
        values (list[float]): Dense vector values, or an empty list if values were not
            requested.
        sparse_values (SparseValues | None): Sparse vector component, or ``None`` if the vector
            has no sparse values.
        metadata (dict[str, Any] | None): User-defined metadata key-value pairs, or ``None`` if
            metadata was not requested or not attached.
    """

    id: str
    score: float
    values: list[float] = []
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None
