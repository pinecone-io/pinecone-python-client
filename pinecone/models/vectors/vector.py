"""Vector and ScoredVector response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import DictLikeStruct
from pinecone.models.vectors.sparse import SparseValues


class Vector(DictLikeStruct, Struct, rename="camel", gc=False):
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

    def __post_init__(self) -> None:
        """Require at least one of ``values`` or ``sparse_values`` to be populated."""
        if not self.values and self.sparse_values is None:
            raise ValueError("Vector must have either values or sparse_values")

    @staticmethod
    def from_dict(vector_dict: dict[str, Any]) -> Vector:
        """Construct a ``Vector`` from a plain dict representation."""
        sparse: SparseValues | None = None
        if vector_dict.get("sparse_values") is not None:
            sparse = SparseValues.from_dict(vector_dict["sparse_values"])
        return Vector(
            id=vector_dict["id"],
            values=vector_dict.get("values", []),
            sparse_values=sparse,
            metadata=vector_dict.get("metadata"),
        )

    def __repr__(self) -> str:
        if len(self.values) > 5:
            preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        parts = [
            f"id={self.id!r}",
            f"values={values_str}",
            f"sparse_values={self.sparse_values!r}",
        ]
        if self.metadata is not None:
            parts.append(f"metadata={self.metadata!r}")
        return f"Vector({', '.join(parts)})"


class ScoredVector(DictLikeStruct, Struct, rename="camel", kw_only=True, gc=False):
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

    def __repr__(self) -> str:
        if len(self.values) > 5:
            preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        parts = [
            f"id={self.id!r}",
            f"score={self.score!r}",
            f"values={values_str}",
        ]
        if self.sparse_values is not None:
            parts.append(f"sparse_values={self.sparse_values!r}")
        if self.metadata is not None:
            parts.append(f"metadata={self.metadata!r}")
        return f"ScoredVector({', '.join(parts)})"
