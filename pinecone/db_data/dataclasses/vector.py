"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.vector`.

Re-exports a dataclass-based Vector that used to live at this path before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: The canonical pinecone.models.vectors.vector.Vector is a msgspec.Struct.
# Legacy callers expect a @dataclass with DictLike inheritance, to_dict(), from_dict(),
# and a __post_init__ that validates the state. This shim carries its own definition.

from __future__ import annotations

import dataclasses
from typing import Any

from pinecone.db_data.dataclasses.sparse_values import SparseValues
from pinecone.db_data.dataclasses.utils import DictLike


@dataclasses.dataclass
class Vector(DictLike):
    """A stored vector with optional sparse values and metadata.

    Attributes:
        id (str): Unique identifier for the vector.
        values (list[float]): Dense vector values.
        sparse_values (SparseValues | None): Sparse vector component, or ``None``.
        metadata (dict[str, Any] | None): User-defined metadata, or ``None``.
    """

    id: str
    values: list[float] = dataclasses.field(default_factory=list)
    sparse_values: SparseValues | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.values and self.sparse_values is None:
            raise ValueError("Vector must have either values or sparse_values")

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this vector."""
        result: dict[str, Any] = {"id": self.id, "values": self.values}
        if self.sparse_values is not None:
            result["sparse_values"] = self.sparse_values.to_dict()
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @staticmethod
    def from_dict(vector_dict: dict[str, Any]) -> Vector:
        """Construct a Vector from a dict."""
        sparse: SparseValues | None = None
        if "sparse_values" in vector_dict and vector_dict["sparse_values"] is not None:
            sparse = SparseValues.from_dict(vector_dict["sparse_values"])
        return Vector(
            id=vector_dict["id"],
            values=vector_dict.get("values", []),
            sparse_values=sparse,
            metadata=vector_dict.get("metadata"),
        )


__all__ = ["Vector"]
