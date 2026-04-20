"""Sparse vector values model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import DictLikeStruct


class SparseValues(DictLikeStruct, Struct, rename="camel", gc=False):
    """Sparse vector representation with indices and values.

    Attributes:
        indices (list[int]): Non-zero dimension indices of the sparse vector.
        values (list[float]): Values corresponding to each index in ``indices``.
    """

    indices: list[int]
    values: list[float]

    @staticmethod
    def from_dict(sparse_values_dict: dict[str, Any]) -> SparseValues:
        """Construct a ``SparseValues`` from a plain dict representation."""
        return SparseValues(
            indices=sparse_values_dict["indices"],
            values=sparse_values_dict["values"],
        )

    def __repr__(self) -> str:
        if len(self.indices) > 5:
            idx_preview = ", ".join(repr(v) for v in self.indices[:3])
            indices_str = f"[{idx_preview}, ...{len(self.indices) - 3} more]"
        else:
            indices_str = repr(self.indices)
        if len(self.values) > 5:
            val_preview = ", ".join(repr(v) for v in self.values[:3])
            values_str = f"[{val_preview}, ...{len(self.values) - 3} more]"
        else:
            values_str = repr(self.values)
        return f"SparseValues(indices={indices_str}, values={values_str})"
