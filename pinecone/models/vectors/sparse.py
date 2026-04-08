"""Sparse vector values model."""

from __future__ import annotations

from msgspec import Struct


class SparseValues(Struct, rename="camel", kw_only=True):
    """Sparse vector representation with indices and values.

    Attributes:
        indices (list[int]): Non-zero dimension indices of the sparse vector.
        values (list[float]): Values corresponding to each index in ``indices``.
    """

    indices: list[int]
    values: list[float]

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
