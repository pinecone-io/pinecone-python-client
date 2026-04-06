"""Sparse vector values model."""

from __future__ import annotations

from msgspec import Struct


class SparseValues(Struct, rename="camel", kw_only=True):
    """Sparse vector representation with indices and values.

    Attributes:
        indices: Non-zero dimension indices of the sparse vector.
        values: Values corresponding to each index in ``indices``.
    """

    indices: list[int]
    values: list[float]
