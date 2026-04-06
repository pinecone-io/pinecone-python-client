"""Sparse vector values model."""

from __future__ import annotations

from msgspec import Struct


class SparseValues(Struct, kw_only=True):
    """Sparse vector representation with indices and values."""

    indices: list[int]
    values: list[float]
