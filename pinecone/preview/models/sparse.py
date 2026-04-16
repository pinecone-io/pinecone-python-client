"""Sparse vector values model for preview document search (2026-01.alpha API)."""

from __future__ import annotations

from msgspec import Struct

__all__ = ["PreviewSparseValues"]


class PreviewSparseValues(Struct, rename="camel", gc=False):
    """Sparse vector representation for preview score-by queries.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        indices: Non-zero dimension indices of the sparse vector.
        values: Values corresponding to each index in ``indices``.
    """

    indices: list[int]
    values: list[float]
