"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.sparse_values`.

Re-exports a dataclass-based SparseValues that used to live at this path before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: The canonical pinecone.models.vectors.sparse.SparseValues is a msgspec.Struct.
# Legacy callers expect a @dataclass with DictLike inheritance, to_dict(), and from_dict().
# This shim carries its own dataclass definition to satisfy that contract.

from __future__ import annotations

import dataclasses
from typing import Any

from pinecone.db_data.dataclasses.utils import DictLike


@dataclasses.dataclass
class SparseValues(DictLike):
    """Sparse vector representation with indices and values.

    Attributes:
        indices (list[int]): Non-zero dimension indices of the sparse vector.
        values (list[float]): Values corresponding to each index in ``indices``.
    """

    indices: list[int]
    values: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this sparse vector."""
        return {"indices": self.indices, "values": self.values}

    @staticmethod
    def from_dict(sparse_values_dict: dict[str, Any]) -> SparseValues:
        """Construct a SparseValues from a dict."""
        return SparseValues(
            indices=sparse_values_dict["indices"],
            values=sparse_values_dict["values"],
        )


__all__ = ["SparseValues"]
