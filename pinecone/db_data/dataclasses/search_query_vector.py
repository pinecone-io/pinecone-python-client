"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.search_query_vector`.

Re-exports classes that used to live at
:mod:`pinecone.db_data.dataclasses.search_query_vector`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: missing canonical; shim carries the only definition.
# SearchQueryVector has no direct equivalent in the new SDK. This dataclass is defined
# here to satisfy legacy callers. A follow-up task should add a canonical implementation
# and redirect this shim to it.

from __future__ import annotations

import dataclasses
from typing import Any

from pinecone.db_data.dataclasses.utils import DictLike


@dataclasses.dataclass
class SearchQueryVector(DictLike):
    """Explicit dense/sparse query vector for search operations.

    Attributes:
        values (list[float] | None): Dense vector values, or ``None`` if not provided.
        sparse_values (list[float] | None): Sparse vector values, or ``None`` if not provided.
        sparse_indices (list[int] | None): Sparse vector indices, or ``None`` if not provided.
    """

    values: list[float] | None = None
    sparse_values: list[float] | None = None
    sparse_indices: list[int] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a dict of non-None field values."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }


__all__ = ["SearchQueryVector"]
