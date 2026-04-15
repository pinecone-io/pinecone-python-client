"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.search_query`.

Re-exports classes that used to live at :mod:`pinecone.db_data.dataclasses.search_query`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: missing canonical; shim carries the only definition.
# SearchQuery has no direct equivalent in the new SDK. This dataclass is defined here
# to satisfy legacy callers that depended on this import path. A follow-up task should
# add a canonical implementation and redirect this shim to it.

from __future__ import annotations

import dataclasses
from typing import Any

from pinecone.db_data.dataclasses.utils import DictLike


@dataclasses.dataclass
class SearchQuery(DictLike):
    """Query parameters for a search operation.

    Attributes:
        inputs (dict[str, Any]): Search inputs (e.g. ``{"text": "hello"}``).
        top_k (int): Number of top results to return.
        filter (dict[str, Any] | None): Metadata filter to apply, or ``None`` for no filter.
        vector (dict[str, Any] | None): Explicit query vector, or ``None`` to use inputs.
        id (str | None): ID of a stored record to use as query vector, or ``None``.
        match_terms (dict[str, Any] | None): Full-text match terms, or ``None``.
    """

    inputs: dict[str, Any]
    top_k: int
    filter: dict[str, Any] | None = None
    vector: dict[str, Any] | None = None
    id: str | None = None
    match_terms: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a dict of non-None field values."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }


__all__ = ["SearchQuery"]
