"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.search_rerank`.

Re-exports classes that used to live at :mod:`pinecone.db_data.dataclasses.search_rerank`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: missing canonical; shim carries the only definition.
# SearchRerank has no direct equivalent in the new SDK. This dataclass is defined here
# to satisfy legacy callers. A follow-up task should add a canonical implementation
# and redirect this shim to it.

from __future__ import annotations

import dataclasses
from typing import Any

from pinecone.db_data.dataclasses.utils import DictLike


@dataclasses.dataclass
class SearchRerank(DictLike):
    """Reranking configuration for a search operation.

    Attributes:
        model (str): Reranking model name (e.g. ``"bge-reranker-v2-m3"``).
        rank_fields (list[str]): Record fields to rank on (e.g. ``["text"]``).
        top_n (int | None): Number of top results after reranking, or ``None`` to use ``top_k``.
        parameters (dict[str, Any] | None): Model-specific parameters, or ``None``.
        query (str | None): Override query text for reranking, or ``None`` to infer from inputs.
    """

    model: str
    rank_fields: list[str]
    top_n: int | None = None
    parameters: dict[str, Any] | None = None
    query: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a dict of non-None field values."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }


__all__ = ["SearchRerank"]
