"""Backwards-compatibility shim for :mod:`pinecone.models.vectors.sparse`.

Re-exports the canonical ``SparseValues`` class that used to live at this
path before the ``python-sdk2`` rewrite. Preserved to keep pre-rewrite
callers working. New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.vectors.sparse import SparseValues

__all__ = ["SparseValues"]
