"""Backwards-compatibility shim for :mod:`pinecone.models.indexes.specs`.

Re-exports the canonical ``ByocSpec`` that used to live at this path before
the ``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.indexes.specs import ByocSpec

__all__ = ["ByocSpec"]
