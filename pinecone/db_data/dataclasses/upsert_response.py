"""Backwards-compatibility shim for :mod:`pinecone.models.vectors.responses`.

Re-exports the canonical ``UpsertResponse`` that used to live at this path
before the ``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers
working. New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.vectors.responses import UpsertResponse

__all__ = ["UpsertResponse"]
