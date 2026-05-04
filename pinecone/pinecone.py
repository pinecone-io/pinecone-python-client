"""Backwards-compatibility shim for :mod:`pinecone._client`.

Re-exports the :class:`Pinecone` client that used to live at
:mod:`pinecone.pinecone` before the rewrite. Preserved to keep
pre-rewrite callers working. New code should import from
:mod:`pinecone` directly.

:meta private:
"""

from __future__ import annotations

from pinecone._client import Pinecone

__all__ = ["Pinecone"]
