"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.search_query_vector`.

Re-exports classes that used to live at
:mod:`pinecone.db_data.dataclasses.search_query_vector`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.vectors.search import SearchQueryVector

__all__ = ["SearchQueryVector"]
