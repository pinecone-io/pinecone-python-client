"""Backwards-compatibility shim for :mod:`pinecone.models.vectors.responses`.

Re-exports classes that used to live at
:mod:`pinecone.db_data.dataclasses.fetch_by_metadata_response`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.vectors.responses import FetchByMetadataResponse, Pagination

__all__ = ["FetchByMetadataResponse", "Pagination"]
