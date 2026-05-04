"""Backwards-compatibility shim for :mod:`pinecone`.

Re-exports public data-plane symbols that used to live under
:mod:`pinecone.data` (which itself was a shim for
:mod:`pinecone.db_data`) before the rewrite. Preserved to keep
pre-rewrite callers working. New code should import from the
canonical top-level :mod:`pinecone` namespace.

:meta private:
"""

from __future__ import annotations

from pinecone.async_client.async_index import AsyncIndex
from pinecone.index import Index
from pinecone.models.imports.error_mode import ImportErrorMode
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    QueryResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchQuery, SearchRerank
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

IndexAsyncio = AsyncIndex

__all__ = [
    "DescribeIndexStatsResponse",
    "FetchResponse",
    "ImportErrorMode",
    "Index",
    "IndexAsyncio",
    "QueryResponse",
    "SearchQuery",
    "SearchRerank",
    "SparseValues",
    "UpsertResponse",
    "Vector",
]
