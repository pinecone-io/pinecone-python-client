"""Backwards-compatibility shim for :mod:`pinecone.db_data`.

Re-exports the data-plane clients and models that used to live here,
matching what :mod:`pinecone.data` exports plus the ``SearchQueryVector``
and underscore-prefixed ``_Index``/``_IndexAsyncio`` aliases this path also
carried. Preserved to keep pre-rewrite callers working. New code should
import these from the top-level :mod:`pinecone` namespace, except
``SearchQueryVector``, whose canonical home is
:mod:`pinecone.models.vectors.search`. The per-field vector validation
errors (``VectorDictionaryMissingKeysError`` and the rest) also used to live
here and have no replacement; vector validation now raises
:class:`pinecone.PineconeValueError` or :class:`pinecone.PineconeTypeError`.
``Inference`` is not re-exported here; import it from :mod:`pinecone.inference`.

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
from pinecone.models.vectors.search import SearchQuery, SearchQueryVector, SearchRerank
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import Vector

IndexAsyncio = AsyncIndex
_Index = Index
_IndexAsyncio = AsyncIndex

__all__ = [
    "DescribeIndexStatsResponse",
    "FetchResponse",
    "ImportErrorMode",
    "Index",
    "IndexAsyncio",
    "QueryResponse",
    "SearchQuery",
    "SearchQueryVector",
    "SearchRerank",
    "SparseValues",
    "UpsertResponse",
    "Vector",
    "_Index",
    "_IndexAsyncio",
]
