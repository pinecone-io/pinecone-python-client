"""Backwards-compatibility shim for :mod:`pinecone.models.vectors`.

Re-exports classes that used to live at :mod:`pinecone.db_data.dataclasses` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.db_data.dataclasses.fetch_by_metadata_response import (
    FetchByMetadataResponse,
    Pagination,
)
from pinecone.db_data.dataclasses.fetch_response import FetchResponse
from pinecone.db_data.dataclasses.query_response import QueryResponse
from pinecone.db_data.dataclasses.search_query import SearchQuery
from pinecone.db_data.dataclasses.search_query_vector import SearchQueryVector
from pinecone.db_data.dataclasses.search_rerank import SearchRerank
from pinecone.db_data.dataclasses.update_response import UpdateResponse
from pinecone.db_data.dataclasses.upsert_response import UpsertResponse
from pinecone.db_data.dataclasses.utils import DictLike

__all__ = [
    "DictLike",
    "FetchByMetadataResponse",
    "FetchResponse",
    "Pagination",
    "QueryResponse",
    "SearchQuery",
    "SearchQueryVector",
    "SearchRerank",
    "UpdateResponse",
    "UpsertResponse",
]
