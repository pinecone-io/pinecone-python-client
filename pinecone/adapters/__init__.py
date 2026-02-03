"""Adapter layer for converting between OpenAPI models and SDK types.

This module provides a centralized adapter layer that isolates SDK code from
the details of generated OpenAPI models. This enables:

- Single source of truth for response transformations
- Easier testing (adapters can be tested in isolation)
- Version flexibility (adapters can handle different API versions)
- Clear contracts between generated and SDK code

Usage:
    >>> from pinecone.adapters import adapt_query_response, adapt_upsert_response
    >>> sdk_response = adapt_query_response(openapi_response)
"""

from pinecone.adapters.protocols import (
    FetchResponseAdapter,
    IndexModelAdapter,
    IndexStatusAdapter,
    QueryResponseAdapter,
    UpsertResponseAdapter,
)
from pinecone.adapters.response_adapters import (
    adapt_fetch_response,
    adapt_query_response,
    adapt_upsert_response,
    UpsertResponseTransformer,
)

__all__ = [
    "adapt_fetch_response",
    "adapt_query_response",
    "adapt_upsert_response",
    "UpsertResponseTransformer",
    "FetchResponseAdapter",
    "IndexModelAdapter",
    "IndexStatusAdapter",
    "QueryResponseAdapter",
    "UpsertResponseAdapter",
]
