"""Adapter for data plane vector operation responses.

Transforms raw API JSON bytes into typed SDK response models. Each method
handles camelCase -> snake_case mapping via msgspec's ``rename="camel"``
on the target Struct, plus any pre-processing documented in its docstring.
"""

from __future__ import annotations

import httpx

from pinecone._internal.adapters._decode import decode_response
from pinecone.models.namespaces.models import (
    ListNamespacesResponse,
    NamespaceDescription,
)
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    ResponseInfo,
    UpdateResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchRecordsResponse


def _parse_lsn(headers: httpx.Headers, name: str) -> int | None:
    """Extract an integer LSN value from a response header.

    Returns ``None`` when the header is absent or the value is not a valid
    integer.  Header lookup is case-insensitive (httpx normalises names).
    """
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def extract_response_info(response: httpx.Response) -> ResponseInfo:
    """Build a :class:`ResponseInfo` from *response* headers.

    Extracts ``x-pinecone-request-id``, ``x-pinecone-lsn-reconciled``, and
    ``x-pinecone-lsn-committed`` headers.
    """
    headers = response.headers
    return ResponseInfo(
        request_id=headers.get("x-pinecone-request-id"),
        lsn_reconciled=_parse_lsn(headers, "x-pinecone-lsn-reconciled"),
        lsn_committed=_parse_lsn(headers, "x-pinecone-lsn-committed"),
    )


class VectorsAdapter:
    """Transforms raw API JSON into typed data-plane response models."""

    @staticmethod
    def to_upsert_response(data: bytes) -> UpsertResponse:
        """Decode raw JSON bytes into an UpsertResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return decode_response(data,UpsertResponse)

    @staticmethod
    def to_query_response(data: bytes) -> QueryResponse:
        """Decode raw JSON bytes into a QueryResponse.

        Transformations:
            - Deprecated ``results`` field silently ignored (forbid_unknown_fields
              is False by default; claim unified-rs-0012).
            - Null namespace normalized to empty string via ``__post_init__``
              on QueryResponse (claim unified-rs-0013).
        """
        return decode_response(data,QueryResponse)

    @staticmethod
    def to_fetch_response(data: bytes) -> FetchResponse:
        """Decode raw JSON bytes into a FetchResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return decode_response(data,FetchResponse)

    @staticmethod
    def to_fetch_by_metadata_response(data: bytes) -> FetchByMetadataResponse:
        """Decode raw JSON bytes into a FetchByMetadataResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return decode_response(data,FetchByMetadataResponse)

    @staticmethod
    def to_stats_response(data: bytes) -> DescribeIndexStatsResponse:
        """Decode raw JSON bytes into a DescribeIndexStatsResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return decode_response(data,DescribeIndexStatsResponse)

    @staticmethod
    def to_list_response(data: bytes) -> ListResponse:
        """Decode raw JSON bytes into a ListResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return decode_response(data,ListResponse)

    @staticmethod
    def to_update_response(data: bytes) -> UpdateResponse:
        """Decode raw JSON bytes into an UpdateResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return decode_response(data,UpdateResponse)

    @staticmethod
    def to_search_response(data: bytes) -> SearchRecordsResponse:
        """Decode raw JSON bytes into a SearchRecordsResponse.

        Transformations:
            - Direct decode; search API uses snake_case natively.
        """
        return decode_response(data,SearchRecordsResponse)

    @staticmethod
    def to_namespace_description(data: bytes) -> NamespaceDescription:
        """Decode raw JSON bytes into a NamespaceDescription.

        Transformations:
            - Direct decode; namespace API uses snake_case natively.
        """
        return decode_response(data,NamespaceDescription)

    @staticmethod
    def to_list_namespaces_response(data: bytes) -> ListNamespacesResponse:
        """Decode raw JSON bytes into a ListNamespacesResponse.

        Transformations:
            - Direct decode; namespace API uses snake_case natively.
        """
        return decode_response(data,ListNamespacesResponse)

    @staticmethod
    def to_delete_response(data: bytes) -> None:
        """Handle delete response (empty body).

        The delete endpoint returns an empty JSON object ``{}``.
        No model is needed — this method exists for adapter completeness.
        """
        return None
