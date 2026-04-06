"""Adapter for data plane vector operation responses.

Transforms raw API JSON bytes into typed SDK response models. Each method
handles camelCase -> snake_case mapping via msgspec's ``rename="camel"``
on the target Struct, plus any pre-processing documented in its docstring.
"""

from __future__ import annotations

from typing import Any

import msgspec

from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    UpdateResponse,
    UpsertResponse,
)


class VectorsAdapter:
    """Transforms raw API JSON into typed data-plane response models."""

    @staticmethod
    def to_upsert_response(data: bytes) -> UpsertResponse:
        """Decode raw JSON bytes into an UpsertResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return msgspec.json.decode(data, type=UpsertResponse)

    @staticmethod
    def to_query_response(data: bytes) -> QueryResponse:
        """Decode raw JSON bytes into a QueryResponse.

        Transformations:
            - Strips deprecated ``results`` field before decoding (claim unified-rs-0012).
            - Normalizes null namespace to empty string (claim unified-rs-0013).
        """
        raw: dict[str, Any] = msgspec.json.decode(data)
        raw.pop("results", None)
        if raw.get("namespace") is None:
            raw["namespace"] = ""
        return msgspec.convert(raw, QueryResponse)

    @staticmethod
    def to_fetch_response(data: bytes) -> FetchResponse:
        """Decode raw JSON bytes into a FetchResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return msgspec.json.decode(data, type=FetchResponse)

    @staticmethod
    def to_stats_response(data: bytes) -> DescribeIndexStatsResponse:
        """Decode raw JSON bytes into a DescribeIndexStatsResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return msgspec.json.decode(data, type=DescribeIndexStatsResponse)

    @staticmethod
    def to_list_response(data: bytes) -> ListResponse:
        """Decode raw JSON bytes into a ListResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return msgspec.json.decode(data, type=ListResponse)

    @staticmethod
    def to_update_response(data: bytes) -> UpdateResponse:
        """Decode raw JSON bytes into an UpdateResponse.

        Transformations:
            - Direct decode; camelCase handled by Struct rename.
        """
        return msgspec.json.decode(data, type=UpdateResponse)

    @staticmethod
    def to_delete_response(data: bytes) -> None:
        """Handle delete response (empty body).

        The delete endpoint returns an empty JSON object ``{}``.
        No model is needed — this method exists for adapter completeness.
        """
        return None
