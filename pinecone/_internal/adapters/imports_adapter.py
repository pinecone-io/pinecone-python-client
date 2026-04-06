"""Adapter for Bulk Import API responses."""

from __future__ import annotations

import msgspec
from msgspec import Struct

from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse


class _Pagination(Struct, kw_only=True):
    """Pagination token from the import list response."""

    next: str | None = None


class _ImportListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-imports response."""

    data: list[ImportModel] = []
    pagination: _Pagination | None = None


class ImportsAdapter:
    """Transforms raw API JSON into ImportModel / ImportList instances."""

    @staticmethod
    def to_start_import_response(data: bytes) -> StartImportResponse:
        """Decode raw JSON bytes into a StartImportResponse."""
        return msgspec.json.decode(data, type=StartImportResponse)

    @staticmethod
    def to_import_model(data: bytes) -> ImportModel:
        """Decode raw JSON bytes into an ImportModel."""
        return msgspec.json.decode(data, type=ImportModel)

    @staticmethod
    def to_import_list(data: bytes) -> ImportList:
        """Decode raw JSON bytes from a list-imports response into an ImportList."""
        envelope = msgspec.json.decode(data, type=_ImportListEnvelope)
        return ImportList(envelope.data)
