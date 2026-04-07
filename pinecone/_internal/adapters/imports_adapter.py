"""Adapter for Bulk Import API responses."""

from __future__ import annotations

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone.models.imports.list import ImportList
from pinecone.models.imports.model import ImportModel, StartImportResponse
from pinecone.models.vectors.responses import Pagination


class _ImportListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-imports response."""

    data: list[ImportModel] = []
    pagination: Pagination | None = None


class ImportsAdapter:
    """Transforms raw API JSON into ImportModel / ImportList instances."""

    @staticmethod
    def to_start_import_response(data: bytes) -> StartImportResponse:
        """Decode raw JSON bytes into a StartImportResponse."""
        return decode_response(data, StartImportResponse)

    @staticmethod
    def to_import_model(data: bytes) -> ImportModel:
        """Decode raw JSON bytes into an ImportModel."""
        return decode_response(data, ImportModel)

    @staticmethod
    def to_import_list(data: bytes) -> ImportList:
        """Decode raw JSON bytes from a list-imports response into an ImportList."""
        envelope = decode_response(data, _ImportListEnvelope)
        return ImportList(envelope.data, pagination=envelope.pagination)
