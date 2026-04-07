"""Adapter for Indexes API responses."""

from __future__ import annotations

import msgspec

from pinecone._internal.adapters._decode import decode_response
from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList


class _IndexListEnvelope(msgspec.Struct, kw_only=True):
    """Internal envelope for the list-indexes response."""

    indexes: list[IndexModel] = []


class IndexesAdapter:
    """Transforms raw API JSON into IndexModel / IndexList instances."""

    @staticmethod
    def to_index_model(data: bytes) -> IndexModel:
        """Decode raw JSON bytes into an IndexModel."""
        return decode_response(data, IndexModel)

    @staticmethod
    def to_index_list(data: bytes) -> IndexList:
        """Decode raw JSON bytes from a list-indexes response into an IndexList."""
        envelope = decode_response(data, _IndexListEnvelope)
        return IndexList(envelope.indexes)
