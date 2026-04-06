"""Adapter for Collections API responses."""

from __future__ import annotations

import msgspec

from pinecone.models.collections.list import CollectionList
from pinecone.models.collections.model import CollectionModel


class _CollectionListEnvelope(msgspec.Struct, kw_only=True):
    """Internal envelope for the list-collections response."""

    collections: list[CollectionModel] = []


class CollectionsAdapter:
    """Transforms raw API JSON into CollectionModel / CollectionList instances."""

    @staticmethod
    def to_collection(data: bytes) -> CollectionModel:
        """Decode raw JSON bytes into a CollectionModel."""
        return msgspec.json.decode(data, type=CollectionModel)

    @staticmethod
    def to_collection_list(data: bytes) -> CollectionList:
        """Decode raw JSON bytes from a list-collections response into a CollectionList."""
        envelope = msgspec.json.decode(data, type=_CollectionListEnvelope)
        return CollectionList(envelope.collections)
