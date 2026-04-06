"""Adapter for Indexes API responses."""

from __future__ import annotations

from typing import Any

import msgspec

from pinecone.models.indexes.index import IndexModel
from pinecone.models.indexes.list import IndexList


class IndexesAdapter:
    """Transforms raw API JSON into IndexModel / IndexList instances."""

    @staticmethod
    def to_index_model(data: dict[str, Any]) -> IndexModel:
        """Convert a single index JSON dict to an IndexModel."""
        return msgspec.convert(data, IndexModel)

    @staticmethod
    def to_index_list(data: dict[str, Any]) -> IndexList:
        """Convert a list-indexes JSON response to an IndexList."""
        indexes = [msgspec.convert(idx, IndexModel) for idx in data.get("indexes", [])]
        return IndexList(indexes)
