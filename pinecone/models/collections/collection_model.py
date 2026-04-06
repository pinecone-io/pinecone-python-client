"""CollectionModel response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class CollectionModel(Struct, kw_only=True):
    """Response model for a Pinecone collection."""

    name: str
    status: str
    environment: str
    size: int | None = None
    dimension: int | None = None
    vector_count: int | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. collection['name'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None
