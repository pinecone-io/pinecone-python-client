"""CollectionModel response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class CollectionModel(Struct, kw_only=True):
    """Response model for a Pinecone collection.

    Attributes:
        name: The name of the collection.
        status: Current status of the collection (e.g. ``"Ready"``,
            ``"Initializing"``, ``"Terminating"``).
        environment: Deployment environment where the collection is hosted.
        size: Size of the collection in bytes, or ``None`` if not yet
            available.
        dimension: Dimensionality of vectors in the collection, or ``None``
            if not yet available.
        vector_count: Number of vectors in the collection, or ``None`` if
            not yet available.
    """

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
