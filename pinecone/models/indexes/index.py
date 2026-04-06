"""Index and IndexStatus response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class IndexStatus(Struct, kw_only=True):
    """Status of an index."""

    ready: bool
    state: str


class IndexModel(Struct, kw_only=True):
    """Response model for a Pinecone index."""

    name: str
    metric: str
    host: str
    status: IndexStatus
    spec: dict[str, Any]
    vector_type: str = "dense"
    dimension: int | None = None
    deletion_protection: str = "disabled"
    tags: dict[str, str] | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. index['name'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None
