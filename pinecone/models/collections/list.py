"""CollectionList wrapper for collection listing responses."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pinecone.models.collections.model import CollectionModel


class CollectionList:
    """Wrapper around a list of CollectionModel with convenience methods."""

    def __init__(self, collections: list[CollectionModel]) -> None:
        self._collections = collections

    def __iter__(self) -> Iterator[CollectionModel]:
        return iter(self._collections)

    def __len__(self) -> int:
        return len(self._collections)

    def __getitem__(self, index: int) -> CollectionModel:
        return self._collections[index]

    def to_dict(self) -> dict[str, Any]:
        return {"data": [c.to_dict() for c in self._collections]}

    def names(self) -> list[str]:
        """Return a list of collection names."""
        return [c.name for c in self._collections]

    def __repr__(self) -> str:
        summaries = ", ".join(f"<name={c.name!r}, status={c.status!r}>" for c in self._collections)
        return f"CollectionList([{summaries}])"
