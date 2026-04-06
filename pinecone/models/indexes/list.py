"""IndexList wrapper for index listing responses."""

from __future__ import annotations

from collections.abc import Iterator

from pinecone.models.indexes.index import IndexModel


class IndexList:
    """Wrapper around a list of IndexModel with convenience methods."""

    def __init__(self, indexes: list[IndexModel]) -> None:
        self._indexes = indexes

    def __iter__(self) -> Iterator[IndexModel]:
        return iter(self._indexes)

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, index: int) -> IndexModel:
        return self._indexes[index]

    def names(self) -> list[str]:
        """Return a list of index names."""
        return [idx.name for idx in self._indexes]

    def __repr__(self) -> str:
        return f"IndexList(indexes={self._indexes!r})"
