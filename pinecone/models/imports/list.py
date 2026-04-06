"""ImportList wrapper for listing responses."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from pinecone.models.imports.model import ImportModel

if TYPE_CHECKING:
    from pinecone.models.vectors.responses import Pagination


class ImportList:
    """Wrapper around a list of ImportModel with convenience methods."""

    def __init__(
        self,
        imports: list[ImportModel],
        *,
        pagination: Pagination | None = None,
    ) -> None:
        """Initialize an ImportList.

        Args:
            imports: List of :class:`ImportModel` instances representing
                bulk import operations.
            pagination: Optional :class:`Pagination` token for fetching
                additional pages of results.
        """
        self._imports = imports
        self.pagination = pagination

    def __iter__(self) -> Iterator[ImportModel]:
        return iter(self._imports)

    def __len__(self) -> int:
        return len(self._imports)

    def __getitem__(self, index: int) -> ImportModel:
        return self._imports[index]

    def __repr__(self) -> str:
        return f"ImportList(imports={self._imports!r})"
