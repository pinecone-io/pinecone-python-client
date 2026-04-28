"""ImportList wrapper for listing responses."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

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

    def to_dict(self) -> dict[str, Any]:
        """Return the list as a serializable dict.

        Returns:
            dict[str, Any]: A dict with a ``"data"`` key containing a list of
            import dicts, each produced by :meth:`ImportModel.to_dict`. When the
            wrapper has a pagination token, the dict also includes a
            ``"pagination"`` key with the token for fetching the next page.

        Examples:
            Serialize a page of bulk imports:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> index = pc.Index("product-search")
            >>> imports = index.list_imports_paginated()
            >>> imports.to_dict()  # doctest: +SKIP
            {'data': [{'id': 'import-abc123', ...}, {'id': 'import-def456', ...}]}
        """
        result: dict[str, Any] = {"data": [i.to_dict() for i in self._imports]}
        if self.pagination is not None:
            result["pagination"] = self.pagination.to_dict()
        return result

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<id={i.id!r}, status={i.status!r}, percent={i.percent_complete}>"
            for i in self._imports
        )
        return f"ImportList([{summaries}])"
