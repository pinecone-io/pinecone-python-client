"""Backwards-compatibility shim for :mod:`pinecone.db_control.models.list_response`.

Re-exports classes that used to live at :mod:`pinecone.db_control.models.list_response`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: The canonical ListResponse is a msgspec.Struct; this shim preserves the
# legacy NamedTuple interface for pre-rewrite callers.

from __future__ import annotations

from typing import NamedTuple


class Pagination(NamedTuple):
    next: str


class ListResponse(NamedTuple):
    namespace: str
    vectors: list[object]
    pagination: Pagination | None


__all__ = ["ListResponse", "Pagination"]
