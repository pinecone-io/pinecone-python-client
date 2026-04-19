"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.fetch_by_metadata_response`.

Re-exports classes that used to live at
:mod:`pinecone.db_data.dataclasses.fetch_by_metadata_response`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: The canonical FetchByMetadataResponse is a msgspec.Struct; this shim preserves
# the legacy dataclass+DictLike interface for pre-rewrite callers.

from __future__ import annotations

import dataclasses
from typing import Any

from pinecone.db_data.dataclasses.utils import DictLike


@dataclasses.dataclass
class Pagination:
    next: str


@dataclasses.dataclass
class FetchByMetadataResponse(DictLike):
    namespace: str
    vectors: dict[str, Any]
    usage: Any | None = None
    pagination: Pagination | None = None
    _response_info: Any = dataclasses.field(default_factory=lambda: {"raw_headers": {}})


__all__ = ["FetchByMetadataResponse", "Pagination"]
