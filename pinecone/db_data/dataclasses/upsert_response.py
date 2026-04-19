"""Backwards-compatibility shim for :mod:`pinecone.db_data.dataclasses.upsert_response`.

Re-exports classes that used to live at :mod:`pinecone.db_data.dataclasses.upsert_response`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

# XXX: The canonical UpsertResponse is a msgspec.Struct; this shim preserves the
# legacy dataclass+DictLike interface for pre-rewrite callers.

from __future__ import annotations

import dataclasses
from typing import Any

from pinecone.db_data.dataclasses.utils import DictLike


@dataclasses.dataclass
class UpsertResponse(DictLike):
    upserted_count: int
    _response_info: Any = dataclasses.field(default_factory=lambda: {"raw_headers": {}})


__all__ = ["UpsertResponse"]
