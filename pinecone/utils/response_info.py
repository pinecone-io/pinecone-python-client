"""Backwards-compatibility shim for :mod:`pinecone.utils.response_info`.

Re-exports classes that used to live at :mod:`pinecone.utils.response_info` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from typing import TypedDict


# XXX: No canonical TypedDict ResponseInfo exists in the new SDK; the new SDK uses
# pinecone.models.vectors.responses.ResponseInfo (a msgspec Struct) for LSN/request
# tracking. This minimal TypedDict definition preserves the legacy interface for
# callers that depend on pinecone.utils.response_info.ResponseInfo.
class ResponseInfo(TypedDict, total=False):
    """HTTP response metadata carrier (legacy TypedDict interface)."""

    raw_headers: dict[str, str]


__all__ = ["ResponseInfo"]
