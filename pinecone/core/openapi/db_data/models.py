"""Backwards-compatibility shim for :mod:`pinecone.core.openapi.db_data.models`.

Re-exports classes that used to live at :mod:`pinecone.core.openapi.db_data.models`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.vectors.responses import DescribeIndexStatsResponse

__all__ = ["DescribeIndexStatsResponse"]
