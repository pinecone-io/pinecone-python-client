"""Backwards-compatibility shim for :mod:`pinecone.models.vectors.query_aggregator`.

Re-exports classes that used to live at :mod:`pinecone.db_data.query_results_aggregator`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.vectors.query_aggregator import (
    QueryResultsAggregator,
    QueryResultsAggregatorInvalidTopKError,
)

__all__ = [
    "QueryResultsAggregator",
    "QueryResultsAggregatorInvalidTopKError",
]
