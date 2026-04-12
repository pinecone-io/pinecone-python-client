"""Query results aggregator for multi-namespace queries."""

from __future__ import annotations

import heapq
from typing import Any

from msgspec import Struct, field

from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector


class QueryNamespacesResults(Struct, kw_only=True):
    """Aggregated results from querying multiple namespaces.

    Attributes:
        matches (list[ScoredVector]): Combined top-k results across all namespaces, sorted by
            relevance according to the metric used.
        usage (Usage): Total aggregated read unit usage across all namespaces.
        ns_usage (dict[str, Usage]): Per-namespace read unit usage keyed by namespace name.
    """

    matches: list[ScoredVector] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    ns_usage: dict[str, Usage] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. result['matches'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'matches' in result``)."""
        return key in self.__struct_fields__


_VALID_METRICS = frozenset({"cosine", "euclidean", "dotproduct"})


class QueryResultsAggregator:
    """Merges per-namespace QueryResponse objects into a single combined result.

    Uses a heap-based algorithm to efficiently merge scored vectors from
    multiple namespaces. For cosine/dotproduct metrics, higher scores rank
    first. For euclidean, lower scores rank first. Ties are broken by
    insertion order.

    Args:
        metric: Distance metric — one of ``"cosine"``, ``"euclidean"``,
            or ``"dotproduct"``.
        top_k: Maximum number of results to return. Defaults to 10.

    Raises:
        ValueError: If *metric* is not a recognized value or *top_k* < 1.
    """

    __slots__ = (
        "_counter",
        "_finalized",
        "_heap",
        "_is_bigger_better",
        "_metric",
        "_ns_usage",
        "_read_units",
        "_top_k",
    )

    def __init__(self, *, metric: str, top_k: int = 10) -> None:
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Invalid metric {metric!r}. Must be one of: {', '.join(sorted(_VALID_METRICS))}"
            )
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        self._metric = metric
        self._top_k = top_k
        self._heap: list[tuple[float, int, ScoredVector]] = []
        self._counter: int = 0
        self._finalized: bool = False
        self._read_units: int = 0
        self._ns_usage: dict[str, Usage] = {}
        self._is_bigger_better: bool = metric in ("cosine", "dotproduct")

    def add_results(self, namespace: str, response: QueryResponse) -> None:
        """Add results from a single namespace query.

        Args:
            namespace: Namespace that was queried.
            response: Query response from that namespace.

        Raises:
            ValueError: If called after :meth:`get_results`.
        """
        if self._finalized:
            raise ValueError("Cannot add results after get_results()")

        if response.usage is not None:
            self._read_units += response.usage.read_units or 0
            self._ns_usage[namespace] = response.usage

        for match in response.matches:
            if self._is_bigger_better:
                key = -match.score
            else:
                key = match.score
            heapq.heappush(self._heap, (key, self._counter, match))
            self._counter += 1

        if len(self._heap) > self._top_k:
            self._heap = heapq.nsmallest(self._top_k, self._heap)
            heapq.heapify(self._heap)

    def get_results(self) -> QueryNamespacesResults:
        """Finalize and return the aggregated results.

        After calling this method, no more results can be added.

        Returns:
            Aggregated query results with the top-k matches across all
            namespaces.
        """
        self._finalized = True
        sorted_entries = sorted(self._heap)
        matches = [entry[2] for entry in sorted_entries[: self._top_k]]
        return QueryNamespacesResults(
            matches=matches,
            usage=Usage(read_units=self._read_units),
            ns_usage=self._ns_usage,
        )
