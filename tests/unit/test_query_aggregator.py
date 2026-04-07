"""Tests for QueryResultsAggregator and QueryNamespacesResults."""

from __future__ import annotations

import pytest

from pinecone.models.vectors.query_aggregator import (
    QueryNamespacesResults,
    QueryResultsAggregator,
)
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector


def _make_response(
    matches: list[ScoredVector],
    namespace: str = "",
    usage: Usage | None = None,
) -> QueryResponse:
    return QueryResponse(matches=matches, namespace=namespace, usage=usage)


def _scored(id: str, score: float) -> ScoredVector:
    return ScoredVector(id=id, score=score)


class TestCosineMetric:
    def test_cosine_bigger_is_better(self) -> None:
        agg = QueryResultsAggregator(metric="cosine", top_k=3)
        agg.add_results(
            "ns1",
            _make_response(
                [_scored("a", 0.9), _scored("b", 0.7)],
                usage=Usage(read_units=5),
            ),
        )
        agg.add_results(
            "ns2",
            _make_response(
                [_scored("c", 0.95), _scored("d", 0.6)],
                usage=Usage(read_units=3),
            ),
        )
        result = agg.get_results()
        assert [m.id for m in result.matches] == ["c", "a", "b"]
        assert [m.score for m in result.matches] == [0.95, 0.9, 0.7]


class TestEuclideanMetric:
    def test_euclidean_smaller_is_better(self) -> None:
        agg = QueryResultsAggregator(metric="euclidean", top_k=3)
        agg.add_results(
            "ns1",
            _make_response(
                [_scored("a", 0.1), _scored("b", 0.5)],
                usage=Usage(read_units=2),
            ),
        )
        agg.add_results(
            "ns2",
            _make_response(
                [_scored("c", 0.05), _scored("d", 0.8)],
                usage=Usage(read_units=1),
            ),
        )
        result = agg.get_results()
        assert [m.id for m in result.matches] == ["c", "a", "b"]
        assert [m.score for m in result.matches] == [0.05, 0.1, 0.5]


class TestDotproductMetric:
    def test_dotproduct_bigger_is_better(self) -> None:
        agg = QueryResultsAggregator(metric="dotproduct", top_k=2)
        agg.add_results(
            "ns1",
            _make_response(
                [_scored("a", 10.0), _scored("b", 5.0)],
                usage=Usage(read_units=1),
            ),
        )
        agg.add_results(
            "ns2",
            _make_response(
                [_scored("c", 8.0)],
                usage=Usage(read_units=1),
            ),
        )
        result = agg.get_results()
        assert [m.id for m in result.matches] == ["a", "c"]


class TestTopK:
    def test_top_k_limits_results(self) -> None:
        agg = QueryResultsAggregator(metric="cosine", top_k=5)
        matches = [_scored(f"v{i}", float(i)) for i in range(20)]
        agg.add_results("ns1", _make_response(matches, usage=Usage(read_units=1)))
        result = agg.get_results()
        assert len(result.matches) == 5
        # Cosine: bigger is better, so top 5 are 19..15
        assert [m.id for m in result.matches] == ["v19", "v18", "v17", "v16", "v15"]

    def test_default_top_k_is_10(self) -> None:
        agg = QueryResultsAggregator(metric="cosine")
        matches = [_scored(f"v{i}", float(i)) for i in range(15)]
        agg.add_results("ns1", _make_response(matches, usage=Usage(read_units=1)))
        result = agg.get_results()
        assert len(result.matches) == 10


class TestTieBreaking:
    def test_insertion_order_tiebreaking(self) -> None:
        agg = QueryResultsAggregator(metric="cosine", top_k=4)
        agg.add_results(
            "ns1",
            _make_response(
                [_scored("first", 0.5), _scored("second", 0.5)],
                usage=Usage(read_units=1),
            ),
        )
        agg.add_results(
            "ns2",
            _make_response(
                [_scored("third", 0.5), _scored("fourth", 0.5)],
                usage=Usage(read_units=1),
            ),
        )
        result = agg.get_results()
        assert [m.id for m in result.matches] == ["first", "second", "third", "fourth"]


class TestDedupNotByAggregator:
    def test_dedup_not_done_by_aggregator(self) -> None:
        """Adding the same namespace twice works — dedup is the caller's job."""
        agg = QueryResultsAggregator(metric="cosine", top_k=10)
        resp = _make_response([_scored("a", 0.9)], usage=Usage(read_units=1))
        agg.add_results("ns1", resp)
        agg.add_results("ns1", resp)
        result = agg.get_results()
        assert len(result.matches) == 2
        assert result.usage.read_units == 2


class TestValidation:
    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid metric"):
            QueryResultsAggregator(metric="invalid")

    def test_invalid_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            QueryResultsAggregator(metric="cosine", top_k=0)


class TestSingleReadSemantics:
    def test_single_read_semantics(self) -> None:
        agg = QueryResultsAggregator(metric="cosine", top_k=5)
        agg.add_results(
            "ns1",
            _make_response([_scored("a", 0.5)], usage=Usage(read_units=1)),
        )
        agg.get_results()
        with pytest.raises(ValueError, match="Cannot add results after"):
            agg.add_results(
                "ns2",
                _make_response([_scored("b", 0.6)], usage=Usage(read_units=1)),
            )


class TestUsageAggregation:
    def test_usage_aggregation(self) -> None:
        agg = QueryResultsAggregator(metric="cosine", top_k=10)
        agg.add_results(
            "ns1",
            _make_response([_scored("a", 0.9)], usage=Usage(read_units=5)),
        )
        agg.add_results(
            "ns2",
            _make_response([_scored("b", 0.8)], usage=Usage(read_units=3)),
        )
        result = agg.get_results()
        assert result.usage.read_units == 8
        assert result.ns_usage["ns1"].read_units == 5
        assert result.ns_usage["ns2"].read_units == 3


class TestEmptyResults:
    def test_empty_results(self) -> None:
        agg = QueryResultsAggregator(metric="cosine", top_k=10)
        result = agg.get_results()
        assert result.matches == []
        assert result.usage.read_units == 0
        assert result.ns_usage == {}


class TestQueryNamespacesResultsBracketAccess:
    def test_bracket_access(self) -> None:
        result = QueryNamespacesResults(
            matches=[_scored("a", 0.5)],
            usage=Usage(read_units=1),
            ns_usage={"ns1": Usage(read_units=1)},
        )
        assert result["matches"] == result.matches
        assert result["usage"] == result.usage
        assert result["ns_usage"] == result.ns_usage

    def test_bracket_access_missing_key(self) -> None:
        result = QueryNamespacesResults()
        with pytest.raises(KeyError):
            result["nonexistent"]
