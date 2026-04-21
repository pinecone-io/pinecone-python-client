"""Tests for to_dict() on search and query-aggregator models."""

from __future__ import annotations

from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.search import (
    Hit,
    SearchRecordsResponse,
    SearchResult,
    SearchUsage,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import ScoredVector


def test_search_usage_to_dict() -> None:
    usage = SearchUsage(read_units=5)
    d = usage.to_dict()
    assert d == {"read_units": 5, "embed_total_tokens": None, "rerank_units": None}


def test_hit_to_dict_required_fields() -> None:
    hit = Hit(id_="v1", score_=0.9)
    d = hit.to_dict()
    assert set(d.keys()) == {"id_", "score_", "fields"}
    assert d["id_"] == "v1"
    assert d["score_"] == 0.9
    assert d["fields"] == {}


def test_hit_to_dict_empty_fields() -> None:
    hit = Hit(id_="v1", score_=0.9, fields={})
    d = hit.to_dict()
    assert d["fields"] == {}


def test_search_result_to_dict_nested_hit() -> None:
    result = SearchResult(hits=[Hit(id_="v1", score_=0.9)])
    d = result.to_dict()
    assert isinstance(d["hits"], list)
    assert isinstance(d["hits"][0], dict)
    assert not isinstance(d["hits"][0], Hit)
    assert d["hits"][0]["id_"] == "v1"
    assert d["hits"][0]["score_"] == 0.9


def test_search_records_response_to_dict() -> None:
    response = SearchRecordsResponse(
        result=SearchResult(hits=[Hit(id_="v1", score_=0.9)]),
        usage=SearchUsage(read_units=3),
    )
    d = response.to_dict()
    assert isinstance(d["result"], dict)
    assert isinstance(d["usage"], dict)
    assert d["result"]["hits"][0]["id_"] == "v1"
    assert d["usage"]["read_units"] == 3


def test_scored_vector_to_dict_required_only() -> None:
    sv = ScoredVector(id="v1", score=0.9)
    d = sv.to_dict()
    assert set(d.keys()) == {"id", "score", "values", "sparse_values", "metadata"}
    assert d["id"] == "v1"
    assert d["score"] == 0.9
    assert d["values"] == []
    assert d["sparse_values"] is None
    assert d["metadata"] is None


def test_scored_vector_to_dict_sparse_values() -> None:
    sv = ScoredVector(
        id="v1",
        score=0.9,
        sparse_values=SparseValues(indices=[1, 2], values=[0.1, 0.2]),
    )
    d = sv.to_dict()
    assert isinstance(d["sparse_values"], dict)
    assert d["sparse_values"] == {"indices": [1, 2], "values": [0.1, 0.2]}


def test_query_namespaces_results_to_dict() -> None:
    result = QueryNamespacesResults()
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "matches" in d
    assert "usage" in d
    assert "ns_usage" in d
    assert isinstance(d["matches"], list)


def test_to_dict_is_pure_read_hit() -> None:
    hit = Hit(id_="v1", score_=0.9, fields={"key": "value"})
    d1 = hit.to_dict()
    d1["id_"] = "mutated"
    d2 = hit.to_dict()
    assert d2["id_"] == "v1"


def test_to_dict_is_pure_read_search_records_response() -> None:
    response = SearchRecordsResponse(
        result=SearchResult(hits=[]),
        usage=SearchUsage(read_units=5),
    )
    d1 = response.to_dict()
    d1["usage"]["read_units"] = 999
    d2 = response.to_dict()
    assert d2["usage"]["read_units"] == 5
