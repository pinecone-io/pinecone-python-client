"""Unit tests for to_dict() on data-plane response models."""

from __future__ import annotations

from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    ListItem,
    ListResponse,
    NamespaceSummary,
    Pagination,
    UpsertRecordsResponse,
)


def test_describe_index_stats_to_dict_empty() -> None:
    result = DescribeIndexStatsResponse().to_dict()
    assert isinstance(result, dict)
    assert "namespaces" in result
    assert "index_fullness" in result
    assert "total_vector_count" in result


def test_describe_index_stats_to_dict_nested_namespace_summary() -> None:
    resp = DescribeIndexStatsResponse(namespaces={"ns1": NamespaceSummary(vector_count=10)})
    result = resp.to_dict()
    ns1 = result["namespaces"]["ns1"]
    assert isinstance(ns1, dict)
    assert ns1["vector_count"] == 10
    assert not isinstance(ns1, NamespaceSummary)


def test_list_response_to_dict_required_only() -> None:
    result = ListResponse().to_dict()
    assert result["vectors"] == []


def test_list_response_to_dict_nested_list_item() -> None:
    resp = ListResponse(vectors=[ListItem(id="v1")])
    result = resp.to_dict()
    assert isinstance(result["vectors"][0], dict)
    assert result["vectors"][0]["id"] == "v1"


def test_list_response_to_dict_with_pagination() -> None:
    resp = ListResponse(pagination=Pagination(next="tok"))
    result = resp.to_dict()
    assert isinstance(result["pagination"], dict)
    assert result["pagination"]["next"] == "tok"


def test_upsert_records_response_to_dict() -> None:
    result = UpsertRecordsResponse(record_count=5).to_dict()
    assert result == {"record_count": 5, "response_info": None}


def test_to_dict_all_optional_none() -> None:
    dis = DescribeIndexStatsResponse().to_dict()
    assert dis["dimension"] is None
    assert dis["metric"] is None
    assert dis["vector_type"] is None
    assert dis["memory_fullness"] is None
    assert dis["storage_fullness"] is None
    assert dis["response_info"] is None

    lr = ListResponse().to_dict()
    assert lr["pagination"] is None
    assert lr["usage"] is None
    assert lr["response_info"] is None

    urr = UpsertRecordsResponse(record_count=0).to_dict()
    assert urr["response_info"] is None


def test_to_dict_is_pure_read() -> None:
    resp = DescribeIndexStatsResponse(total_vector_count=42)
    first = resp.to_dict()
    first["total_vector_count"] = 999
    second = resp.to_dict()
    assert second["total_vector_count"] == 42
