"""Tests for response_info field on data-plane response structs."""

from __future__ import annotations

from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListResponse,
    QueryResponse,
    ResponseInfo,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchRecordsResponse, SearchResult, SearchUsage


class TestUpsertResponseInfo:
    def test_upsert_response_has_response_info_field(self) -> None:
        result = UpsertResponse(upserted_count=5)
        assert result.response_info is None

    def test_upsert_response_accepts_response_info(self) -> None:
        info = ResponseInfo(raw_headers={"x-pinecone-request-id": "req-1"})
        result = UpsertResponse(upserted_count=5, response_info=info)
        assert result.response_info is not None
        assert result.response_info.request_id == "req-1"


class TestQueryResponseInfo:
    def test_query_response_has_response_info_field(self) -> None:
        result = QueryResponse()
        assert result.response_info is None

    def test_query_response_accepts_response_info(self) -> None:
        info = ResponseInfo(raw_headers={"x-pinecone-request-id": "req-2", "x-pinecone-lsn-reconciled": "42"})
        result = QueryResponse(response_info=info)
        assert result.response_info is not None
        assert result.response_info.request_id == "req-2"
        assert result.response_info.lsn_reconciled == 42


class TestSearchResponseInfo:
    def test_search_response_has_response_info_field(self) -> None:
        result = SearchRecordsResponse(
            result=SearchResult(hits=[]),
            usage=SearchUsage(read_units=1),
        )
        assert result.response_info is None

    def test_search_response_accepts_response_info(self) -> None:
        info = ResponseInfo(raw_headers={"x-pinecone-request-id": "req-3"})
        result = SearchRecordsResponse(
            result=SearchResult(hits=[]),
            usage=SearchUsage(read_units=1),
            response_info=info,
        )
        assert result.response_info is not None
        assert result.response_info.request_id == "req-3"


class TestOtherResponsesHaveResponseInfo:
    def test_fetch_response(self) -> None:
        result = FetchResponse()
        assert result.response_info is None

    def test_fetch_by_metadata_response(self) -> None:
        result = FetchByMetadataResponse()
        assert result.response_info is None

    def test_describe_index_stats_response(self) -> None:
        result = DescribeIndexStatsResponse()
        assert result.response_info is None

    def test_list_response(self) -> None:
        result = ListResponse()
        assert result.response_info is None

    def test_update_response(self) -> None:
        result = UpdateResponse()
        assert result.response_info is None

    def test_upsert_records_response(self) -> None:
        result = UpsertRecordsResponse(record_count=10)
        assert result.response_info is None


class TestResponseInfoMutable:
    def test_response_info_can_be_set_after_construction(self) -> None:
        result = UpsertResponse(upserted_count=5)
        assert result.response_info is None
        result.response_info = ResponseInfo(raw_headers={"x-pinecone-request-id": "req-post"})
        assert result.response_info is not None
        assert result.response_info.request_id == "req-post"


class TestResponseInfoImportable:
    def test_response_info_importable_from_pinecone(self) -> None:
        from pinecone import ResponseInfo

        assert ResponseInfo is not None
