"""Tests that sync Index data-plane methods populate response_info."""

from __future__ import annotations

import httpx
import respx

from pinecone import Index

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"

RESPONSE_HEADERS = {
    "x-pinecone-request-id": "req-abc",
    "x-pinecone-lsn-committed": "42",
    "x-pinecone-lsn-reconciled": "40",
}


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


class TestUpsertPopulatesResponseInfo:
    """upsert() populates response_info from response headers."""

    @respx.mock
    def test_upsert_populates_response_info(self) -> None:
        respx.post(f"{INDEX_HOST_HTTPS}/vectors/upsert").mock(
            return_value=httpx.Response(
                200,
                json={"upsertedCount": 1},
                headers=RESPONSE_HEADERS,
            )
        )
        idx = _make_index()
        result = idx.upsert(vectors=[("id1", [0.1, 0.2])])
        assert result.response_info is not None
        assert result.response_info.request_id == "req-abc"
        assert result.response_info.lsn_committed == 42
        assert result.response_info.lsn_reconciled == 40


class TestQueryPopulatesResponseInfo:
    """query() populates response_info from response headers."""

    @respx.mock
    def test_query_populates_response_info(self) -> None:
        respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(
                200,
                json={"matches": [], "namespace": "", "usage": {"readUnits": 1}},
                headers={"x-pinecone-request-id": "req-def"},
            )
        )
        idx = _make_index()
        result = idx.query(top_k=10, vector=[0.1, 0.2])
        assert result.response_info is not None
        assert result.response_info.request_id == "req-def"


class TestUpsertRecordsPopulatesResponseInfo:
    """upsert_records() populates response_info from response headers."""

    @respx.mock
    def test_upsert_records_populates_response_info(self) -> None:
        respx.post(f"{INDEX_HOST_HTTPS}/records/namespaces/ns/upsert").mock(
            return_value=httpx.Response(
                200,
                content=b"",
                headers=RESPONSE_HEADERS,
            )
        )
        idx = _make_index()
        result = idx.upsert_records(records=[{"_id": "r1", "text": "hi"}], namespace="ns")
        assert result.response_info is not None
        assert result.response_info.request_id == "req-abc"
        assert result.response_info.lsn_committed == 42


class TestSearchPopulatesResponseInfo:
    """search() populates response_info from response headers."""

    @respx.mock
    def test_search_populates_response_info(self) -> None:
        respx.post(f"{INDEX_HOST_HTTPS}/records/namespaces/ns/search").mock(
            return_value=httpx.Response(
                200,
                json={"result": {"hits": []}, "usage": {"read_units": 1}},
                headers={"x-pinecone-request-id": "req-search"},
            )
        )
        idx = _make_index()
        result = idx.search(namespace="ns", top_k=5, vector=[0.1])
        assert result.response_info is not None
        assert result.response_info.request_id == "req-search"


class TestFetchPopulatesResponseInfo:
    """fetch() populates response_info from response headers."""

    @respx.mock
    def test_fetch_populates_response_info(self) -> None:
        respx.get(f"{INDEX_HOST_HTTPS}/vectors/fetch").mock(
            return_value=httpx.Response(
                200,
                json={"vectors": {}, "namespace": "", "usage": {"readUnits": 1}},
                headers={"x-pinecone-request-id": "req-fetch"},
            )
        )
        idx = _make_index()
        result = idx.fetch(ids=["id1"])
        assert result.response_info is not None
        assert result.response_info.request_id == "req-fetch"


class TestDescribeIndexStatsPopulatesResponseInfo:
    """describe_index_stats() populates response_info from response headers."""

    @respx.mock
    def test_stats_populates_response_info(self) -> None:
        respx.post(f"{INDEX_HOST_HTTPS}/describe_index_stats").mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": {},
                    "dimension": 128,
                    "indexFullness": 0.0,
                    "totalVectorCount": 0,
                },
                headers={"x-pinecone-request-id": "req-stats"},
            )
        )
        idx = _make_index()
        result = idx.describe_index_stats()
        assert result.response_info is not None
        assert result.response_info.request_id == "req-stats"


class TestUpdatePopulatesResponseInfo:
    """update() populates response_info from response headers."""

    @respx.mock
    def test_update_populates_response_info(self) -> None:
        respx.post(f"{INDEX_HOST_HTTPS}/vectors/update").mock(
            return_value=httpx.Response(
                200,
                json={},
                headers={"x-pinecone-request-id": "req-update"},
            )
        )
        idx = _make_index()
        result = idx.update(id="id1", values=[0.1, 0.2])
        assert result.response_info is not None
        assert result.response_info.request_id == "req-update"


class TestListPaginatedPopulatesResponseInfo:
    """list_paginated() populates response_info from response headers."""

    @respx.mock
    def test_list_paginated_populates_response_info(self) -> None:
        respx.get(f"{INDEX_HOST_HTTPS}/vectors/list").mock(
            return_value=httpx.Response(
                200,
                json={"vectors": [], "namespace": "", "usage": {"readUnits": 1}},
                headers={"x-pinecone-request-id": "req-list"},
            )
        )
        idx = _make_index()
        result = idx.list_paginated()
        assert result.response_info is not None
        assert result.response_info.request_id == "req-list"


class TestFetchByMetadataPopulatesResponseInfo:
    """fetch_by_metadata() populates response_info from response headers."""

    @respx.mock
    def test_fetch_by_metadata_populates_response_info(self) -> None:
        respx.post(f"{INDEX_HOST_HTTPS}/vectors/fetch_by_metadata").mock(
            return_value=httpx.Response(
                200,
                json={"vectors": {}, "namespace": "", "usage": {"readUnits": 1}},
                headers={"x-pinecone-request-id": "req-fbm"},
            )
        )
        idx = _make_index()
        result = idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})
        assert result.response_info is not None
        assert result.response_info.request_id == "req-fbm"
