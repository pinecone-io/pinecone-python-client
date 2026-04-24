"""Tests for extract_response_info — header lowercasing and allocation."""

from __future__ import annotations

import tracemalloc

import httpx

from pinecone._internal.adapters.vectors_adapter import extract_response_info
from pinecone.models.response_info import ResponseInfo


class TestHttpxNormalizesHeadersToLowercase:
    """httpx normalizes header names to lowercase, so dict(response.headers) is safe."""

    def test_mixed_case_headers_become_lowercase(self) -> None:
        response = httpx.Response(
            200,
            headers={
                "X-Pinecone-Request-Id": "req-abc",
                "X-Pinecone-Lsn-Reconciled": "10",
                "X-Pinecone-Lsn-Committed": "9",
                "Content-Type": "application/json",
            },
        )
        assert dict(response.headers) == {
            "x-pinecone-request-id": "req-abc",
            "x-pinecone-lsn-reconciled": "10",
            "x-pinecone-lsn-committed": "9",
            "content-type": "application/json",
        }

    def test_already_lowercase_headers_unaffected(self) -> None:
        response = httpx.Response(
            200,
            headers={
                "x-pinecone-request-id": "req-xyz",
            },
        )
        info = extract_response_info(response)
        assert info.request_id == "req-xyz"

    def test_uppercase_source_headers_accessible_via_typed_properties(self) -> None:
        response = httpx.Response(
            200,
            headers={
                "X-Pinecone-Request-Id": "req-123",
                "X-Pinecone-Lsn-Reconciled": "55",
                "X-Pinecone-Lsn-Committed": "50",
            },
        )
        info = extract_response_info(response)
        assert info.request_id == "req-123"
        assert info.lsn_reconciled == 55
        assert info.lsn_committed == 50

    def test_missing_headers_return_none(self) -> None:
        response = httpx.Response(200, headers={})
        info = extract_response_info(response)
        assert info.request_id is None
        assert info.lsn_reconciled is None
        assert info.lsn_committed is None

    def test_invalid_lsn_value_returns_none(self) -> None:
        response = httpx.Response(
            200,
            headers={"x-pinecone-lsn-reconciled": "not-an-int"},
        )
        info = extract_response_info(response)
        assert info.lsn_reconciled is None

    def test_is_reconciled_false_when_lsn_below_target(self) -> None:
        response = httpx.Response(
            200,
            headers={"x-pinecone-lsn-reconciled": "9"},
        )
        info = extract_response_info(response)
        assert not info.is_reconciled(10)

    def test_is_reconciled_true_when_lsn_meets_target(self) -> None:
        response = httpx.Response(
            200,
            headers={"x-pinecone-lsn-reconciled": "10"},
        )
        info = extract_response_info(response)
        assert info.is_reconciled(10)

    def test_returns_response_info_instance(self) -> None:
        response = httpx.Response(200, headers={})
        assert isinstance(extract_response_info(response), ResponseInfo)


class TestExtractResponseInfoAllocation:
    """extract_response_info must allocate at most one object in the common path.

    Uses tracemalloc to count allocations. One dict allocation (raw_headers) plus
    one ResponseInfo struct allocation is the expected baseline; no extra per-key
    str.lower() allocations.
    """

    def test_single_dict_allocation_for_ten_headers(self) -> None:
        headers = {f"x-header-{i}": f"value-{i}" for i in range(10)}
        response = httpx.Response(200, headers=headers)

        tracemalloc.start()
        tracemalloc.clear_traces()
        _ = extract_response_info(response)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # filter_traces returns a new Snapshot; .statistics() returns the list
        filtered = snapshot.filter_traces((tracemalloc.Filter(True, "*vectors_adapter*"),))
        stats = filtered.statistics("filename")
        # One allocation: the dict passed to ResponseInfo.__init__
        assert len(stats) <= 2, (
            f"Expected ≤2 allocations from vectors_adapter, got {len(stats)}: {stats}"
        )
