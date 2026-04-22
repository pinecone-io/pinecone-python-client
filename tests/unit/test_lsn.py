"""Tests for LSN extraction from response headers and ResponseInfo.is_reconciled()."""

from __future__ import annotations

import httpx

from pinecone._internal.adapters.vectors_adapter import extract_response_info
from pinecone.models.vectors.responses import ResponseInfo


class TestParseLsnFromHeaders:
    """Tests for LSN header extraction via extract_response_info()."""

    @staticmethod
    def _make_response(headers: dict[str, str]) -> httpx.Response:
        """Build a minimal httpx.Response with the given headers."""
        return httpx.Response(status_code=200, headers=headers)

    def test_lsn_reconciled_extracted(self) -> None:
        response = self._make_response({"X-Pinecone-LSN-Reconciled": "42"})
        info = extract_response_info(response)
        assert info.lsn_reconciled == 42

    def test_lsn_committed_extracted(self) -> None:
        response = self._make_response({"X-Pinecone-LSN-Committed": "100"})
        info = extract_response_info(response)
        assert info.lsn_committed == 100

    def test_lsn_absent_returns_none(self) -> None:
        response = self._make_response({})
        info = extract_response_info(response)
        assert info.lsn_reconciled is None
        assert info.lsn_committed is None

    def test_lsn_invalid_returns_none(self) -> None:
        response = self._make_response(
            {
                "X-Pinecone-LSN-Reconciled": "not-a-number",
                "X-Pinecone-LSN-Committed": "also-bad",
            }
        )
        info = extract_response_info(response)
        assert info.lsn_reconciled is None
        assert info.lsn_committed is None

    def test_lsn_case_insensitive(self) -> None:
        response = self._make_response({"x-pinecone-lsn-reconciled": "5"})
        info = extract_response_info(response)
        assert info.lsn_reconciled == 5

    def test_request_id_still_extracted(self) -> None:
        response = self._make_response(
            {
                "x-pinecone-request-id": "req-abc",
                "X-Pinecone-LSN-Reconciled": "10",
            }
        )
        info = extract_response_info(response)
        assert info.request_id == "req-abc"
        assert info.lsn_reconciled == 10


class TestIsReconciled:
    """Tests for ResponseInfo.is_reconciled()."""

    def test_is_reconciled_true_equal(self) -> None:
        info = ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "10"})
        assert info.is_reconciled(10) is True

    def test_is_reconciled_true_exceeds(self) -> None:
        info = ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "10"})
        assert info.is_reconciled(5) is True

    def test_is_reconciled_false(self) -> None:
        info = ResponseInfo(raw_headers={"x-pinecone-lsn-reconciled": "5"})
        assert info.is_reconciled(10) is False

    def test_is_reconciled_none(self) -> None:
        info = ResponseInfo(raw_headers={})
        assert info.is_reconciled(1) is False

    def test_is_reconciled_default_none(self) -> None:
        info = ResponseInfo()
        assert info.is_reconciled(1) is False


class TestRawHeaders:
    """Tests for raw_headers capture and normalization."""

    @staticmethod
    def _make_response(headers: dict[str, str]) -> httpx.Response:
        return httpx.Response(status_code=200, headers=headers)

    def test_raw_headers_default_empty(self) -> None:
        info = ResponseInfo()
        assert info.raw_headers == {}

    def test_raw_headers_captured_from_response(self) -> None:
        response = self._make_response(
            {"X-Pinecone-Request-Id": "r1", "X-Custom-Header": "v1", "X-Ratelimit-Remaining": "7"}
        )
        info = extract_response_info(response)
        assert info.raw_headers["x-pinecone-request-id"] == "r1"
        assert info.raw_headers["x-custom-header"] == "v1"
        assert info.raw_headers["x-ratelimit-remaining"] == "7"

    def test_raw_headers_keys_lowercased(self) -> None:
        response = self._make_response({"X-Pinecone-Request-Id": "r-mixed-case"})
        info = extract_response_info(response)
        assert "x-pinecone-request-id" in info.raw_headers
        assert "X-Pinecone-Request-Id" not in info.raw_headers

    def test_raw_headers_empty_for_no_headers_response(self) -> None:
        response = self._make_response({})
        info = extract_response_info(response)
        assert "x-pinecone-request-id" not in info.raw_headers

    def test_raw_headers_arbitrary_header_accessible(self) -> None:
        response = self._make_response({"x-ratelimit-remaining": "42", "retry-after": "60"})
        info = extract_response_info(response)
        assert info.raw_headers["x-ratelimit-remaining"] == "42"
        assert info.raw_headers["retry-after"] == "60"
