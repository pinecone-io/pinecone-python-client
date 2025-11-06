"""Unit tests for LSN utilities."""

from tests.integration.helpers.lsn_utils import (
    extract_lsn_reconciled,
    extract_lsn_committed,
    extract_lsn_values,
    is_lsn_reconciled,
    get_headers_from_response,
)
from pinecone.openapi_support.rest_utils import RESTResponse


class TestExtractLSNReconciled:
    """Tests for extract_lsn_reconciled function."""

    def test_extract_standard_header(self):
        """Test extraction with standard header name."""
        headers = {"x-pinecone-max-indexed-lsn": "100"}
        assert extract_lsn_reconciled(headers) == 100

    def test_case_insensitive(self):
        """Test that header matching is case-insensitive."""
        headers = {"X-PINECONE-MAX-INDEXED-LSN": "500"}
        assert extract_lsn_reconciled(headers) == 500

    def test_missing_header(self):
        """Test that None is returned when header is missing."""
        headers = {"other-header": "value"}
        assert extract_lsn_reconciled(headers) is None

    def test_empty_headers(self):
        """Test that None is returned for empty headers."""
        assert extract_lsn_reconciled({}) is None
        assert extract_lsn_reconciled(None) is None

    def test_invalid_value(self):
        """Test that None is returned for invalid values."""
        headers = {"x-pinecone-max-indexed-lsn": "not-a-number"}
        assert extract_lsn_reconciled(headers) is None


class TestExtractLSNCommitted:
    """Tests for extract_lsn_committed function."""

    def test_extract_standard_header(self):
        """Test extraction with standard header name."""
        headers = {"x-pinecone-request-lsn": "150"}
        assert extract_lsn_committed(headers) == 150

    def test_case_insensitive(self):
        """Test that header matching is case-insensitive."""
        headers = {"X-PINECONE-REQUEST-LSN": "550"}
        assert extract_lsn_committed(headers) == 550

    def test_missing_header(self):
        """Test that None is returned when header is missing."""
        headers = {"other-header": "value"}
        assert extract_lsn_committed(headers) is None


class TestExtractLSNValues:
    """Tests for extract_lsn_values function."""

    def test_extract_both_values(self):
        """Test extraction of both reconciled and committed."""
        headers = {"x-pinecone-max-indexed-lsn": "100", "x-pinecone-request-lsn": "150"}
        reconciled, committed = extract_lsn_values(headers)
        assert reconciled == 100
        assert committed == 150

    def test_extract_only_reconciled(self):
        """Test extraction when only reconciled is present."""
        headers = {"x-pinecone-max-indexed-lsn": "100"}
        reconciled, committed = extract_lsn_values(headers)
        assert reconciled == 100
        assert committed is None

    def test_extract_only_committed(self):
        """Test extraction when only committed is present."""
        headers = {"x-pinecone-request-lsn": "150"}
        reconciled, committed = extract_lsn_values(headers)
        assert reconciled is None
        assert committed == 150

    def test_extract_neither(self):
        """Test extraction when neither is present."""
        headers = {"other-header": "value"}
        reconciled, committed = extract_lsn_values(headers)
        assert reconciled is None
        assert committed is None


class TestIsLSNReconciled:
    """Tests for is_lsn_reconciled function."""

    def test_reconciled_when_equal(self):
        """Test that LSN is considered reconciled when equal."""
        assert is_lsn_reconciled(100, 100) is True

    def test_reconciled_when_greater(self):
        """Test that LSN is considered reconciled when reconciled > target."""
        assert is_lsn_reconciled(100, 150) is True

    def test_not_reconciled_when_less(self):
        """Test that LSN is not reconciled when reconciled < target."""
        assert is_lsn_reconciled(100, 50) is False

    def test_none_reconciled_lsn(self):
        """Test that False is returned when reconciled LSN is None."""
        assert is_lsn_reconciled(100, None) is False


class TestGetHeadersFromResponse:
    """Tests for get_headers_from_response function."""

    def test_tuple_response(self):
        """Test extraction from tuple response."""
        headers_dict = {"x-pinecone-max-indexed-lsn": "100"}
        response = ("data", 200, headers_dict)
        assert get_headers_from_response(response) == headers_dict

    def test_rest_response_object(self):
        """Test extraction from RESTResponse object."""
        headers_dict = {"x-pinecone-max-indexed-lsn": "100"}
        response = RESTResponse(200, b"data", headers_dict, "OK")
        assert get_headers_from_response(response) == headers_dict

    def test_dict_response(self):
        """Test extraction from dict response."""
        headers_dict = {"x-pinecone-max-indexed-lsn": "100"}
        assert get_headers_from_response(headers_dict) == headers_dict

    def test_invalid_response(self):
        """Test that None is returned for invalid response types."""
        assert get_headers_from_response("string") is None
        assert get_headers_from_response(123) is None
        assert get_headers_from_response(None) is None

    def test_rest_response_without_getheaders(self):
        """Test handling of object without getheaders method."""

        class MockResponse:
            pass

        assert get_headers_from_response(MockResponse()) is None
