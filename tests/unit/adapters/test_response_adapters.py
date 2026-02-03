"""Unit tests for response adapter functions."""

from pinecone.adapters import adapt_query_response, adapt_upsert_response, adapt_fetch_response
from pinecone.db_data.dataclasses import QueryResponse, UpsertResponse, FetchResponse
from tests.fixtures import (
    make_openapi_query_response,
    make_openapi_upsert_response,
    make_openapi_fetch_response,
    make_scored_vector,
    make_usage,
)


class TestAdaptQueryResponse:
    """Tests for adapt_query_response function."""

    def test_basic_query_response(self):
        """Test adapting a basic query response with matches."""
        match1 = make_scored_vector(id="vec1", score=0.95)
        match2 = make_scored_vector(id="vec2", score=0.85)
        openapi_response = make_openapi_query_response(
            matches=[match1, match2], namespace="test-ns"
        )

        result = adapt_query_response(openapi_response)

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        assert result.matches[0].id == "vec1"
        assert result.matches[1].id == "vec2"
        assert result.namespace == "test-ns"

    def test_query_response_with_empty_namespace(self):
        """Test that empty namespace is handled correctly."""
        openapi_response = make_openapi_query_response(matches=[], namespace="")

        result = adapt_query_response(openapi_response)

        assert result.namespace == ""

    def test_query_response_with_none_namespace(self):
        """Test that None namespace is converted to empty string."""
        openapi_response = make_openapi_query_response(matches=[])
        # Manually set namespace to None to simulate API response
        openapi_response._data_store["namespace"] = None

        result = adapt_query_response(openapi_response)

        assert result.namespace == ""

    def test_query_response_with_usage(self):
        """Test adapting query response with usage information."""
        usage = make_usage(read_units=10)
        openapi_response = make_openapi_query_response(matches=[], namespace="", usage=usage)

        result = adapt_query_response(openapi_response)

        assert result.usage is not None
        assert result.usage.read_units == 10

    def test_query_response_without_usage(self):
        """Test that missing usage is handled as None."""
        openapi_response = make_openapi_query_response(matches=[])

        result = adapt_query_response(openapi_response)

        assert result.usage is None

    def test_query_response_removes_deprecated_results_field(self):
        """Test that deprecated 'results' field is removed from _data_store."""
        openapi_response = make_openapi_query_response(matches=[])
        openapi_response._data_store["results"] = [{"deprecated": "data"}]

        adapt_query_response(openapi_response)

        assert "results" not in openapi_response._data_store

    def test_query_response_has_response_info(self):
        """Test that response info is extracted."""
        openapi_response = make_openapi_query_response(matches=[])

        result = adapt_query_response(openapi_response)

        # Should have response_info with raw_headers
        assert hasattr(result, "_response_info")
        assert "raw_headers" in result._response_info


class TestAdaptUpsertResponse:
    """Tests for adapt_upsert_response function."""

    def test_basic_upsert_response(self):
        """Test adapting a basic upsert response."""
        openapi_response = make_openapi_upsert_response(upserted_count=100)

        result = adapt_upsert_response(openapi_response)

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 100

    def test_upsert_response_zero_count(self):
        """Test adapting upsert response with zero count."""
        openapi_response = make_openapi_upsert_response(upserted_count=0)

        result = adapt_upsert_response(openapi_response)

        assert result.upserted_count == 0

    def test_upsert_response_has_response_info(self):
        """Test that response info is extracted."""
        openapi_response = make_openapi_upsert_response(upserted_count=1)

        result = adapt_upsert_response(openapi_response)

        assert hasattr(result, "_response_info")
        assert "raw_headers" in result._response_info


class TestAdaptFetchResponse:
    """Tests for adapt_fetch_response function."""

    def test_basic_fetch_response(self):
        """Test adapting a basic fetch response with vectors."""
        vectors = {
            "vec1": {"id": "vec1", "values": [0.1, 0.2, 0.3]},
            "vec2": {"id": "vec2", "values": [0.4, 0.5, 0.6]},
        }
        openapi_response = make_openapi_fetch_response(vectors=vectors, namespace="test-ns")

        result = adapt_fetch_response(openapi_response)

        assert isinstance(result, FetchResponse)
        assert len(result.vectors) == 2
        assert "vec1" in result.vectors
        assert "vec2" in result.vectors
        assert result.namespace == "test-ns"

    def test_fetch_response_empty_vectors(self):
        """Test adapting fetch response with no vectors."""
        openapi_response = make_openapi_fetch_response(vectors={})

        result = adapt_fetch_response(openapi_response)

        assert result.vectors == {}

    def test_fetch_response_with_usage(self):
        """Test adapting fetch response with usage information."""
        usage = make_usage(read_units=5)
        openapi_response = make_openapi_fetch_response(vectors={}, usage=usage)

        result = adapt_fetch_response(openapi_response)

        assert result.usage is not None
        assert result.usage.read_units == 5

    def test_fetch_response_converts_vectors_to_sdk_type(self):
        """Test that vectors are converted to SDK Vector type."""
        from pinecone.db_data.dataclasses import Vector

        vectors = {"vec1": {"id": "vec1", "values": [0.1, 0.2, 0.3]}}
        openapi_response = make_openapi_fetch_response(vectors=vectors)

        result = adapt_fetch_response(openapi_response)

        assert isinstance(result.vectors["vec1"], Vector)
        assert result.vectors["vec1"].id == "vec1"
        assert result.vectors["vec1"].values == [0.1, 0.2, 0.3]

    def test_fetch_response_has_response_info(self):
        """Test that response info is extracted."""
        openapi_response = make_openapi_fetch_response(vectors={})

        result = adapt_fetch_response(openapi_response)

        assert hasattr(result, "_response_info")
        assert "raw_headers" in result._response_info

    def test_fetch_response_with_none_namespace(self):
        """Test that None namespace is converted to empty string."""
        openapi_response = make_openapi_fetch_response(vectors={})
        # Manually set namespace to None to simulate API response
        openapi_response._data_store["namespace"] = None

        result = adapt_fetch_response(openapi_response)

        assert result.namespace == ""
