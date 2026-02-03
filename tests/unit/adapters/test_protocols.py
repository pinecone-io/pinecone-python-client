"""Unit tests for adapter protocol compliance.

These tests verify that the actual OpenAPI models satisfy the protocol
interfaces defined in pinecone.adapters.protocols. This ensures that the
adapter layer's contracts are maintained even as the OpenAPI models change.
"""

from pinecone.adapters.protocols import (
    QueryResponseAdapter,
    UpsertResponseAdapter,
    FetchResponseAdapter,
    IndexModelAdapter,
    IndexStatusAdapter,
)
from pinecone.adapters.response_adapters import (
    adapt_query_response,
    adapt_upsert_response,
    adapt_fetch_response,
)
from tests.fixtures import (
    make_openapi_query_response,
    make_openapi_upsert_response,
    make_openapi_fetch_response,
)


class TestQueryResponseProtocolCompliance:
    """Tests that OpenAPI QueryResponse satisfies QueryResponseAdapter protocol."""

    def test_has_matches_attribute(self):
        """Test that QueryResponse has matches attribute."""
        response = make_openapi_query_response(matches=[])
        # This satisfies the protocol check
        _protocol_check: QueryResponseAdapter = response
        assert hasattr(response, "matches")

    def test_has_namespace_attribute(self):
        """Test that QueryResponse has namespace attribute."""
        response = make_openapi_query_response(matches=[], namespace="test")
        _protocol_check: QueryResponseAdapter = response
        assert hasattr(response, "namespace")

    def test_has_usage_attribute(self):
        """Test that QueryResponse has usage attribute."""
        response = make_openapi_query_response(matches=[])
        _protocol_check: QueryResponseAdapter = response
        assert hasattr(response, "usage")

    def test_has_data_store_attribute(self):
        """Test that QueryResponse has _data_store attribute."""
        response = make_openapi_query_response(matches=[])
        _protocol_check: QueryResponseAdapter = response
        assert hasattr(response, "_data_store")

    def test_has_response_info_attribute(self):
        """Test that QueryResponse has _response_info attribute."""
        response = make_openapi_query_response(matches=[])
        _protocol_check: QueryResponseAdapter = response
        assert hasattr(response, "_response_info")


class TestUpsertResponseProtocolCompliance:
    """Tests that OpenAPI UpsertResponse satisfies UpsertResponseAdapter protocol."""

    def test_has_upserted_count_attribute(self):
        """Test that UpsertResponse has upserted_count attribute."""
        response = make_openapi_upsert_response(upserted_count=10)
        _protocol_check: UpsertResponseAdapter = response
        assert hasattr(response, "upserted_count")
        assert response.upserted_count == 10

    def test_has_response_info_attribute(self):
        """Test that UpsertResponse has _response_info attribute."""
        response = make_openapi_upsert_response(upserted_count=10)
        _protocol_check: UpsertResponseAdapter = response
        assert hasattr(response, "_response_info")


class TestFetchResponseProtocolCompliance:
    """Tests that OpenAPI FetchResponse satisfies FetchResponseAdapter protocol."""

    def test_has_namespace_attribute(self):
        """Test that FetchResponse has namespace attribute."""
        response = make_openapi_fetch_response(vectors={}, namespace="test")
        _protocol_check: FetchResponseAdapter = response
        assert hasattr(response, "namespace")
        assert response.namespace == "test"

    def test_has_vectors_attribute(self):
        """Test that FetchResponse has vectors attribute."""
        response = make_openapi_fetch_response(vectors={})
        _protocol_check: FetchResponseAdapter = response
        assert hasattr(response, "vectors")

    def test_has_usage_attribute(self):
        """Test that FetchResponse has usage attribute."""
        response = make_openapi_fetch_response(vectors={})
        _protocol_check: FetchResponseAdapter = response
        assert hasattr(response, "usage")

    def test_has_response_info_attribute(self):
        """Test that FetchResponse has _response_info attribute."""
        response = make_openapi_fetch_response(vectors={})
        _protocol_check: FetchResponseAdapter = response
        assert hasattr(response, "_response_info")


class TestIndexModelProtocolCompliance:
    """Tests that OpenAPI IndexModel satisfies IndexModelAdapter protocol."""

    def test_openapi_index_model_has_required_attributes(self):
        """Test that OpenAPI IndexModel has all required protocol attributes."""
        from pinecone.core.openapi.db_control.model.index_model import (
            IndexModel as OpenAPIIndexModel,
        )
        from pinecone.core.openapi.db_control.model.index_model_status import IndexModelStatus

        # Create a minimal OpenAPI IndexModel
        index = OpenAPIIndexModel._new_from_openapi_data(
            name="test-index",
            dimension=128,
            metric="cosine",
            host="test-host.pinecone.io",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            status=IndexModelStatus._new_from_openapi_data(ready=True, state="Ready"),
        )

        # This satisfies the protocol check
        _protocol_check: IndexModelAdapter = index

        # Verify all required attributes exist
        assert hasattr(index, "name")
        assert hasattr(index, "dimension")
        assert hasattr(index, "metric")
        assert hasattr(index, "host")
        assert hasattr(index, "spec")
        assert hasattr(index, "status")
        assert hasattr(index, "_data_store")
        assert hasattr(index, "_configuration")
        assert hasattr(index, "_path_to_item")
        assert hasattr(index, "to_dict")
        assert callable(index.to_dict)


class TestIndexStatusProtocolCompliance:
    """Tests that IndexModelStatus satisfies IndexStatusAdapter protocol."""

    def test_openapi_index_status_has_required_attributes(self):
        """Test that IndexModelStatus has all required protocol attributes."""
        from pinecone.core.openapi.db_control.model.index_model_status import IndexModelStatus

        status = IndexModelStatus._new_from_openapi_data(ready=True, state="Ready")

        # This satisfies the protocol check
        _protocol_check: IndexStatusAdapter = status

        # Verify all required attributes exist
        assert hasattr(status, "ready")
        assert hasattr(status, "state")
        assert status.ready is True
        assert status.state == "Ready"


class TestAdapterNoneHandling:
    """Tests that adapters handle None/optional fields correctly."""

    def test_adapt_query_response_with_none_matches(self):
        """Test that adapt_query_response handles None matches gracefully."""
        openapi_response = make_openapi_query_response(matches=None, namespace="test")
        sdk_response = adapt_query_response(openapi_response)

        assert sdk_response.matches == []
        assert sdk_response.namespace == "test"

    def test_adapt_query_response_with_none_namespace(self):
        """Test that adapt_query_response handles None namespace gracefully."""
        openapi_response = make_openapi_query_response(matches=[], namespace=None)
        sdk_response = adapt_query_response(openapi_response)

        assert sdk_response.matches == []
        assert sdk_response.namespace == ""

    def test_adapt_upsert_response_with_none_upserted_count(self):
        """Test that adapt_upsert_response handles None upserted_count gracefully."""
        openapi_response = make_openapi_upsert_response(upserted_count=None)
        sdk_response = adapt_upsert_response(openapi_response)

        assert sdk_response.upserted_count == 0

    def test_adapt_fetch_response_with_none_namespace(self):
        """Test that adapt_fetch_response handles None namespace gracefully."""
        openapi_response = make_openapi_fetch_response(vectors={}, namespace=None)
        sdk_response = adapt_fetch_response(openapi_response)

        assert sdk_response.namespace == ""
        assert sdk_response.vectors == {}

    def test_adapt_fetch_response_with_none_vectors(self):
        """Test that adapt_fetch_response handles None vectors gracefully."""
        openapi_response = make_openapi_fetch_response(vectors=None, namespace="test")
        sdk_response = adapt_fetch_response(openapi_response)

        assert sdk_response.namespace == "test"
        assert sdk_response.vectors == {}

    def test_adapt_fetch_response_with_all_none_optionals(self):
        """Test that adapt_fetch_response handles all None optional fields."""
        openapi_response = make_openapi_fetch_response(vectors=None, namespace=None)
        sdk_response = adapt_fetch_response(openapi_response)

        assert sdk_response.namespace == ""
        assert sdk_response.vectors == {}
