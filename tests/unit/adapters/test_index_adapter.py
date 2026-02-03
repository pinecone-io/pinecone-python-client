"""Unit tests for index adapter functions.

Tests the adapt_index_spec function that handles oneOf schema resolution
for IndexModel spec fields.
"""

from pinecone.adapters import adapt_index_spec
from pinecone.core.openapi.db_control.model.serverless import Serverless
from pinecone.core.openapi.db_control.model.pod_based import PodBased
from pinecone.core.openapi.db_control.model.byoc import BYOC
from tests.fixtures import make_index_model


class TestAdaptIndexSpec:
    """Test adapt_index_spec with different spec types."""

    def test_adapt_serverless_spec_basic(self):
        """Test adapting a basic serverless spec without read_capacity."""
        openapi_model = make_index_model(
            name="test-index", spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )

        result = adapt_index_spec(openapi_model)

        assert result is not None
        assert isinstance(result, Serverless)
        assert hasattr(result, "serverless")
        assert result.serverless.cloud == "aws"
        assert result.serverless.region == "us-east-1"

    def test_adapt_pod_spec(self):
        """Test adapting a pod-based spec."""
        openapi_model = make_index_model(
            name="test-index",
            spec={
                "pod": {
                    "environment": "us-east-1-aws",
                    "replicas": 1,
                    "shards": 1,
                    "pod_type": "p1.x1",
                    "pods": 1,
                }
            },
        )

        result = adapt_index_spec(openapi_model)

        assert result is not None
        assert isinstance(result, PodBased)
        assert hasattr(result, "pod")
        assert result.pod.environment == "us-east-1-aws"
        assert result.pod.replicas == 1
        assert result.pod.shards == 1
        assert result.pod.pod_type == "p1.x1"

    def test_adapt_byoc_spec(self):
        """Test adapting a BYOC (Bring Your Own Cloud) spec."""
        openapi_model = make_index_model(
            name="test-index", spec={"byoc": {"environment": "custom-env"}}
        )

        result = adapt_index_spec(openapi_model)

        assert result is not None
        assert isinstance(result, BYOC)
        assert hasattr(result, "byoc")
        assert result.byoc.environment == "custom-env"

    def test_adapt_spec_returns_none_when_spec_is_none(self):
        """Test that None is returned when spec is not present."""
        # Create a model without spec in _data_store
        openapi_model = make_index_model(name="test-index")
        # Manually remove spec from _data_store to simulate missing spec
        if "spec" in openapi_model._data_store:
            del openapi_model._data_store["spec"]

        result = adapt_index_spec(openapi_model)

        assert result is None

    def test_adapt_spec_caching_in_index_model(self):
        """Test that IndexModel properly caches the adapted spec."""
        from pinecone.db_control.models import IndexModel

        openapi_model = make_index_model(
            name="test-index", spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )

        wrapped = IndexModel(openapi_model)

        # First access should populate cache
        spec1 = wrapped.spec
        assert spec1 is not None

        # Second access should return cached value
        spec2 = wrapped.spec
        assert spec2 is spec1  # Same object instance

    def test_adapt_spec_handles_already_deserialized_spec(self):
        """Test that adapter handles specs that are already IndexSpec instances."""
        from pinecone.core.openapi.db_control.model.serverless import Serverless
        from pinecone.core.openapi.db_control.model.serverless_spec_response import (
            ServerlessSpecResponse,
        )

        # Create a fully deserialized spec
        serverless_spec = ServerlessSpecResponse._from_openapi_data(
            cloud="aws", region="us-east-1", read_capacity=None, _check_type=False
        )
        already_deserialized = Serverless._new_from_openapi_data(
            serverless=serverless_spec, _check_type=False
        )

        openapi_model = make_index_model(name="test-index")
        # Replace the dict spec with an already-deserialized one
        openapi_model._data_store["spec"] = already_deserialized

        result = adapt_index_spec(openapi_model)

        # Should return the already-deserialized spec as-is
        assert result is already_deserialized


class TestIndexModelAdapterProtocolCompliance:
    """Test that OpenAPI IndexModel conforms to IndexModelAdapter protocol."""

    def test_openapi_index_model_has_data_store(self):
        """Verify OpenAPI IndexModel has _data_store attribute."""
        from pinecone.adapters.protocols import IndexModelAdapter

        openapi_model = make_index_model()
        _protocol_check: IndexModelAdapter = openapi_model
        assert hasattr(openapi_model, "_data_store")
        assert isinstance(openapi_model._data_store, dict)

    def test_openapi_index_model_has_configuration(self):
        """Verify OpenAPI IndexModel has _configuration attribute."""
        from pinecone.adapters.protocols import IndexModelAdapter

        openapi_model = make_index_model()
        _protocol_check: IndexModelAdapter = openapi_model
        assert hasattr(openapi_model, "_configuration")

    def test_openapi_index_model_has_path_to_item(self):
        """Verify OpenAPI IndexModel has _path_to_item attribute."""
        from pinecone.adapters.protocols import IndexModelAdapter

        openapi_model = make_index_model()
        _protocol_check: IndexModelAdapter = openapi_model
        assert hasattr(openapi_model, "_path_to_item")
