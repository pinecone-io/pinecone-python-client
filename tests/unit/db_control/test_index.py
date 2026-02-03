import json
import pytest

from pinecone.config import Config, OpenApiConfiguration

from pinecone.db_control.resources.sync.index import IndexResource
from pinecone.openapi_support.api_client import ApiClient
from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi


def build_client_w_faked_response(mocker, body: str, status: int = 200):
    response = mocker.Mock()
    response.headers = {"content-type": "application/json"}
    response.status = status
    # Parse the JSON string into a dict
    response_data = json.loads(body)
    response.data = json.dumps(response_data).encode("utf-8")

    api_client = ApiClient()
    mock_request = mocker.patch.object(
        api_client.rest_client.pool_manager, "request", return_value=response
    )
    index_api = ManageIndexesApi(api_client=api_client)
    resource = IndexResource(
        index_api=index_api,
        config=Config(api_key="test-api-key"),
        openapi_config=OpenApiConfiguration(),
        pool_threads=1,
    )
    return resource, mock_request


class TestIndexResource:
    def test_describe_index(self, mocker):
        body = """
        {
            "name": "test-index",
            "description": "test-description",
            "schema": {
                "fields": {
                    "_values": {
                        "type": "dense_vector",
                        "dimension": 1024,
                        "metric": "cosine"
                    }
                }
            },
            "deployment": {
                "deployment_type": "byoc",
                "environment": "test-environment"
            },
            "status": {
                "ready": true,
                "state": "Ready"
            },
            "host": "test-host.pinecone.io",
            "deletion_protection": "disabled",
            "tags": {
                "test-tag": "test-value"
            }
        }
        """
        index_resource, mock_request = build_client_w_faked_response(mocker, body)

        desc = index_resource.describe(name="test-index")
        assert desc.name == "test-index"
        assert desc.description == "test-description"
        # Test backward compatibility properties
        assert desc.dimension == 1024
        assert desc.metric == "cosine"
        assert desc.spec.byoc.environment == "test-environment"
        assert desc.vector_type == "dense"
        assert desc.status.ready == True
        assert desc.deletion_protection == "disabled"
        assert desc.tags["test-tag"] == "test-value"


class TestIndexResourceCreateValidation:
    """Tests for create() method parameter validation."""

    def test_create_requires_spec_or_schema(self, mocker):
        """Test that create() raises error when neither spec nor schema is provided."""
        body = (
            """{"name": "test", "status": {"ready": true, "state": "Ready"}, "host": "test.io"}"""
        )
        index_resource, _ = build_client_w_faked_response(mocker, body)

        with pytest.raises(ValueError, match="Either 'spec' or 'schema' must be provided"):
            index_resource.create(name="test-index", timeout=-1)

    def test_create_rejects_spec_and_schema_together(self, mocker):
        """Test that create() raises error when both spec and schema are provided."""
        from pinecone.db_control.models import ServerlessSpec, DenseVectorField

        body = (
            """{"name": "test", "status": {"ready": true, "state": "Ready"}, "host": "test.io"}"""
        )
        index_resource, _ = build_client_w_faked_response(mocker, body)

        with pytest.raises(ValueError, match="Cannot specify both 'spec' and 'schema'"):
            index_resource.create(
                name="test-index",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                dimension=1536,
                schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
                timeout=-1,
            )

    def test_create_with_spec_uses_legacy_path(self, mocker):
        """Test that create() with spec uses the legacy request factory method."""
        body = """{
            "name": "test-index",
            "schema": {
                "fields": {
                    "_values": {
                        "type": "dense_vector",
                        "dimension": 1536,
                        "metric": "cosine"
                    }
                }
            },
            "deployment": {"deployment_type": "serverless", "cloud": "aws", "region": "us-east-1"},
            "status": {"ready": true, "state": "Ready"},
            "host": "test.pinecone.io"
        }"""
        index_resource, mock_request = build_client_w_faked_response(mocker, body)

        from pinecone.db_control.models import ServerlessSpec

        result = index_resource.create(
            name="test-index",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            dimension=1536,
            timeout=-1,
        )

        assert result.name == "test-index"
        # Verify the request was made
        assert mock_request.call_count == 1

    def test_create_with_schema_uses_new_path(self, mocker):
        """Test that create() with schema uses the new request factory method."""
        body = """{
            "name": "test-index",
            "deployment": {"deployment_type": "serverless", "cloud": "aws", "region": "us-east-1"},
            "schema": {"fields": {"embedding": {"type": "dense_vector", "dimension": 1536}}},
            "status": {"ready": true, "state": "Ready"},
            "host": "test.pinecone.io"
        }"""
        index_resource, mock_request = build_client_w_faked_response(mocker, body)

        from pinecone.db_control.models import DenseVectorField

        result = index_resource.create(
            name="test-index",
            schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
            timeout=-1,
        )

        assert result.name == "test-index"
        # Verify the request was made
        assert mock_request.call_count == 1

    def test_create_with_schema_and_custom_deployment(self, mocker):
        """Test create() with schema and custom deployment."""
        body = """{
            "name": "test-index",
            "deployment": {"deployment_type": "serverless", "cloud": "gcp", "region": "us-central1"},
            "schema": {"fields": {"embedding": {"type": "dense_vector", "dimension": 1536}}},
            "status": {"ready": true, "state": "Ready"},
            "host": "test.pinecone.io"
        }"""
        index_resource, mock_request = build_client_w_faked_response(mocker, body)

        from pinecone.db_control.models import DenseVectorField, ServerlessDeployment

        result = index_resource.create(
            name="test-index",
            schema={"embedding": DenseVectorField(dimension=1536, metric="cosine")},
            deployment=ServerlessDeployment(cloud="gcp", region="us-central1"),
            timeout=-1,
        )

        assert result.name == "test-index"
        assert mock_request.call_count == 1
