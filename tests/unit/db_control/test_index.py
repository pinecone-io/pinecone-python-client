import json

from pinecone import Config

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
    return IndexResource(index_api=index_api, config=Config(api_key="test-api-key")), mock_request


class TestIndexResource:
    def test_describe_index(self, mocker):
        body = """
        {
            "name": "test-index",
            "description": "test-description",
            "dimension": 1024,
            "metric": "cosine",
            "spec": {
                "byoc": {
                    "environment": "test-environment"
                }
            },
            "vector_type": "dense",
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
        assert desc.dimension == 1024
        assert desc.metric == "cosine"
        assert desc.spec.byoc.environment == "test-environment"
        assert desc.vector_type == "dense"
        assert desc.status.ready == True
        assert desc.deletion_protection == "disabled"
        assert desc.tags["test-tag"] == "test-value"
