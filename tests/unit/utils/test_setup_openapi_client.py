import re
from pinecone.config import ConfigBuilder
from pinecone.core.client.api.manage_indexes_api import ManageIndexesApi
from pinecone.utils.setup_openapi_client import setup_openapi_client

class TestSetupOpenAPIClient():
    def test_setup_openapi_client(self):
        ""
        # config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host")
        # api_client = setup_openapi_client(ManageIndexesApi, config=config, pool_threads=2)
        # # assert api_client.user_agent == "pinecone-python-client/0.0.1"
