import pytest
import os

from pinecone import Pinecone
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from urllib3 import make_headers


@pytest.mark.skipif(os.getenv("USE_GRPC") != "false", reason="Only test when using REST")
class TestIndexOpenapiConfig:
    def test_passing_openapi_config(self, api_key_fixture, index_host):
        oai_config = OpenApiConfiguration.get_default_copy()
        p = Pinecone(api_key=api_key_fixture, openapi_config=oai_config)
        assert p.config.api_key == api_key_fixture
        p.list_indexes()  # should not throw

        index = p.Index(host=index_host)
        assert index._config.api_key == api_key_fixture
        index.describe_index_stats()
