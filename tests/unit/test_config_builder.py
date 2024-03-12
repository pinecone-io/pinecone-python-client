import pytest

from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from pinecone.config import ConfigBuilder
from pinecone import PineconeConfigurationError

class TestConfigBuilder:
    def test_build_simple(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host")
        assert config.api_key == "my-api-key"
        assert config.host == "https://my-controller-host"
        assert config.additional_headers == {}
        assert config.openapi_config.host == "https://my-controller-host"
        assert config.openapi_config.api_key == {"ApiKeyAuth": "my-api-key"}

    def test_build_merges_key_and_host_when_openapi_config_provided(self):
        config = ConfigBuilder.build(
            api_key="my-api-key", 
            host="https://my-controller-host", 
            openapi_config=OpenApiConfiguration()
        )
        assert config.api_key == "my-api-key"
        assert config.host == "https://my-controller-host"
        assert config.additional_headers == {}
        assert config.openapi_config.host == "https://my-controller-host"
        assert config.openapi_config.api_key == {"ApiKeyAuth": "my-api-key"}

    def test_build_errors_when_no_api_key_is_present(self):
        with pytest.raises(PineconeConfigurationError) as e:
            ConfigBuilder.build()
        assert str(e.value) == "You haven't specified an Api-Key."

    def test_build_errors_when_no_host_is_present(self):
        with pytest.raises(PineconeConfigurationError) as e:
            ConfigBuilder.build(api_key='my-api-key')
        assert str(e.value) == "You haven't specified a host."