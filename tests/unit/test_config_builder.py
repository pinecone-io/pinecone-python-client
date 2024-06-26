import pytest

from pinecone.core.control.client.configuration import Configuration as OpenApiConfiguration
from pinecone.config import ConfigBuilder
from pinecone import PineconeConfigurationError


class TestConfigBuilder:
    def test_build_simple(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host")
        assert config.api_key == "my-api-key"
        assert config.host == "https://my-controller-host"
        assert config.additional_headers == {}

    def test_build_merges_key_and_host_when_openapi_config_provided(self):
        config = ConfigBuilder.build(
            api_key="my-api-key", host="https://my-controller-host", openapi_config=OpenApiConfiguration()
        )
        assert config.api_key == "my-api-key"
        assert config.host == "https://my-controller-host"
        assert config.additional_headers == {}

    def test_build_with_source_tag(self):
        config = ConfigBuilder.build(
            api_key="my-api-key",
            host="https://my-controller-host",
            source_tag="my-source-tag",
        )
        assert config.api_key == "my-api-key"
        assert config.host == "https://my-controller-host"
        assert config.additional_headers == {}
        assert config.source_tag == "my-source-tag"

    def test_build_errors_when_no_api_key_is_present(self):
        with pytest.raises(PineconeConfigurationError) as e:
            ConfigBuilder.build()
        assert str(e.value) == "You haven't specified an Api-Key."

    def test_build_errors_when_no_host_is_present(self):
        with pytest.raises(PineconeConfigurationError) as e:
            ConfigBuilder.build(api_key="my-api-key")
        assert str(e.value) == "You haven't specified a host."

    def test_build_openapi_config(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host")
        openapi_config = ConfigBuilder.build_openapi_config(config)
        assert openapi_config.host == "https://my-controller-host"
        assert openapi_config.api_key == {"ApiKeyAuth": "my-api-key"}

    def test_build_openapi_config_merges_with_existing_config(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="https://my-controller-host")
        openapi_config = OpenApiConfiguration()
        openapi_config.ssl_ca_cert = "path/to/bundle"
        openapi_config.proxy = "http://my-proxy:8080"

        openapi_config = ConfigBuilder.build_openapi_config(config, openapi_config)

        assert openapi_config.api_key == {"ApiKeyAuth": "my-api-key"}
        assert openapi_config.host == "https://my-controller-host"
        assert openapi_config.ssl_ca_cert == "path/to/bundle"
        assert openapi_config.proxy == "http://my-proxy:8080"

    def test_build_openapi_config_does_not_mutate_input(self):
        config = ConfigBuilder.build(api_key="my-api-key", host="foo", ssl_ca_certs="path/to/bundle.foo")

        input_openapi_config = OpenApiConfiguration()
        input_openapi_config.host = "bar"
        input_openapi_config.ssl_ca_cert = "asdfasdf"

        openapi_config = ConfigBuilder.build_openapi_config(config, input_openapi_config)
        assert openapi_config.host == "https://foo"
        assert openapi_config.ssl_ca_cert == "path/to/bundle.foo"

        assert input_openapi_config.host == "bar"
        assert input_openapi_config.ssl_ca_cert == "asdfasdf"
