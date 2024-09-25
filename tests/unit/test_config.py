from pinecone import Pinecone
from pinecone.exceptions.exceptions import PineconeConfigurationError
from pinecone.config import PineconeConfig
from pinecone.core.openapi.shared.configuration import Configuration as OpenApiConfiguration

import pytest
import os

from urllib3 import make_headers


class TestConfig:
    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(tmpdir):
        """Fixture to execute asserts before and after a test is run"""

        # Defend against unexpected env vars. Since we clear these variables below
        # after each test execution, these should only be raised if there is
        # test pollution in the environment coming from some other test file/setup.
        known_env_vars = [
            "PINECONE_API_KEY",
            "PINECONE_ENVIRONMENT",
            "PINECONE_CONTROLLER_HOST",
            "PINECONE_ADDITIONAL_HEADERS",
        ]
        for var in known_env_vars:
            if os.getenv(var):
                raise ValueError(
                    f"Unexpected env var {var} found in environment. Check for test pollution."
                )

        yield  # this is where the testing happens

        # Teardown : Unset any env vars created during test execution
        for var in known_env_vars:
            if os.getenv(var):
                del os.environ[var]

    def test_init_with_environment_vars(self):
        os.environ["PINECONE_API_KEY"] = "test-api-key"
        os.environ["PINECONE_CONTROLLER_HOST"] = "https://test-controller-host"
        os.environ["PINECONE_ADDITIONAL_HEADERS"] = '{"header": "value"}'

        config = PineconeConfig.build()

        assert config.api_key == "test-api-key"
        assert config.host == "https://test-controller-host"
        assert config.additional_headers == {"header": "value"}

    def test_init_with_positional_args(self):
        api_key = "my-api-key"
        host = "https://my-controller-host"

        config = PineconeConfig.build(api_key, host)

        assert config.api_key == api_key
        assert config.host == host

    def test_init_with_kwargs(self):
        api_key = "my-api-key"
        controller_host = "my-controller-host"
        ssl_ca_cert = "path/to/cert-bundle.pem"

        openapi_config = OpenApiConfiguration()

        config = PineconeConfig.build(
            api_key=api_key,
            host=controller_host,
            ssl_ca_certs=ssl_ca_cert,
            openapi_config=openapi_config,
        )

        assert config.api_key == api_key
        assert config.host == "https://" + controller_host
        assert config.ssl_ca_certs == "path/to/cert-bundle.pem"

    def test_resolution_order_kwargs_over_env_vars(self):
        """
        Test that when config is present from multiple sources,
        the order of precedence is kwargs > env vars
        """
        os.environ["PINECONE_API_KEY"] = "env-var-api-key"
        os.environ["PINECONE_CONTROLLER_HOST"] = "env-var-controller-host"
        os.environ["PINECONE_ADDITIONAL_HEADERS"] = '{"header": "value1"}'

        api_key = "kwargs-api-key"
        controller_host = "kwargs-controller-host"
        additional_headers = {"header": "value2"}

        config = PineconeConfig.build(
            api_key=api_key, host=controller_host, additional_headers=additional_headers
        )

        assert config.api_key == api_key
        assert config.host == "https://" + controller_host
        assert config.additional_headers == additional_headers

    def test_errors_when_no_api_key_is_present(self):
        with pytest.raises(PineconeConfigurationError):
            PineconeConfig.build()

    def test_config_pool_threads(self):
        pc = Pinecone(api_key="test-api-key", host="test-controller-host", pool_threads=10)
        assert pc.index_api.api_client.pool_threads == 10
        idx = pc.Index(host="my-index-host", name="my-index-name")
        assert idx._vector_api.api_client.pool_threads == 10

    def test_ssl_config_passed_to_index_client(self):
        proxy_headers = make_headers(proxy_basic_auth="asdf")
        pc = Pinecone(api_key="key", ssl_ca_certs="path/to/cert", proxy_headers=proxy_headers)

        assert pc.openapi_config.ssl_ca_cert == "path/to/cert"
        assert pc.openapi_config.proxy_headers == proxy_headers

        idx = pc.Index(host="host")
        assert idx._vector_api.api_client.configuration.ssl_ca_cert == "path/to/cert"
        assert idx._vector_api.api_client.configuration.proxy_headers == proxy_headers

    def test_host_config_not_clobbered_by_index(self):
        proxy_headers = make_headers(proxy_basic_auth="asdf")
        pc = Pinecone(api_key="key", ssl_ca_certs="path/to/cert", proxy_headers=proxy_headers)

        assert pc.openapi_config.ssl_ca_cert == "path/to/cert"
        assert pc.openapi_config.proxy_headers == proxy_headers
        assert pc.openapi_config.host == "https://api.pinecone.io"

        idx = pc.Index(host="host")
        assert idx._vector_api.api_client.configuration.ssl_ca_cert == "path/to/cert"
        assert idx._vector_api.api_client.configuration.proxy_headers == proxy_headers
        assert idx._vector_api.api_client.configuration.host == "https://host"

        assert pc.openapi_config.host == "https://api.pinecone.io"

    def test_proxy_config(self):
        pc = Pinecone(
            api_key="asdf",
            proxy_url="http://localhost:8080",
            ssl_ca_certs="path/to/cert-bundle.pem",
        )

        assert pc.config.proxy_url == "http://localhost:8080"
        assert pc.config.ssl_ca_certs == "path/to/cert-bundle.pem"

        assert pc.openapi_config.proxy == "http://localhost:8080"
        assert pc.openapi_config.ssl_ca_cert == "path/to/cert-bundle.pem"

        assert pc.index_api.api_client.configuration.proxy == "http://localhost:8080"
        assert pc.index_api.api_client.configuration.ssl_ca_cert == "path/to/cert-bundle.pem"
