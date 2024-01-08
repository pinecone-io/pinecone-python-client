from pinecone import Pinecone
from pinecone.exceptions import PineconeConfigurationError
from pinecone.config import PineconeConfig
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

import pytest
import os

class TestConfig:
    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(tmpdir):
        """Fixture to execute asserts before and after a test is run"""

        # Defend against unexpected env vars. Since we clear these variables below
        # after each test execution, these should only be raised if there is
        # test pollution in the environment coming from some other test file/setup.
        known_env_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_CONTROLLER_HOST"]
        for var in known_env_vars:
            if os.getenv(var):
                raise ValueError(f"Unexpected env var {var} found in environment. Check for test pollution.")

        yield  # this is where the testing happens

        # Teardown : Unset any env vars created during test execution
        for var in known_env_vars:
            if os.getenv(var):
                del os.environ[var]

    def test_init_with_environment_vars(self):
        os.environ["PINECONE_API_KEY"] = "test-api-key"
        os.environ["PINECONE_CONTROLLER_HOST"] = "https://test-controller-host"

        config = PineconeConfig.build()

        assert config.api_key == "test-api-key"
        assert config.host == "https://test-controller-host"

    def test_init_with_positional_args(self):
        api_key = "my-api-key"
        host = "https://my-controller-host"

        config = PineconeConfig.build(api_key, host)

        assert config.api_key == api_key
        assert config.host == host

    def test_init_with_kwargs(self):
        api_key = "my-api-key"
        controller_host = "my-controller-host"
        openapi_config = OpenApiConfiguration(api_key="openapi-api-key")

        config = PineconeConfig.build(api_key=api_key, host=controller_host, openapi_config=openapi_config)

        assert config.api_key == api_key
        assert config.host == 'https://' + controller_host
        assert config.openapi_config == openapi_config

    def test_resolution_order_kwargs_over_env_vars(self):
        """
        Test that when config is present from multiple sources,
        the order of precedence is kwargs > env vars
        """
        os.environ["PINECONE_API_KEY"] = "env-var-api-key"
        os.environ["PINECONE_CONTROLLER_HOST"] = "env-var-controller-host"

        api_key = "kwargs-api-key"
        controller_host = "kwargs-controller-host"

        config = PineconeConfig.build(api_key=api_key, host=controller_host)

        assert config.api_key == api_key
        assert config.host == 'https://' + controller_host

    def test_errors_when_no_api_key_is_present(self):
        with pytest.raises(PineconeConfigurationError):
            PineconeConfig.build()
    
    def test_config_pool_threads(self):
        pc = Pinecone(api_key="test-api-key", host="test-controller-host", pool_threads=10)
        assert pc.index_api.api_client.pool_threads == 10
        idx = pc.Index(host='my-index-host', name='my-index-name')
        assert idx._api_client.pool_threads == 10
        
