import pinecone
from pinecone.exceptions import ApiKeyError
from pinecone.config.config import Config
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

    def test_init_with_environment_vars():
        os.environ["PINECONE_API_KEY"] = "test-api-key"
        os.environ["PINECONE_CONTROLLER_HOST"] = "test-controller-host"

        config = Config()

        assert config.API_KEY == "test-api-key"
        assert config.CONTROLLER_HOST == "test-controller-host"

    def test_init_with_positional_args():
        api_key = "my-api-key"
        host = "my-controller-host"
        openapi_config = OpenApiConfiguration(api_key="openapi-api-key")

        config = Config(api_key, host, openapi_config)

        assert config.API_KEY == api_key
        assert config.CONTROLLER_HOST == host
        assert config.OPENAPI_CONFIG == openapi_config

    def test_init_with_kwargs():
        api_key = "my-api-key"
        controller_host = "my-controller-host"
        openapi_config = OpenApiConfiguration(api_key="openapi-api-key")

        config = Config(api_key=api_key, host=controller_host, openapi_config=openapi_config)

        assert config.API_KEY == api_key
        assert config.CONTROLLER_HOST == controller_host
        assert config.OPENAPI_CONFIG == openapi_config

    def test_init_with_mispelled_kwargs(caplog):
        Config(api_key='my-api-key', unknown_kwarg='bogus')
        assert "__init__ had unexpected keyword argument(s): unknown_kwarg" in caplog.text

    def test_resolution_order_kwargs_over_env_vars():
        """
        Test that when config is present from multiple sources,
        the order of precedence is kwargs > env vars
        """
        os.environ["PINECONE_API_KEY"] = "env-var-api-key"
        os.environ["PINECONE_CONTROLLER_HOST"] = "env-var-controller-host"

        api_key = "kwargs-api-key"
        controller_host = "kwargs-controller-host"

        config = Config(api_key=api_key, host=controller_host)

        assert config.API_KEY == api_key
        assert config.CONTROLLER_HOST == controller_host

    def test_errors_when_no_api_key_is_present():
        with pytest.raises(ApiKeyError):
            config = Config()