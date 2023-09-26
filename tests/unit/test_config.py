import pinecone
from pinecone.config import Config
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

import pytest
import tempfile
import os

@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    
    # Defend against unexpected env vars. Since we clear these variables below
    # after each test execution, these should only be raised if there is 
    # test pollution in the environment coming from some other test file/setup.
    known_env_vars = ['PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'PINECONE_PROJECT_NAME', 'PINECONE_CONTROLLER_HOST']
    for var in known_env_vars:
        if os.getenv(var):
            raise ValueError(f'Unexpected env var {var} found in environment. Check for test pollution.')

    # Unfortunately since config is a singleton, we need to reset it manually between tests
    pinecone.init()

    yield # this is where the testing happens

    # Teardown : Unset any env vars created during test execution
    for var in known_env_vars:
        if os.getenv(var):
            del os.environ[var]

def test_default_config():
    """
    Test that default config is loaded when no config is specified. This not really a valid config that can be used,
    but adding this test just to document the legacy behavior.
    """
    pinecone.init()
    assert Config.API_KEY == ''
    assert Config.ENVIRONMENT == 'us-west1-gcp'
    assert Config.PROJECT_NAME == 'UNKNOWN'
    assert Config.CONTROLLER_HOST == 'https://controller.us-west1-gcp.pinecone.io'
    assert Config.LOG_LEVEL == 'ERROR'

def test_init_with_environment_vars():
    os.environ['PINECONE_ENVIRONMENT'] = 'test-env'
    os.environ['PINECONE_API_KEY'] = 'test-api-key'
    os.environ['PINECONE_PROJECT_NAME'] = 'test-project-name'
    os.environ['PINECONE_CONTROLLER_HOST'] = 'test-controller-host'

    pinecone.init()

    assert Config.API_KEY == 'test-api-key'
    assert Config.ENVIRONMENT == 'test-env'
    assert Config.PROJECT_NAME == 'test-project-name'
    assert Config.CONTROLLER_HOST == 'test-controller-host'

def test_init_with_positional_args():
    api_key = 'my-api-key'
    environment = 'test-env'
    host = 'my-controller-host'
    project_name = 'my-project-name'
    log_level = None # deprecated property but still in positional list
    openapi_config = OpenApiConfiguration(api_key='openapi-api-key')

    pinecone.init(api_key, host, environment, project_name, log_level, openapi_config)

    assert Config.API_KEY == api_key
    assert Config.ENVIRONMENT == environment
    assert Config.PROJECT_NAME == project_name
    assert Config.CONTROLLER_HOST == host
    assert Config.OPENAPI_CONFIG == openapi_config

def test_init_with_kwargs():
    env = 'test-env'
    api_key = 'my-api-key'
    project_name = 'my-project-name'
    controller_host = 'my-controller-host'
    openapi_config = OpenApiConfiguration(api_key='openapi-api-key')

    pinecone.init(api_key=api_key, environment=env, project_name=project_name, host=controller_host, openapi_config=openapi_config)

    assert Config.API_KEY == api_key
    assert Config.ENVIRONMENT == env
    assert Config.PROJECT_NAME == project_name
    assert Config.CONTROLLER_HOST == controller_host
    assert Config.OPENAPI_CONFIG == openapi_config

def test_init_with_mispelled_kwargs(caplog):
    pinecone.init(invalid_kwarg="value")
    assert 'init had unexpected keyword argument(s): invalid_kwarg' in caplog.text

def test_init_with_file_based_configuration():
    """Test that config can be loaded from a file"""
    env = 'ini-test-env'
    api_key = 'ini-api-key'
    project_name = 'ini-project-name'
    controller_host = 'ini-controller-host'

    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(f"""
        [default]
        environment: {env}
        api_key: {api_key}
        project_name: {project_name}
        controller_host: {controller_host}
        """)
        f.flush()

        pinecone.init(config=f.name)

    assert Config.API_KEY == api_key
    assert Config.ENVIRONMENT == env
    assert Config.PROJECT_NAME == project_name
    assert Config.CONTROLLER_HOST == controller_host

def test_resolution_order_kwargs_over_env_vars():
    """
    Test that when config is present from multiple sources, 
    the order of precedence is kwargs > env vars
    """
    os.environ['PINECONE_ENVIRONMENT'] = 'env-var-env'
    os.environ['PINECONE_API_KEY'] = 'env-var-api-key'
    os.environ['PINECONE_PROJECT_NAME'] = 'env-var-project-name'
    os.environ['PINECONE_CONTROLLER_HOST'] = 'env-var-controller-host'

    env = 'kwargs-env'
    api_key = 'kwargs-api-key'
    project_name = 'kwargs-project-name'
    controller_host = 'kwargs-controller-host'

    pinecone.init(environment=env, api_key=api_key, project_name=project_name, host=controller_host)

    assert Config.API_KEY == api_key
    assert Config.ENVIRONMENT == env
    assert Config.PROJECT_NAME == project_name
    assert Config.CONTROLLER_HOST == controller_host

def test_resolution_order_kwargs_over_config_file():
    """
    Test that when config is present from multiple sources, the order of 
    precedence is kwargs > config file
    """
    env = 'ini-test-env'
    api_key = 'ini-api-key'
    project_name = 'ini-project-name'
    controller_host = 'ini-controller-host'

    kwargs_api_key = 'kwargs-api-key'
    kwargs_project_name = 'kwargs-project-name'

    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(f"""
        [default]
        environment: {env}
        api_key: {api_key}
        project_name: {project_name}
        controller_host: {controller_host}
        """)
        f.flush()

        pinecone.init(api_key=kwargs_api_key, project_name=kwargs_project_name, config=f.name)

    # Properties passed as kwargs take precedence over config file
    assert Config.API_KEY == kwargs_api_key
    assert Config.PROJECT_NAME == kwargs_project_name

    # Properties not passed as kwargs loaded from config file
    assert Config.ENVIRONMENT == env
    assert Config.CONTROLLER_HOST == controller_host

def test_resolution_order_env_vars_over_config_file():
    """
    Test that when config is present from multiple sources, the order of precedence is 
    env vars > config file
    """
    
    os.environ['PINECONE_ENVIRONMENT'] = 'env-var-env'
    os.environ['PINECONE_API_KEY'] = 'env-var-api-key'
    os.environ['PINECONE_PROJECT_NAME'] = 'env-var-project-name'
    os.environ['PINECONE_CONTROLLER_HOST'] = 'env-var-controller-host'
    
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(f"""
        [default]
        environment: ini-test-env
        api_key: ini-api-key
        project_name: ini-project-name
        controller_host: ini-controller-host
        """)
        f.flush()

        pinecone.init(config=f.name)

    assert Config.API_KEY == 'env-var-api-key'
    assert Config.ENVIRONMENT == 'env-var-env'
    assert Config.PROJECT_NAME == 'env-var-project-name'
    assert Config.CONTROLLER_HOST == 'env-var-controller-host'
    

def test_init_from_mixed_sources():
    """
    Test that even when some vars are found in a higher precedence source, the rest 
    are still loaded from lower precedence sources
    """

    os.environ['PINECONE_ENVIRONMENT'] = 'env-var-env'
    os.environ['PINECONE_API_KEY'] = 'env-var-api-key'
    project_name = 'kwargs-project-name'
    controller_host = 'kwargs-controller-host'

    pinecone.init(project_name=project_name, host=controller_host)

    assert Config.API_KEY == 'env-var-api-key'
    assert Config.ENVIRONMENT == 'env-var-env'
    assert Config.PROJECT_NAME == project_name
    assert Config.CONTROLLER_HOST == controller_host