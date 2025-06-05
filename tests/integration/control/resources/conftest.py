import os
import pytest
import uuid
import logging
import dotenv
from pinecone import Pinecone, PodIndexEnvironment
from ...helpers import delete_indexes_from_run, delete_backups_from_run, default_create_index_params

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
""" :meta private: """

# Generate a unique ID for the entire test run
RUN_ID = str(uuid.uuid4())


@pytest.fixture()
def pc():
    return Pinecone()


@pytest.fixture()
def create_index_params(request):
    return default_create_index_params(request, RUN_ID)


@pytest.fixture()
def index_name(create_index_params):
    return create_index_params["name"]


@pytest.fixture()
def index_tags(create_index_params):
    return create_index_params["tags"]


@pytest.fixture
def pod_environment():
    return os.getenv("POD_ENVIRONMENT", PodIndexEnvironment.US_EAST1_AWS.value)


@pytest.fixture()
def ready_sl_index(pc, index_name, create_index_params):
    create_index_params["timeout"] = None
    pc.create_index(**create_index_params)
    yield index_name
    pc.db.index.delete(name=index_name, timeout=-1)


@pytest.fixture()
def notready_sl_index(pc, index_name, create_index_params):
    pc.create_index(**create_index_params, timeout=-1)
    yield index_name


def pytest_sessionfinish(session, exitstatus):
    """
    Hook that runs after all tests have completed.
    This is a good place to clean up any resources that were created during the test session.
    """
    logger.info("Running final cleanup after all tests...")

    pc = Pinecone()
    delete_indexes_from_run(pc, RUN_ID)
    delete_backups_from_run(pc, RUN_ID)
