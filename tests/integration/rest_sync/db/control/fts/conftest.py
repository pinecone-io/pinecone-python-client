"""Fixtures for FTS control plane integration tests."""

import pytest
import uuid
import logging
import dotenv
from pinecone import Pinecone
from tests.integration.helpers import delete_indexes_from_run, index_tags

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Generate a unique ID for the entire test run
RUN_ID = str(uuid.uuid4())


@pytest.fixture()
def pc():
    """Create a Pinecone client."""
    return Pinecone()


@pytest.fixture()
def index_name_and_tags(request):
    """Generate a unique index name and tags for each test."""
    name = f"{str(uuid.uuid4())}"
    tags = index_tags(request, RUN_ID)
    return name, tags


def pytest_sessionfinish(session, exitstatus):
    """Clean up indexes created during the test session."""
    logger.info("Running final cleanup after FTS control plane tests...")
    pc = Pinecone()
    delete_indexes_from_run(pc, RUN_ID)
