import os
import pytest
from pinecone import PodIndexEnvironment


@pytest.fixture
def pod_environment():
    return os.getenv("POD_ENVIRONMENT", PodIndexEnvironment.US_EAST1_AWS.value)
