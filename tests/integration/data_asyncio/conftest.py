import pytest
import os
from ..helpers import get_environment_var, random_string


@pytest.fixture(scope="session")
def api_key():
    return get_environment_var("PINECONE_API_KEY")


@pytest.fixture(scope="session")
def host():
    return get_environment_var("INDEX_HOST")


@pytest.fixture(scope="session")
def dimension():
    return int(get_environment_var("DIMENSION"))


def use_grpc():
    return os.environ.get("USE_GRPC", "false") == "true"


def build_client(api_key):
    if use_grpc():
        from pinecone.grpc import PineconeGRPC

        return PineconeGRPC(api_key=api_key)
    else:
        from pinecone import Pinecone

        return Pinecone(
            api_key=api_key, additional_headers={"sdk-test-suite": "pinecone-python-client"}
        )


@pytest.fixture(scope="session")
async def pc(api_key):
    return build_client(api_key=api_key)


@pytest.fixture(scope="session")
async def asyncio_idx(pc, host):
    return pc.AsyncioIndex(host=host)


@pytest.fixture(scope="session")
async def namespace():
    return random_string(10)


@pytest.fixture(scope="session")
async def list_namespace():
    return random_string(10)
