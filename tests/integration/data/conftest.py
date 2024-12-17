import pytest
import os
import json
from ..helpers import get_environment_var, generate_index_name
import logging

logger = logging.getLogger(__name__)


def api_key():
    return get_environment_var("PINECONE_API_KEY")


def use_grpc():
    return os.environ.get("USE_GRPC", "false") == "true"


def build_client():
    config = {"api_key": api_key()}

    if use_grpc():
        from pinecone.grpc import PineconeGRPC

        return PineconeGRPC(**config)
    else:
        from pinecone import Pinecone

        return Pinecone(**config)


@pytest.fixture(scope="session")
def api_key_fixture():
    return api_key()


@pytest.fixture(scope="session")
def client():
    return build_client()


@pytest.fixture(scope="session")
def metric():
    return "cosine"


@pytest.fixture(scope="session")
def spec():
    spec_json = get_environment_var(
        "SPEC", '{"serverless": {"cloud": "aws", "region": "us-east-1" }}'
    )
    return json.loads(spec_json)


@pytest.fixture(scope="session")
def index_name():
    return generate_index_name("dense")


@pytest.fixture(scope="session")
def sparse_index_name():
    return generate_index_name("sparse")


def build_index_client(client, index_name, index_host):
    if use_grpc():
        return client.Index(name=index_name, host=index_host)
    else:
        return client.Index(name=index_name, host=index_host)


@pytest.fixture(scope="session")
def idx(client, index_name, index_host):
    return build_index_client(client, index_name, index_host)


@pytest.fixture(scope="session")
def sparse_idx(client, sparse_index_name, sparse_index_host):
    return build_index_client(client, sparse_index_name, sparse_index_host)


@pytest.fixture(scope="session")
def index_host(index_name, metric, spec):
    pc = build_client()

    if index_name not in pc.list_indexes().names():
        logger.info("Creating index with name: " + index_name)
        pc.create_index(name=index_name, dimension=2, metric=metric, spec=spec)
    else:
        logger.info("Index with name " + index_name + " already exists")

    description = pc.describe_index(name=index_name)
    yield description.host

    logger.info("Deleting index with name: " + index_name)
    pc.delete_index(index_name, -1)


@pytest.fixture(scope="session")
def sparse_index_host(sparse_index_name, spec):
    pc = build_client()

    if sparse_index_name not in pc.list_indexes().names():
        logger.info("Creating sparse index with name: " + sparse_index_name)
        pc.create_index(
            name=sparse_index_name, metric="dotproduct", spec=spec, vector_type="sparse"
        )
    else:
        logger.info("Sparse index with name " + sparse_index_name + " already exists")

    description = pc.describe_index(name=sparse_index_name)
    yield description.host

    logger.info("Deleting sparse index with name: " + sparse_index_name)
    pc.delete_index(sparse_index_name, -1)
