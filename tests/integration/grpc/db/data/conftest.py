import pytest
import json
import os
import uuid
from tests.integration.helpers import (
    get_environment_var,
    index_tags as index_tags_helper,
    generate_name,
)
import logging
from pinecone import EmbedModel, CloudProvider, AwsRegion, IndexEmbed
from pinecone.grpc import PineconeGRPC

logger = logging.getLogger(__name__)

RUN_ID = str(uuid.uuid4())

created_indexes: list[str] = []


@pytest.fixture(scope="session")
def index_tags(request):
    return index_tags_helper(request, RUN_ID)


@pytest.fixture(scope="session")
def pc():
    return PineconeGRPC()


@pytest.fixture(scope="session")
def spec():
    spec_json = get_environment_var(
        "SPEC", '{"serverless": {"cloud": "aws", "region": "us-east-1" }}'
    )
    return json.loads(spec_json)


@pytest.fixture(scope="session")
def model_idx(pc, index_tags, request):
    env_host = os.getenv("INDEX_HOST_EMBEDDED_MODEL")
    if env_host:
        logger.info(f"Using pre-created index host from INDEX_HOST_EMBEDDED_MODEL: {env_host}")
        return pc.Index(host=env_host)

    model_index_name = generate_name(request.node.name, "embed")
    if not pc.has_index(name=model_index_name):
        logger.warning(
            f"INDEX_HOST_EMBEDDED_MODEL not set. Creating new index {model_index_name}. "
            "Consider using pre-created indexes via environment variables for CI parallelization."
        )
        pc.create_index_for_model(
            name=model_index_name,
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_WEST_2,
            embed=IndexEmbed(
                model=EmbedModel.Multilingual_E5_Large,
                field_map={"text": "my_text_field"},
                metric="cosine",
            ),
            tags=index_tags,
        )
        created_indexes.append(model_index_name)
    else:
        logger.info(f"Index {model_index_name} already exists")

    description = pc.describe_index(name=model_index_name)
    return pc.Index(host=description.host)


def create_index(pc, create_args):
    if not pc.has_index(name=create_args["name"]):
        logger.info(f"Creating index {create_args['name']}")
        pc.create_index(**create_args)
    else:
        logger.info(f"Index {create_args['name']} already exists")

    host = pc.describe_index(name=create_args["name"]).host

    return host


@pytest.fixture(scope="session")
def idx(pc, spec, index_tags, request):
    env_host = os.getenv("INDEX_HOST_DENSE")
    if env_host:
        logger.info(f"Using pre-created index host from INDEX_HOST_DENSE: {env_host}")
        return pc.Index(host=env_host)

    index_name = generate_name(request.node.name, "dense")
    logger.info(f"Request: {request.node}")
    create_args = {
        "name": index_name,
        "dimension": 2,
        "metric": "cosine",
        "spec": spec,
        "tags": index_tags,
    }
    if not pc.has_index(name=create_args["name"]):
        logger.warning(
            f"INDEX_HOST_DENSE not set. Creating new index {index_name}. "
            "Consider using pre-created indexes via environment variables for CI parallelization."
        )
    host = create_index(pc, create_args)
    logger.info(f"Using index {index_name} with host {host} as idx")
    created_indexes.append(index_name)
    return pc.Index(host=host)


@pytest.fixture(scope="session")
def sparse_idx(pc, spec, index_tags, request):
    env_host = os.getenv("INDEX_HOST_SPARSE")
    if env_host:
        logger.info(f"Using pre-created index host from INDEX_HOST_SPARSE: {env_host}")
        return pc.Index(host=env_host)

    index_name = generate_name(request.node.name, "sparse")
    create_args = {
        "name": index_name,
        "metric": "dotproduct",
        "spec": spec,
        "vector_type": "sparse",
        "tags": index_tags,
    }
    if not pc.has_index(name=create_args["name"]):
        logger.warning(
            f"INDEX_HOST_SPARSE not set. Creating new index {index_name}. "
            "Consider using pre-created indexes via environment variables for CI parallelization."
        )
    host = create_index(pc, create_args)
    created_indexes.append(index_name)
    return pc.Index(host=host)


def pytest_sessionfinish(session, exitstatus):
    for index in created_indexes:
        try:
            logger.info(f"Deleting index {index}")
            pc = PineconeGRPC()
            pc.delete_index(name=index, timeout=-1)
        except Exception as e:
            logger.error(f"Error deleting index {index}: {e}")
