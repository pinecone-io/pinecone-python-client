import pytest
import json
import asyncio
from ..helpers import get_environment_var, generate_index_name
from pinecone.data import _AsyncioIndex
import logging
from typing import Callable, Optional, Awaitable, Union

from pinecone import CloudProvider, AwsRegion, IndexEmbed, EmbedModel

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def metric():
    return "cosine"


@pytest.fixture(scope="session")
def dimension():
    return 2


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


@pytest.fixture(scope="session")
def model_index_name():
    return generate_index_name("embed")


def build_asyncioindex_client(index_host) -> _AsyncioIndex:
    from pinecone import PineconeAsyncio

    return PineconeAsyncio().Index(host=index_host)


@pytest.fixture(scope="session")
def idx(client, index_name, index_host):
    print("Building client for {}".format(index_name))
    return build_asyncioindex_client(index_host)


@pytest.fixture(scope="session")
def sparse_idx(client, sparse_index_name, sparse_index_host):
    print("Building client for {}".format(sparse_index_name))
    return build_asyncioindex_client(sparse_index_host)


@pytest.fixture(scope="session")
def index_host(index_name, metric, spec, dimension):
    from pinecone import Pinecone

    pc = Pinecone()

    if index_name not in pc.list_indexes().names():
        logger.info("Creating index with name: " + index_name)
        pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
    else:
        logger.info("Index with name " + index_name + " already exists")

    description = pc.describe_index(name=index_name)
    yield description.host

    logger.info("Deleting index with name: " + index_name)
    pc.delete_index(index_name, -1)


@pytest.fixture(scope="session")
def sparse_index_host(sparse_index_name, spec):
    from pinecone import Pinecone

    pc = Pinecone()

    if sparse_index_name not in pc.list_indexes().names():
        logger.info(f"Creating index with name {sparse_index_name}")
        pc.create_index(
            name=sparse_index_name, metric="dotproduct", spec=spec, vector_type="sparse"
        )
    else:
        logger.info(f"Index with name {sparse_index_name} already exists")

    description = pc.describe_index(name=sparse_index_name)
    yield description.host

    logger.info(f"Deleting index with name {sparse_index_name}")
    pc.delete_index(sparse_index_name, -1)


@pytest.fixture(scope="session")
def model_index_host(model_index_name):
    from pinecone import Pinecone

    pc = Pinecone()

    if model_index_name not in pc.list_indexes().names():
        logger.info(f"Creating index {model_index_name}")
        pc.create_index_for_model(
            name=model_index_name,
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_WEST_2,
            embed=IndexEmbed(
                model=EmbedModel.Multilingual_E5_Large,
                field_map={"text": "my_text_field"},
                metric="cosine",
            ),
        )
    else:
        logger.info(f"Index {model_index_name} already exists")

    description = pc.describe_index(name=model_index_name)
    yield description.host

    logger.info(f"Deleting index {model_index_name}")
    pc.delete_index(model_index_name, -1)


async def poll_for_freshness(asyncio_idx, target_namespace, target_vector_count):
    max_wait_time = 60 * 3  # 3 minutes
    time_waited = 0
    wait_per_iteration = 5

    while True:
        stats = await asyncio_idx.describe_index_stats()
        logger.debug(
            "Polling for freshness on index %s. Current vector count: %s. Waiting for: %s",
            asyncio_idx,
            stats.total_vector_count,
            target_vector_count,
        )
        if target_namespace == "":
            if stats.total_vector_count >= target_vector_count:
                break
        else:
            if (
                target_namespace in stats.namespaces
                and stats.namespaces[target_namespace].vector_count >= target_vector_count
            ):
                break
        time_waited += wait_per_iteration
        if time_waited >= max_wait_time:
            raise TimeoutError(
                "Timeout waiting for index to have expected vector count of {}".format(
                    target_vector_count
                )
            )
        await asyncio.sleep(wait_per_iteration)

    return stats


async def wait_until(
    condition: Union[Callable[[], bool], Callable[[], Awaitable[bool]]],
    timeout: Optional[float] = 10,
    interval: float = 0.1,
) -> None:
    """
    Waits asynchronously until the given (async or sync) condition returns True or times out.

    Args:
        condition: A callable that returns a boolean or an awaitable boolean, indicating if the wait is over.
        timeout: Maximum time in seconds to wait for the condition to become True. If None, wait indefinitely.
        interval: Time in seconds between checks of the condition.

    Raises:
        asyncio.TimeoutError: If the condition is not met within the timeout period.
    """
    start_time = asyncio.get_event_loop().time()

    while True:
        result = await condition() if asyncio.iscoroutinefunction(condition) else condition()
        if result:
            return

        if timeout is not None and (asyncio.get_event_loop().time() - start_time) > timeout:
            raise asyncio.TimeoutError("Condition not met within the timeout period.")

        remaining_time = (
            (start_time + timeout) - asyncio.get_event_loop().time()
            if timeout is not None
            else None
        )
        logger.debug(
            "Condition not met yet. Waiting for %.2f seconds. Timeout in %.2f seconds.",
            interval,
            remaining_time,
        )
        await asyncio.sleep(interval)
