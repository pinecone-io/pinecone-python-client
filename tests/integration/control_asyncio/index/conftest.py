import pytest
import time
import random
import asyncio
from ...helpers import get_environment_var, generate_index_name
import logging
from typing import Callable, Optional, Awaitable, Union

from pinecone import (
    CloudProvider,
    AwsRegion,
    ServerlessSpec,
    PineconeApiException,
    NotFoundException,
)

logger = logging.getLogger(__name__)


def build_client():
    from pinecone import PineconeAsyncio

    return PineconeAsyncio()


@pytest.fixture(scope="session")
def client():
    # This returns the sync client. Not for use in tests
    # but can be used to help with cleanup after test runs
    from pinecone import Pinecone

    return Pinecone()


@pytest.fixture(scope="session")
def build_pc():
    return build_client


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
    timeout: Optional[float] = 10.0,
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


@pytest.fixture()
def serverless_cloud():
    return get_environment_var("SERVERLESS_CLOUD", "aws")


@pytest.fixture()
def serverless_region():
    return get_environment_var("SERVERLESS_REGION", "us-west-2")


@pytest.fixture()
def spec1(serverless_cloud, serverless_region):
    return {"serverless": {"cloud": serverless_cloud, "region": serverless_region}}


@pytest.fixture()
def spec2():
    return ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1)


@pytest.fixture()
def spec3():
    return {"serverless": {"cloud": CloudProvider.AWS, "region": AwsRegion.US_EAST_1}}


@pytest.fixture()
def create_sl_index_params(index_name, serverless_cloud, serverless_region):
    spec = {"serverless": {"cloud": serverless_cloud, "region": serverless_region}}
    return dict(name=index_name, dimension=10, metric="cosine", spec=spec)


@pytest.fixture()
def random_vector():
    return [random.uniform(0, 1) for _ in range(10)]


@pytest.fixture()
def index_name(request):
    test_name = request.node.name
    return generate_index_name(test_name)


@pytest.fixture()
def ready_sl_index(client, index_name, create_sl_index_params):
    create_sl_index_params["timeout"] = None
    client.create_index(**create_sl_index_params)
    yield index_name
    client.delete_index(index_name, -1)


@pytest.fixture()
def notready_sl_index(client, index_name, create_sl_index_params):
    client.create_index(**create_sl_index_params, timeout=-1)
    yield index_name


def delete_with_retry(client, index_name, retries=0, sleep_interval=5):
    print(
        "Deleting index "
        + index_name
        + ", retry "
        + str(retries)
        + ", next sleep interval "
        + str(sleep_interval)
    )
    try:
        client.delete_index(index_name, -1)
    except NotFoundException:
        pass
    except PineconeApiException as e:
        if e.error.code == "PRECONDITON_FAILED":
            if retries > 5:
                raise "Unable to delete index " + index_name
            time.sleep(sleep_interval)
            delete_with_retry(client, index_name, retries + 1, sleep_interval * 2)
        else:
            print(e.__class__)
            print(e)
            raise "Unable to delete index " + index_name
    except Exception as e:
        print(e.__class__)
        print(e)
        raise "Unable to delete index " + index_name


@pytest.fixture(autouse=True)
async def cleanup(client, index_name):
    yield

    try:
        logger.debug("Attempting to delete index with name: " + index_name)
        client.index.delete(name=index_name, timeout=-1)
    except Exception:
        pass
