import pytest
import pytest_asyncio
import json
import asyncio
from ..helpers import get_environment_var, generate_index_name
from pinecone.db_data import _IndexAsyncio
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


def build_asyncioindex_client(index_host) -> _IndexAsyncio:
    from pinecone import Pinecone

    return Pinecone().IndexAsyncio(host=index_host)


@pytest_asyncio.fixture(scope="function")
async def idx(index_host):
    print("Building client for async index")
    client = build_asyncioindex_client(index_host)
    yield client
    await client.close()


@pytest_asyncio.fixture(scope="function")
async def sparse_idx(sparse_index_host):
    print("Building client for async sparse index")
    client = build_asyncioindex_client(sparse_index_host)
    yield client
    await client.close()


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


async def get_query_response(asyncio_idx, namespace: str, dimension: Optional[int] = None):
    if dimension is not None:
        return await asyncio_idx.query(top_k=1, vector=[0.0] * dimension, namespace=namespace)
    else:
        from pinecone import SparseValues

        response = await asyncio_idx.query(
            top_k=1, namespace=namespace, sparse_vector=SparseValues(indices=[0], values=[1.0])
        )
        return response


async def poll_until_lsn_reconciled_async(
    asyncio_idx, target_lsn: int, namespace: str, max_wait_time: int = 60 * 3
) -> None:
    """Poll until a target LSN has been reconciled using LSN headers (async).

    This function uses LSN headers from fetch/query operations to determine
    freshness instead of polling describe_index_stats, which is faster.

    Args:
        asyncio_idx: The async index client to use for polling
        target_lsn: The LSN value to wait for (from a write operation)
        namespace: The namespace to wait for
        max_wait_time: Maximum time to wait in seconds

    Raises:
        TimeoutError: If the LSN is not reconciled within max_wait_time seconds
    """
    # Get index dimension for query vector (once, not every iteration)
    dimension = None
    try:
        stats = await asyncio_idx.describe_index_stats()
        dimension = stats.dimension
    except Exception:
        logger.debug("Could not get index dimension")

    delta_t = 2  # Use shorter interval for LSN polling
    total_time = 0
    done = False

    while not done:
        logger.debug(
            f"Polling for LSN reconciliation (async). Target LSN: {target_lsn}, "
            f"namespace: {namespace}, total time: {total_time}s"
        )

        # Try query as a lightweight operation to check LSN
        # Query operations return x-pinecone-max-indexed-lsn header
        try:
            from tests.integration.helpers.lsn_utils import is_lsn_reconciled

            # Use a minimal query to get headers (this is more efficient than describe_index_stats)
            response = await get_query_response(asyncio_idx, namespace, dimension)
            reconciled_lsn = response._response_info.get("lsn_reconciled")

            logger.debug(f"Current reconciled LSN: {reconciled_lsn}, target: {target_lsn}")
            if is_lsn_reconciled(target_lsn, reconciled_lsn):
                # LSN is reconciled, check if additional condition is met
                done = True
                logger.debug(f"LSN {target_lsn} is reconciled after {total_time}s")
            else:
                logger.debug(
                    f"LSN not yet reconciled. Reconciled: {reconciled_lsn}, target: {target_lsn}"
                )
        except Exception as e:
            logger.debug(f"Error checking LSN: {e}")

        if not done:
            if total_time >= max_wait_time:
                raise TimeoutError(
                    f"Timeout waiting for LSN {target_lsn} to be reconciled after {total_time}s"
                )
            total_time += delta_t
            await asyncio.sleep(delta_t)


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
