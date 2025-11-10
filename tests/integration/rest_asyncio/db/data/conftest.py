import pytest
import pytest_asyncio
import json
import os
import asyncio
from tests.integration.helpers import get_environment_var, generate_index_name
from pinecone.db_data import _IndexAsyncio
import logging
from typing import Optional, Dict, Any

from pinecone import CloudProvider, AwsRegion, IndexEmbed, EmbedModel

logger = logging.getLogger(__name__)


def build_sync_client():
    from pinecone import Pinecone

    return Pinecone()


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
    if os.getenv("INDEX_HOST_DENSE"):
        host = os.getenv("INDEX_HOST_DENSE")
        logger.info(
            f"Looking up index name from pre-created index host from INDEX_HOST_DENSE: {host}"
        )
        pc = build_sync_client()
        index_name = pc.describe_index(host=host).name
        logger.info(
            f"Found index name: {index_name} for pre-created index host from INDEX_HOST_DENSE: {host}"
        )
        return index_name
    else:
        return generate_index_name("dense")


@pytest.fixture(scope="session")
def sparse_index_name():
    if os.getenv("INDEX_HOST_SPARSE"):
        host = os.getenv("INDEX_HOST_SPARSE")
        logger.info(
            f"Looking up index name from pre-created index host from INDEX_HOST_SPARSE: {host}"
        )
        pc = build_sync_client()
        index_name = pc.describe_index(host=host).name
        logger.info(
            f"Found index name: {index_name} for pre-created index host from INDEX_HOST_SPARSE: {host}"
        )
        return index_name
    else:
        return generate_index_name("sparse")


@pytest.fixture(scope="session")
def model_index_name():
    if os.getenv("INDEX_HOST_EMBEDDED_MODEL"):
        host = os.getenv("INDEX_HOST_EMBEDDED_MODEL")
        logger.info(
            f"Looking up index name from pre-created index host from INDEX_HOST_EMBEDDED_MODEL: {host}"
        )
        pc = build_sync_client()
        index_name = pc.describe_index(host=host).name
        logger.info(
            f"Found index name: {index_name} for pre-created index host from INDEX_HOST_EMBEDDED_MODEL: {host}"
        )
        return index_name
    else:
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
    env_host = os.getenv("INDEX_HOST_DENSE")
    if env_host:
        logger.info(f"Using pre-created index host from INDEX_HOST_DENSE: {env_host}")
        yield env_host
        return

    from pinecone import Pinecone

    pc = Pinecone()

    if index_name not in pc.list_indexes().names():
        logger.warning(
            f"INDEX_HOST_DENSE not set. Creating new index {index_name}. "
            "Consider using pre-created indexes via environment variables for CI parallelization."
        )
        pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
    else:
        logger.info("Index with name " + index_name + " already exists")

    description = pc.describe_index(name=index_name)
    yield description.host

    logger.info("Deleting index with name: " + index_name)
    pc.delete_index(index_name, -1)


@pytest.fixture(scope="session")
def sparse_index_host(sparse_index_name, spec):
    env_host = os.getenv("INDEX_HOST_SPARSE")
    if env_host:
        logger.info(f"Using pre-created index host from INDEX_HOST_SPARSE: {env_host}")
        yield env_host
        return

    from pinecone import Pinecone

    pc = Pinecone()

    if sparse_index_name not in pc.list_indexes().names():
        logger.warning(
            f"INDEX_HOST_SPARSE not set. Creating new index {sparse_index_name}. "
            "Consider using pre-created indexes via environment variables for CI parallelization."
        )
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
    env_host = os.getenv("INDEX_HOST_EMBEDDED_MODEL")
    if env_host:
        logger.info(f"Using pre-created index host from INDEX_HOST_EMBEDDED_MODEL: {env_host}")
        yield env_host
        return

    from pinecone import Pinecone

    pc = Pinecone()

    if model_index_name not in pc.list_indexes().names():
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
    asyncio_idx, response_info: Dict[str, Any], namespace: str, max_wait_time: int = 60 * 3
) -> None:
    """Poll until a target LSN has been reconciled using LSN headers (async).

    This function uses LSN headers from fetch/query operations to determine
    freshness instead of polling describe_index_stats, which is faster.

    Args:
        asyncio_idx: The async index client to use for polling
        response_info: ResponseInfo dictionary from a write operation (upsert, delete)
                       containing raw_headers with the committed LSN
        namespace: The namespace to wait for
        max_wait_time: Maximum time to wait in seconds

    Raises:
        TimeoutError: If the LSN is not reconciled within max_wait_time seconds
        ValueError: If target_lsn cannot be extracted from response_info (LSN should always be available)
    """
    from tests.integration.helpers.lsn_utils import (
        extract_lsn_committed,
        extract_lsn_reconciled,
        is_lsn_reconciled,
    )

    # Extract target_lsn from response_info.raw_headers
    raw_headers = response_info.get("raw_headers", {})
    target_lsn = extract_lsn_committed(raw_headers)
    if target_lsn is None:
        raise ValueError("No target LSN found in response_info.raw_headers")

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
            # Use a minimal query to get headers (this is more efficient than describe_index_stats)
            response = await get_query_response(asyncio_idx, namespace, dimension)
            # Extract reconciled_lsn from query response's raw_headers
            query_raw_headers = response._response_info.get("raw_headers", {})
            reconciled_lsn = extract_lsn_reconciled(query_raw_headers)

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
