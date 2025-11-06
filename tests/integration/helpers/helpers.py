import re
import os
import time
import random
import string
import logging
import uuid
import asyncio
from typing import Any
from datetime import datetime
import json
from pinecone.db_data import _Index
from pinecone import Pinecone, NotFoundException, PineconeApiException
from tests.integration.helpers.lsn_utils import is_lsn_reconciled
from typing import List, Callable, Awaitable, Optional, Union

logger = logging.getLogger(__name__)


def embedding_values(dimension: int = 2) -> list[float]:
    return [random.random() for _ in range(dimension)]


def random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


def generate_collection_name(label: str) -> str:
    return generate_index_name(label)


def generate_index_name(label: str) -> str:
    github_actor = os.getenv("GITHUB_ACTOR", None)
    user = os.getenv("USER", None)
    index_owner = github_actor or user

    formatted_date = datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-3]

    github_job = os.getenv("GITHUB_JOB", None)

    if label.startswith("test_"):
        label = label[5:]

    # Remove trailing underscore, if any
    if label.endswith("_"):
        label = label[:-1]

    name_parts = [index_owner, formatted_date, github_job, label]
    index_name = "-".join([x for x in name_parts if x is not None])

    # Remove invalid characters
    replace_with_hyphen = re.compile(r"[\[\(_,\s]")
    index_name = re.sub(replace_with_hyphen, "-", index_name)
    replace_with_empty = re.compile(r"[\]\)\.]")
    index_name = re.sub(replace_with_empty, "", index_name)

    max_length = 45
    index_name = index_name[:max_length]

    # Trim final character if it is not alphanumeric
    if index_name.endswith("_") or index_name.endswith("-"):
        index_name = index_name[:-1]

    return index_name.lower()


def get_environment_var(name: str, defaultVal: Any = None) -> str:
    val = os.getenv(name, defaultVal)
    if val is None:
        raise Exception("Expected environment variable " + name + " is not set")
    else:
        return val


def poll_stats_for_namespace(
    idx: _Index,
    namespace: str,
    expected_count: int,
    max_sleep: int = int(os.environ.get("FRESHNESS_TIMEOUT_SECONDS", 180)),
) -> None:
    delta_t = 5
    total_time = 0
    done = False
    while not done:
        logger.debug(
            f'Waiting for namespace "{namespace}" to have vectors. Total time waited: {total_time} seconds'
        )
        stats = idx.describe_index_stats()
        # The default namespace may be represented as "" or "__default__" in the API response
        namespace_key = (
            "__default__" if namespace == "" and "__default__" in stats.namespaces else namespace
        )
        if (
            namespace_key in stats.namespaces
            and stats.namespaces[namespace_key].vector_count >= expected_count
        ):
            done = True
        elif total_time > max_sleep:
            raise TimeoutError(f"Timed out waiting for namespace {namespace} to have vectors")
        else:
            total_time += delta_t
            logger.debug(f"Found index stats: {stats}.")
            logger.debug(
                f"Waiting for {expected_count} vectors in namespace {namespace}. Found {stats.namespaces.get(namespace_key, {'vector_count': 0})['vector_count']} vectors."
            )
            time.sleep(delta_t)


def poll_fetch_for_ids_in_namespace(idx: _Index, ids: List[str], namespace: str) -> None:
    max_sleep = int(os.environ.get("FRESHNESS_TIMEOUT_SECONDS", 60))
    delta_t = 5
    total_time = 0
    done = False
    while not done:
        logger.debug(
            f'Attempting to fetch from "{namespace}". Total time waited: {total_time} seconds'
        )
        results = idx.fetch(ids=ids, namespace=namespace)
        logger.debug(results)

        all_present = all(key in results.vectors for key in ids)
        if all_present:
            done = True

        if total_time > max_sleep:
            raise TimeoutError(f"Timed out waiting for namespace {namespace} to have vectors")
        else:
            total_time += delta_t
            time.sleep(delta_t)


def poll_until_lsn_reconciled(
    idx: _Index,
    target_lsn: Optional[int],
    operation_name: str = "read",
    max_sleep: int = int(os.environ.get("FRESHNESS_TIMEOUT_SECONDS", 180)),
    check_fn: Optional[Callable[[], bool]] = None,
) -> None:
    """Poll until a target LSN has been reconciled using LSN headers.

    This function uses LSN headers from query operations to determine
    freshness instead of polling describe_index_stats, which is faster.

    Args:
        idx: The index client to use for polling
        target_lsn: The LSN value to wait for (from a write operation)
        operation_name: Name of the operation being checked (for logging)
        max_sleep: Maximum time to wait in seconds
        check_fn: Optional additional check function that must return True

    Raises:
        TimeoutError: If the LSN is not reconciled within max_sleep seconds
    """
    if target_lsn is None:
        logger.debug("No target LSN provided, cannot use LSN-based polling")
        return

    # Get index dimension for query vector (once, not every iteration)
    dimension = None
    try:
        stats = idx.describe_index_stats()
        dimension = stats.dimension
    except Exception:
        logger.debug("Could not get index dimension, will use check_fn")

    delta_t = 2  # Use shorter interval for LSN polling
    total_time = 0
    done = False

    while not done:
        logger.debug(
            f"Polling for LSN reconciliation. Target LSN: {target_lsn}, "
            f"operation: {operation_name}, total time: {total_time}s"
        )

        # Try query as a lightweight operation to check LSN
        # Query operations return x-pinecone-max-indexed-lsn header
        if dimension is not None:
            try:
                # Use a minimal query to get headers (this is more efficient than describe_index_stats)
                # We'll use a zero vector query, but we only care about headers
                response = idx.query(
                    top_k=1,
                    vector=[0.0] * dimension,  # Zero vector matching index dimension
                    namespace="",
                )
                reconciled_lsn = None
                if hasattr(response, "_response_info") and response._response_info:
                    reconciled_lsn = response._response_info.get("lsn_reconciled")

                if reconciled_lsn is not None:
                    logger.debug(f"Current reconciled LSN: {reconciled_lsn}, target: {target_lsn}")
                    if is_lsn_reconciled(target_lsn, reconciled_lsn):
                        # LSN is reconciled, check if additional condition is met
                        if check_fn is None or check_fn():
                            done = True
                            logger.debug(f"LSN {target_lsn} is reconciled after {total_time}s")
                    else:
                        logger.debug(
                            f"LSN not yet reconciled. Reconciled: {reconciled_lsn}, "
                            f"target: {target_lsn}"
                        )
                else:
                    # LSN headers not available, fallback to check_fn or return
                    logger.debug("LSN headers not available in response")
                    if check_fn is not None and check_fn():
                        done = True
            except Exception as e:
                logger.debug(f"Error checking LSN: {e}")
                # Fall through to check_fn fallback

        # If dimension is None or LSN check failed, fallback to check_fn
        if not done and dimension is None:
            if check_fn is not None and check_fn():
                done = True

        if not done:
            if total_time >= max_sleep:
                raise TimeoutError(
                    f"Timeout waiting for LSN {target_lsn} to be reconciled after {total_time}s"
                )
            total_time += delta_t
            time.sleep(delta_t)


def fake_api_key():
    return "-".join([random_string(x) for x in [8, 4, 4, 4, 12]])


def jsonprint(obj):
    print(json.dumps(obj.to_dict(), indent=2))


def index_tags(request, run_id):
    test_name = request.node.name
    if test_name is None:
        test_name = ""
    else:
        test_name = test_name.replace(":", "_").replace("[", "_").replace("]", "_")

    tags = {
        "test-suite": "pinecone-python-client",
        "test-run": run_id,
        "test": test_name,
        "created-at": datetime.now().strftime("%Y-%m-%d"),
    }

    if os.getenv("USER"):
        tags["user"] = os.getenv("USER")
    return tags


def delete_backups_from_run(pc: Pinecone, run_id: str):
    for backup in pc.db.backup.list():
        if backup.tags is not None and backup.tags.get("test-run") == run_id:
            pc.db.backup.delete(backup_id=backup.backup_id)
        else:
            logger.info(f"Backup {backup.name} is not a test backup from run {run_id}. Skipping.")


def delete_indexes_from_run(pc: Pinecone, run_id: str):
    indexes_to_delete = []

    for index in pc.db.index.list():
        if index.tags is not None and index.tags.get("test-run") == run_id:
            logger.info(f"Found index {index.name} to delete")
            if index.deletion_protection == "enabled":
                logger.info(f"Index {index.name} has deletion protection enabled. Disabling...")
                pc.update_index(index.name, deletion_protection="disabled")
            else:
                logger.debug(
                    f"Index {index.name} has deletion protection disabled. Proceeding to delete."
                )

            indexes_to_delete.append(index.name)
        else:
            logger.info(f"Index {index.name} is not a test index from run {run_id}. Skipping.")

    for index_name in indexes_to_delete:
        delete_index_with_retry(client=pc, index_name=index_name, retries=3, sleep_interval=10)


def delete_index_with_retry(
    client: Pinecone, index_name: str, retries: int = 0, sleep_interval: int = 5
):
    logger.info(
        f"Deleting index {index_name}, retry {retries}, next sleep interval {sleep_interval}"
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
            delete_index_with_retry(client, index_name, retries + 1, sleep_interval * 2)
        else:
            print(e.__class__)
            print(e)
            raise "Unable to delete index " + index_name
    except Exception as e:
        logger.warning(f"Failed to delete index: {index_name}: {str(e)}")
        raise "Unable to delete index " + index_name


async def asyncio_poll_for_freshness(asyncio_idx, target_namespace, target_vector_count):
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


async def asyncio_wait_until(
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


def default_create_index_params(request, run_id):
    index_name = f"{str(uuid.uuid4())}"
    tags = index_tags(request, run_id)
    cloud = get_environment_var("SERVERLESS_CLOUD", "aws")
    region = get_environment_var("SERVERLESS_REGION", "us-west-2")

    spec = {"serverless": {"cloud": cloud, "region": region}}
    return {"name": index_name, "dimension": 10, "metric": "cosine", "spec": spec, "tags": tags}
