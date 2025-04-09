import re
import os
import time
import random
import string
import logging
from typing import Any
from datetime import datetime
import json
from pinecone.db_data import _Index
from typing import List

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
    max_sleep: int = int(os.environ.get("FRESHNESS_TIMEOUT_SECONDS", 60)),
) -> None:
    delta_t = 5
    total_time = 0
    done = False
    while not done:
        logger.debug(
            f'Waiting for namespace "{namespace}" to have vectors. Total time waited: {total_time} seconds'
        )
        stats = idx.describe_index_stats()
        if (
            namespace in stats.namespaces
            and stats.namespaces[namespace].vector_count >= expected_count
        ):
            done = True
        elif total_time > max_sleep:
            raise TimeoutError(f"Timed out waiting for namespace {namespace} to have vectors")
        else:
            total_time += delta_t
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


def fake_api_key():
    return "-".join([random_string(x) for x in [8, 4, 4, 4, 12]])


def jsonprint(obj):
    print(json.dumps(obj.to_dict(), indent=2))
