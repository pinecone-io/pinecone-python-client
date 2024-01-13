import re
import os
import time
import random
import string
from typing import Any

def random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def generate_index_name(test_name: str) -> str:
    buildNumber = os.getenv('GITHUB_BUILD_NUMBER', None)
    
    if test_name.startswith('test_'):
        test_name = test_name[5:]

    # Trim name length to save space for other info in name
    test_name = test_name[:20]

    # Remove trailing underscore, if any
    if test_name.endswith('_'):
        test_name = test_name[:-1]

    name_parts = [buildNumber, test_name, random_string(45)]
    index_name = '-'.join([x for x in name_parts if x is not None])
    
    # Remove invalid characters
    replace_with_hyphen = re.compile(r'[\[\(_,\s]')
    index_name = re.sub(replace_with_hyphen, '-', index_name)
    replace_with_empty = re.compile(r'[\]\)\.]')
    index_name = re.sub(replace_with_empty, '', index_name)

    max_length = 45
    index_name = index_name[:max_length]

    # Trim final character if it is not alphanumeric
    if test_name.endswith('_') or test_name.endswith('-'):
        test_name = test_name[:-1]

    return index_name.lower()

def get_environment_var(name: str, defaultVal: Any = None) -> str:
    val = os.getenv(name, defaultVal)
    if (val is None):
        raise Exception('Expected environment variable '  + name + ' is not set')
    else:
        return val

def poll_stats_for_namespace(idx, namespace, expected_count, max_sleep=int(os.environ.get('FRESHNESS_TIMEOUT_SECONDS', 60))):
    delta_t = 5
    total_time=0
    done = False
    while not done:
        print(f'Waiting for namespace "{namespace}" to have vectors. Total time waited: {total_time} seconds')
        stats = idx.describe_index_stats()
        if namespace in stats.namespaces and stats.namespaces[namespace].vector_count >= expected_count:
            done = True
        elif total_time > max_sleep:
            raise TimeoutError(f'Timed out waiting for namespace {namespace} to have vectors')
        else:
            total_time += delta_t
            time.sleep(delta_t)

def poll_fetch_for_ids_in_namespace(idx, ids, namespace):
    max_sleep=int(os.environ.get('FRESHNESS_TIMEOUT_SECONDS', 60))
    delta_t = 5
    total_time=0
    done = False
    while not done:
        print(f'Attempting to fetch from "{namespace}". Total time waited: {total_time} seconds')
        results = idx.fetch(ids=ids, namespace=namespace)
        print(results)

        all_present = all(key in results.vectors for key in ids)
        if all_present:
            done = True

        if total_time > max_sleep:
            raise TimeoutError(f'Timed out waiting for namespace {namespace} to have vectors')
        else:
            total_time += delta_t
            time.sleep(delta_t)

def fake_api_key():
    return '-'.join([random_string(x) for x in [8, 4, 4, 4, 12]])