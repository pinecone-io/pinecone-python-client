#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

import json
import time
from typing import NamedTuple

from pinecone.config import Config
from pinecone.core.client.api.index_operations_api import IndexOperationsApi
from pinecone.core.client.api_client import ApiClient
from pinecone.core.client.configuration import Configuration
from pinecone.core.client.model.create_request import CreateRequest
from pinecone.core.client.model.patch_request import PatchRequest
from pinecone.core.utils.sentry import sentry_decorator as sentry
from pinecone.core.utils import get_user_agent

__all__ = [
    "create_index", "delete_index", "describe_index", "list_indexes", "scale_index", "IndexDescription"
]


class IndexDescription(NamedTuple):
    name: str
    index_type: str
    metric: str
    replicas: int
    dimension: int
    shards: int
    index_config: None
    status: None


def _get_api_instance():
    client_config = Config.OPENAPI_CONFIG
    client_config.api_key = client_config.api_key or {}
    client_config.api_key['ApiKeyAuth'] = client_config.api_key.get('ApiKeyAuth', Config.API_KEY)
    client_config.server_variables = {
        **{
            'environment': Config.ENVIRONMENT
        },
        **client_config.server_variables
    }
    api_client = ApiClient(configuration=client_config)
    api_client.user_agent = get_user_agent()
    api_instance = IndexOperationsApi(api_client)
    return api_instance


def _get_status(name: str):
    api_instance = _get_api_instance()
    response = api_instance.describe_index(name)
    return response['status']


@sentry
def create_index(
        name: str,
        dimension: int,
        timeout: int = None,
        index_type: str = "approximated",
        metric: str = "cosine",
        replicas: int = 1,
        shards: int = 1,
        index_config: dict = None
):
    """Creates a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param dimension: the dimension of vectors that would be inserted in the index
    :param index_type: type of index, one of {"approximated", "exact"}, defaults to "approximated".
        The "approximated" index uses fast approximate search algorithms developed by Pinecone.
        The "exact" index uses accurate exact search algorithms.
        It performs exhaustive searches and thus it is usually slower than the "approximated" index.
    :type index_type: str, optional
    :param metric: type of metric used in the vector index, one of {"cosine", "dotproduct", "euclidean"}, defaults to "cosine".
        Use "cosine" for cosine similarity,
        "dotproduct" for dot-product,
        and "euclidean" for euclidean distance.
    :type metric: str, optional
    :param replicas: the number of replicas, defaults to 1.
        Use at least 2 replicas if you need high availability (99.99% uptime) for querying.
        For additional throughput (QPS) your index needs to support, provision additional replicas.
    :type replicas: int, optional
    :param shards: the number of shards per index, defaults to 1.
        Use 1 shard per 1GB of vectors
    :type shards: int,optional
    :param index_config: Advanced configuration options for the index
    :type timeout: int, optional
    :param timeout: Timeout for wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds; if -1, return immediately and do not wait. Default: None
    """
    api_instance = _get_api_instance()

    api_instance.create_index(create_request=CreateRequest(
        name=name,
        dimension=dimension,
        index_type=index_type,
        metric=metric,
        replicas=replicas,
        shards=shards,
        index_config=index_config or {}
    ))

    def is_ready():
        status = _get_status(name)
        ready = status['ready']
        return ready

    if timeout == -1:
        return
    if timeout is None:
        while not is_ready():
            time.sleep(5)
    else:
        while (not is_ready()) and timeout >= 0:
            time.sleep(5)
            timeout -= 5
    if timeout and timeout < 0:
        raise (TimeoutError(
            'Please call the describe_index API ({}) to confirm index status.'.format(
                'https://www.pinecone.io/docs/api/operation/describe_index/')))


@sentry
def delete_index(name: str, timeout: int = None):
    """Deletes a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param timeout: Timeout for wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds; if -1, return immediately and do not wait. Default: None
    :type timeout: int, optional
    """
    api_instance = _get_api_instance()
    api_instance.delete_index(name)

    def get_remaining():
        return name in api_instance.list_indexes()

    if timeout == -1:
        return

    if timeout is None:
        while get_remaining():
            time.sleep(5)
    else:
        while get_remaining() and timeout >= 0:
            time.sleep(5)
            timeout -= 5
    if timeout and timeout < 0:
        raise (TimeoutError(
            'Please call the list_indexes API ({}) to confirm if index is deleted'.format(
                'https://www.pinecone.io/docs/api/operation/list_indexes/')))


@sentry
def list_indexes():
    """Lists all indexes."""
    api_instance = _get_api_instance()
    response = api_instance.list_indexes()
    return response


@sentry
def describe_index(name: str):
    """Describes a Pinecone index.

    :param: the name of the index
    :return: Description of an index
    """
    api_instance = _get_api_instance()
    response = api_instance.describe_index(name)
    db = response['database']
    ready = response['status']['ready']
    return IndexDescription(name=db['name'], index_type=db['index_type'], metric=db['metric'],
                            replicas=db['replicas'], dimension=db['dimension'], shards=db['shards'],
                            index_config=db['index_config'], status={'ready': ready})


@sentry
def scale_index(name: str, replicas: int):
    """Increases number of replicas for the index.

    :param name: the name of the Index
    :type name: str
    :param replicas: the number of replicas in the index now, lowest value is 0.
    :type replicas: int
    """
    api_instance = _get_api_instance()
    api_instance.scale_index(name, patch_request=PatchRequest(replicas=replicas))
