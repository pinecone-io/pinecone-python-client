from typing import NamedTuple, Iterable, Tuple, List

from pinecone.utils.sentry import sentry_decorator as sentry
from pinecone.constants import Config
from pinecone.constants import ENABLE_PROGRESS_BAR
import enum
from pinecone import logger
from pinecone.utils.progressbar import ProgressBar
from pinecone.experimental.openapi.api.database_service_api import DatabaseServiceApi
from pinecone.experimental.openapi.api_client import ApiClient
from pinecone.experimental.openapi.model.create_request import CreateRequest
from pinecone.experimental.openapi.model.patch_request import PatchRequest
from pinecone.experimental.openapi.model.index_meta import IndexMeta
from pinecone.experimental.openapi.model.status_response import StatusResponse
from pinecone.experimental.openapi.configuration import Configuration


def get_api_instance():
    client_config = Configuration.get_default_copy()
    client_config.api_key = client_config.api_key or {}
    client_config.api_key['ApiKeyAuth'] = client_config.api_key.get('ApiKeyAuth', Config.API_KEY)
    client_config.server_variables = {
        **{
            'environment': Config.ENVIRONMENT
        },
        **client_config.server_variables
    }
    api_client = ApiClient(configuration=client_config)
    api_instance = DatabaseServiceApi(api_client)
    return api_instance


@sentry
def create_index(
        name: str,
        dimension: int,
        wait: bool = True,
        index_type: str = "approximated",
        metric: str = "cosine",
        replicas: int = 1,
        shards: int = 1,
        index_config: dict = {}
):
    """Creates a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param dimension: the dimension of vectors that would be inserted in the index
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
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
    """
    api_instance = get_api_instance()

    response = api_instance.create_index(create_request=CreateRequest(
        name=name,
        dimension=dimension,
        index_type=index_type,
        metric=metric,
        replicas=replicas,
        shards=shards,
        index_config=index_config
    ))

    return response


@sentry
def delete_index(name: str, wait: bool = True):
    """Deletes a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    """
    api_instance = get_api_instance()
    response = api_instance.delete_index(name)
    # while name in api_instance.list_indexes():
    #     continue
    return response

@sentry
def list_indexes():
    """Lists all indexes."""
    api_instance = get_api_instance()
    response = api_instance.list_indexes()
    return response


@sentry
def describe_index(name: str):
    """Describes a Pinecone index.

    :param: the name of the index
    :return: Description of an index
    """
    api_instance = get_api_instance()
    response = api_instance.describe_index(name)
    return response['database']


@sentry
def scale_index(name: str, replicas: int):
    """Increases number of replicas for the index.

    :param name: the name of the Index
    :type name: str
    :param replicas: the number of replicas in the index now, lowest value is 0.
    :type replicas: int
    """
    api_instance = get_api_instance()
    response = api_instance.scale_index(name, patch_request=PatchRequest(replicas=replicas))
    return response