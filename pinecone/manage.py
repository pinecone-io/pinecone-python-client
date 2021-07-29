from typing import NamedTuple, Optional, Tuple, List
import enum
from pinecone import logger

from pinecone.utils.sentry import sentry_decorator as sentry
from .database import deploy as index_deploy, stop as index_stop, ls as index_ls, describe as get_index, update as update_index
from .database_spec import Database

__all__ = [
    "create",
    "delete",
    "ls",
    "create_index",
    "delete_index",
    "describe_index",
    "list_indexes",
    "ResourceType",
    "IndexDescription",
]


class ResourceType(enum.Enum):
    DATABASE = "database"


class IndexDescription(NamedTuple):
    """Description of an index."""

    name: str
    index_type: str
    metric : str
    dimension: int
    replicas: int
    status: dict
    engine_config: dict


@sentry
def create(name: str, dimension: int, wait: bool = True, index_type: str='approximated',metric: str='cosine',replicas: int= 1,shards: int= 1, engine_config: {}=None) -> Optional[dict]:
    """Creates a Pinecone index.
        name=name,
        dimension=dimension,
        wait=wait,
        index_type=index_type,
        metric=metric,
        replicas=replicas,
        index_args=index_args,
    :param name: the name of the index.
    :type name: str
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    :param `**kwargs`: see resource-specific configurations.
        For example, you can refer to :class:`IndexConfig` for
        configuration options for a Pinecone index.
    """
    response, _ = index_deploy(name=name, dimension=dimension, wait=wait, index_type=index_type, metric=metric, replicas=replicas, shards=shards, engine_config=engine_config)
    return response
    # logger.warning("Unrecognized resource type '{}'.".format(kind))


@sentry
def delete(name: str, wait: bool = True) -> Optional[dict]:
    """Deletes a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    """

    response, _ = index_stop(index_name=name, wait=wait)
    return response
    # logger.warning("Unrecognized resource type '{}'.".format(kind))


@sentry
def ls() -> Optional[List[str]]:
    """Lists all indexes.
    """

    return index_ls()
    logger.warning("Unrecognized resource type '{}'.".format(kind))


@sentry
def describe(name: str) -> Optional[IndexDescription]:
    """Describes the index.

    :param name: the name of the index.
    :type name: str
    """
    response = get_index(name)
    return response

@sentry
def update(name: str,replicas: int)->Optional[dict]:
    """Updates the number of replicas for an index
    """
    return update_index(name,replicas)

@sentry
def create_index(
    name: str,
    dimension: int,
    wait: bool = True,
    index_type: str = "approximated",
    metric: str = "cosine",
    replicas: int = 1,
    shards: int = 1,
    engine_config: dict = None
) -> Optional[dict]:
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
        For additional throughput (QPS) your service needs to support, provision additional replicas.
    :type replicas: int, optional
    :param shards: the number of shards per index, defaults to 1.
        Use 1 shard per 1GB of vectors
    :type shards: int,optional
    """
    return create(
        name=name,
        dimension=dimension,
        wait=wait,
        index_type=index_type,
        metric=metric,
        replicas=replicas,
        shards=shards,
        engine_config=engine_config
    )


@sentry
def delete_index(name: str, wait: bool = True) -> Optional[dict]:
    """Deletes a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    """
    return delete(name=name, kind=ResourceType.INDEX.value, wait=wait)


@sentry
def list_indexes() -> Optional[List[str]]:
    """Lists all indexes."""
    return ls()


@sentry
def describe_index(name: str) -> Optional[IndexDescription]:
    """Describes an index.

    :param name: the name of the index.
    :type name: str
    :returns ResourceDescription containing DatabaseSpec
    """
    return describe(name=name)


@sentry
def scale_index(name:str,replicas:int) -> Optional[IndexDescription]:
    """Increases number of replicas for the index.

    :param name: the name of the Index
    :type name: str
    :param replicas: the number of replicas in the index now, lowest value is 0.
    :type replicas: int
    """
    return update()
