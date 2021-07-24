from typing import NamedTuple, Optional, Tuple, List
import enum
from pinecone import logger

from pinecone.utils.sentry import sentry_decorator as sentry
from .service import deploy as service_deploy, stop as service_stop, ls as service_ls, describe as service_describe
from .graph import IndexGraph, IndexConfig

__all__ = [
    "create",
    "delete",
    "ls",
    "describe",
    "create_index",
    "delete_index",
    "describe_index",
    "list_indexes",
    "ResourceType",
    "ResourceDescription",
]


class ResourceType(enum.Enum):
    INDEX = "index"


class ResourceDescription(NamedTuple):
    """Description of a resource."""

    name: str
    kind: str
    status: dict
    config: dict


@sentry
def create(name: str, kind: str = "index", wait: bool = True, **kwargs) -> Optional[dict]:
    """Creates a Pinecone resource.

    :param name: the name of the resource.
    :type name: str
    :param kind: what kind of resource. Defaults to ``index``.
    :type kind: str
    :param wait: wait for the resource to deploy. Defaults to ``True``
    :type wait: bool
    :param `**kwargs`: see resource-specific configurations.
        For example, you can refer to :class:`IndexConfig` for
        configuration options for a Pinecone index.
    """
    if kind == ResourceType.INDEX.value:
        response, _ = service_deploy(service_name=name, graph=IndexGraph(**kwargs), wait=wait)
        return response
    logger.warning("Unrecognized resource type '{}'.".format(kind))


@sentry
def delete(name: str, kind: str = "index", wait: bool = True) -> Optional[dict]:
    """Deletes a Pinecone resource.

    :param name: the name of the resource.
    :type name: str
    :param kind: what kind of resource. Defaults to ``index``.
    :type kind: str
    :param wait: wait for the resource to deploy. Defaults to ``True``
    :type wait: bool
    """
    if kind == ResourceType.INDEX.value:
        response, _ = service_stop(service_name=name, wait=wait)
        return response
    logger.warning("Unrecognized resource type '{}'.".format(kind))


@sentry
def ls(kind: str = "index") -> Optional[List[str]]:
    """Lists all resources of a certain kind.

    :param kind: what kind of resource. Defaults to ``index``.
    :type kind: str
    """
    if kind == ResourceType.INDEX.value:
        return service_ls()
    logger.warning("Unrecognized resource type '{}'.".format(kind))


@sentry
def describe(name: str, kind: str = "index") -> Optional[ResourceDescription]:
    """Describes the resource.

    :param name: the name of the resource.
    :type name: str
    :param kind: what kind of resource. Defaults to ``index``.
    :type kind: str
    """
    if kind == ResourceType.INDEX.value:
        desc = service_describe(service_name=name)
        graph = desc.graph
        config = IndexConfig._from_graph(graph)._asdict()
        return ResourceDescription(name=desc.name, kind=kind, status=desc.status, config=config)
    logger.warning("Unrecognized resource type '{}'.".format(kind))


@sentry
def create_index(
    name: str,
    wait: bool = True,
    index_type: str = "approximated",
    metric: str = "cosine",
    shards: int = 1,
    replicas: int = 1,
    gateway_replicas: int = 1,
    index_args: dict = None,
) -> Optional[dict]:
    """Creates a Pinecone index.

    :param name: the name of the index.
    :type name: str
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
    :param shards: the number of shards for the index, defaults to 1.
        As a general guideline, use 1 shard per 1 GB of data.
    :type shards: int, optional
    :param replicas: the number of replicas, defaults to 1.
        Use at least 2 replicas if you need high availability (99.99% uptime) for querying.
        For additional throughput (QPS) your service needs to support, provision additional replicas.
    :type replicas: int, optional
    :param gateway_replicas: number of replicas of both the gateway and the aggregator.
    :type gateway_replicas: int
    :param index_args: advanced arguments for the index instance in the graph.
    :type index_args: dict
    """
    return create(
        name=name,
        kind=ResourceType.INDEX.value,
        wait=wait,
        index_type=index_type,
        metric=metric,
        shards=shards,
        replicas=replicas,
        gateway_replicas=gateway_replicas,
        index_args=index_args,
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
    return ls(kind=ResourceType.INDEX.value)


@sentry
def describe_index(name: str) -> Optional[ResourceDescription]:
    """Describes an index.

    :param name: the name of the index.
    :type name: str
    """
    return describe(name=name, kind=ResourceType.INDEX.value)