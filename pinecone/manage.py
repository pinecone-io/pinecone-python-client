from typing import NamedTuple, Optional, Tuple, List
import enum
from pinecone import logger

from pinecone.utils.sentry import sentry_decorator as sentry
from typing import List, NamedTuple, Tuple
from .api_database import DatabaseAPI
from .constants import Config
from pinecone.utils.progressbar import ProgressBar
from pinecone.legacy.specs import database as db_specs


__all__ = [
    "create",
    "delete",
    "ls",
    "update",
    "describe"
    "create_index",
    "delete_index",
    "describe_index",
    "scale_index",
    "list_indexes",
    "ResourceType",
    "IndexDescription",
]

class IndexMeta(NamedTuple):
    name : str  
    index_type : str
    metric : str
    replicas : int
    dimension : int
    shards : int
    index_config : None

class Database(db_specs.DatabaseSpec):
    """The index as a database."""

    def __init__(self, name: str, dimension: int, index_type: str = 'approximated', metric: str = 'cosine', replicas: int = 1, shards: int = 1, index_config: dict = None):
        """"""
        super().__init__(name, dimension, index_type, metric, replicas, shards, index_config)

class ResourceType(enum.Enum):
    INDEX = "index"


class IndexDescription(NamedTuple):
    """Description of an index."""

    name: str
    index_type: str
    metric : str
    dimension: int
    replicas: int
    status: dict
    index_config: dict


def _get_database_api():
    return DatabaseAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)

def index_deploy(name: str, dimension: int, wait: bool = True, index_type: str = 'approximated', metric: str = 'cosine', replicas: int = 1, shards: int = 1, index_config: dict = None)-> Tuple[dict, ProgressBar]:

    db_ = Database(name, dimension, index_type, metric, replicas, shards, index_config)

    api = _get_database_api()

    if name in api.list_services():
        raise RuntimeError(
            "An index with the name '{}' already exists. Please deploy your index with a different name.".format(
                name
            )
        )
    else:
        response = api.deploy(db_.to_json())

    #Wait for index to deploy
    status = api.get_status(name)
    logger.info("Deployment status: {}".format(status))

    return response

def index_stop(db_name:str, wait:bool = True,**kwargs)-> Tuple[dict, ProgressBar]:

    api = _get_database_api()
    response = api.stop(db_name)

    # Wait for the index to stop
    def get_remaining():
        return 1*(db_name in api.list_services())

    pbar = ProgressBar(total=1, get_remaining_fn=get_remaining)
    if wait:
        pbar.watch()
    return response, pbar


def index_ls() -> List[str]:

    api = _get_database_api()
    return api.list_services()


def get_index(name: str) -> IndexMeta:

    api = _get_database_api()
    db_json = api.get_database(name)
    db = Database.from_json(db_json) if db_json else None
    return IndexMeta(name = name, index_type = db.index_type, metric = db.metric, replicas = db.replicas, dimension = db.dimension, shards = db.shards, index_config = db.index_config) or {}


def update_index(name:str,replicas:int):

    api = _get_database_api()
    response = api.update(name,replicas)

    return response


@sentry
def create(name: str, dimension: int, wait: bool = True, index_type: str = 'approximated', metric: str = 'cosine', replicas: int = 1, shards: int = 1, index_config: dict = None, kind:str = "index") -> Optional[dict]:
    """Creates a Pinecone index.
        name=name,
        dimension=dimension,
        wait=wait,
        index_type=index_type,
        metric=metric,
        replicas=replicas,
        index_config=index_config,
    :param name: the name of the index.
    :type name: str
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    """
    if kind == ResourceType.INDEX.value:
        response= index_deploy(name=name, dimension=dimension, wait=wait, index_type=index_type, metric=metric, replicas=replicas, shards=shards, index_config=index_config)
        return response


@sentry
def delete(name: str, wait: bool = True, kind:str = "index") -> Optional[dict]:
    """Deletes a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    """
    if kind == ResourceType.INDEX.value:
        response, _ = index_stop(db_name=name, wait=wait)
        return response


@sentry
def ls(kind:str = "index") -> Optional[List[str]]:
    """Lists all indexes.
    """
    if kind == ResourceType.INDEX.value:
        return index_ls()


@sentry
def describe(name: str,kind:str = "index") -> Optional[IndexDescription]:
    """Describes the index.

    :param name: the name of the index.
    :type name: str
    """
    if kind == ResourceType.INDEX.value:
        response = get_index(name)
        return response

@sentry
def update(name: str, replicas: int, kind:str = "index")->Optional[dict]:
    """Updates the number of replicas for an index
    """
    if kind == ResourceType.INDEX.value:
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
    index_config: dict = None
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
        For additional throughput (QPS) your index needs to support, provision additional replicas.
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
        index_config=index_config,
        kind=ResourceType.INDEX.value
    )


@sentry
def delete_index(name: str, wait: bool = True) -> Optional[dict]:
    """Deletes a Pinecone index.

    :param name: the name of the index.
    :type name: str
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    """
    return delete(name=name, wait=wait, kind=ResourceType.INDEX.value)


@sentry
def list_indexes() -> Optional[List[str]]:
    """Lists all indexes."""
    return ls(kind=ResourceType.INDEX.value)


@sentry
def describe_index(name: str) -> Optional[IndexDescription]:
    """Describes an index.

    :param name: the name of the index.
    :type name: str
    :returns IndexDescription 
    """
    return describe(name=name, kind=ResourceType.INDEX.value)


@sentry
def scale_index(name:str, replicas:int) -> Optional[IndexDescription]:
    """Increases number of replicas for the index.

    :param name: the name of the Index
    :type name: str
    :param replicas: the number of replicas in the index now, lowest value is 0.
    :type replicas: int
    """
    return update(name,replicas, kind=ResourceType.INDEX.value)
