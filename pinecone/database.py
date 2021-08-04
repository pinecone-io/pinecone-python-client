#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
from typing import List, NamedTuple, Tuple
import json

from pinecone import logger

from .api_database import DatabaseAPI
from .constants import Config
from pinecone.utils.sentry import sentry_decorator as sentry
from pinecone.utils.progressbar import ProgressBar
# from .database_spec import Database
from pinecone.specs import database as db_specs

__all__ = ["deploy", "stop", "ls","describe","update"]

def _get_database_api():
    return DatabaseAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)

class DatabaseMeta(NamedTuple):
    name : str
    index_type : str
    metric : str
    replicas : int
    dimension : int
    shards : int
    index_config : None

class Database(db_specs.DatabaseSpec):
    """The index as a database."""

    def __init__(self, name: str, dimension: int, index_type: str = 'approximated', metric: str = 'cosine', replicas: int = 1, shards: int = 1, index_config: {} = None):
        """"""
        super().__init__(name, dimension, index_type, metric, replicas, shards, index_config)


@sentry
def deploy(name: str, dimension: int, wait: bool = True, index_type: str = 'approximated', metric: str = 'cosine', replicas: int = 1, shards: int = 1, index_config: {} = None)-> Tuple[dict, ProgressBar]:
    """Create a new Pinecone index from the database spec
    :param db_name : name of the index
    :type db_name : str
    :param db : database spec that defines the index
    :type database: class:'pinecone.specs.database'
    :param wait: wait for the index to deploy. Defaults to ``True``
    :type wait: bool
    :param index_type: type of index, one of {"approximated", "exact"}, defaults to "approximated".
        The "approximated" index uses fast approximate search algorithms developed by Pinecone.
        The "exact" index uses accurate exact search algorithms.
        It performs exhaustive searches and thus it is usually slower than the "approximated" index.
    :param index_type: str
    :type metric: str, optional
    :param replicas: the number of replicas, defaults to 1.
        Use at least 2 replicas if you need high availability (99.99% uptime) for querying.
        For additional throughput (QPS) your service needs to support, provision additional replicas.
    :type replicas: int, optional
    :param shards: the number of shards per index, defaults to 1.
        Use 1 shard per 1GB of vectors
    :type shards: int,optional
    :param index_config: configurations for specific index types
    :type index_config: dict,optional
    """
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

    pbar = ProgressBar(total=1, get_remaining_fn=1)
    if wait:
        pbar.watch()
    return response

@sentry
def stop(db_name:str, wait:bool = True,**kwargs)-> Tuple[dict, ProgressBar]:
    """
    Stops a database
    :param db_name: name of the index
    :type db_name:str
    :param wait: wait for the database to stop. Default=True
    :type wait:bool
    """
    api = _get_database_api()
    response = api.stop(db_name)

    # Wait for the index to stop
    def get_remaining():
        return 1*(db_name in api.list_services())

    pbar = ProgressBar(total=1, get_remaining_fn=get_remaining)
    if wait:
        pbar.watch()
    return response, pbar

@sentry
def ls() -> List[str]:
    """Returns all index names."""
    api = _get_database_api()
    return api.list_services()

@sentry
def describe(name: str) -> DatabaseMeta:
    """Returns the metadata of a service.

    :param service_name: name of the service
    :type service_name: str
    :return: :class:`ServiceMeta`
    """
    api = _get_database_api()
    db_json = api.get_database(name)
    db = Database.from_json(db_json) if db_json else None
    return DatabaseMeta(name=name, index_type=db.index_type,metric=db.metric,replicas=db.replicas,dimension=db.dimension,shards= db.shards,index_config=db.index_config) or {}


def update(name:str,replicas:int):
    """Returns the new spec for the updated index

    :param name: name of the index
    :type name: str
    :param replicas: number of replicas
    :param type: int
    """
    api = _get_database_api()
    response = api.update(name,replicas)

    return response


