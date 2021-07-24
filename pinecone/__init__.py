#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import os
import configparser
from loguru import logger
import sys
from pinecone.specs.service import Service  # noqa
from pinecone.specs.traffic_router import TrafficRouter  # noqa
from pinecone.utils.sentry import sentry_decorator as sentry
from .constants import Config
from .manage import create_index, delete_index, describe_index, list_indexes, ResourceDescription
from .index import Index, UpsertResult, DeleteResult, QueryResult, FetchResult, InfoResult

__all__ = [
    "init",
    "create_index",
    "delete_index",
    "describe_index",
    "list_indexes",
    "ResourceDescription",
    "Index",
    "UpsertResult",
    "DeleteResult",
    "QueryResult",
    "FetchResult",
    "InfoResult",
]

logging_level = os.environ.get("PINECONE_LOGGING", default="ERROR")
logger.remove()
logger.add(sys.stdout, level=logging_level)

__version__ = open(os.path.join(os.path.dirname(__file__), "__version__")).read().strip()


@sentry
def init(api_key: str = None, host: str = None, environment: str = None, config: str = "~/.pinecone", **kwargs):
    """Initializes the Pinecone client.

    :param api_key: Required if not set in config file or by environment variable ``PINECONE_API_KEY``.
    :param host: Optional. Controller host.
    :param environment: Optional. Deployment environment.
    :param config: Optional. An INI configuration file.
    """
    Config.reset(api_key=api_key, controller_host=host, environment=environment, config_file=config, **kwargs)
    if not bool(Config.API_KEY):
        logger.warning("API key is required.")


# Init
init()