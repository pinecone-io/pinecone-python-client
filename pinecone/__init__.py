#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import os
from loguru import logger
import sys
from .utils.sentry import sentry_decorator as sentry
from .constants import Config, CLIENT_VERSION as __version__
from .database import  create_index,delete_index,describe_index,list_indexes,scale_index,get_status
from .index import Index
from .experimental.openapi.models import FetchResponse, ListNamespacesResponse, ListResponse, ProtobufAny, \
    QueryRequest, QueryResponse, QueryVector, RpcStatus, ScoredVector, SingleQueryResults, SummarizeResponse, \
    UpsertRequest, Vector
from .experimental.openapi.exceptions import OpenApiException, ApiAttributeError, ApiTypeError, ApiValueError, \
    ApiKeyError, ApiException

__all__ = [
    "init",
    # Control plane names
    "create_index", "delete_index", "describe_index", "list_indexes", "scale_index", "IndexDescription",
    # Data plane
    "Index",
    # Data plane OpenAPI models
    "FetchResponse", "ListNamespacesResponse", "ListResponse", "ProtobufAny", "QueryRequest", "QueryResponse",
    "QueryVector", "RpcStatus", "ScoredVector", "SingleQueryResults", "SummarizeResponse", "UpsertRequest", "Vector",
    # Data plane OpenAPI exceptions
    "OpenApiException", "ApiAttributeError", "ApiTypeError", "ApiValueError", "ApiKeyError", "ApiException",
    # Kept for backwards-compatibility
    "UpsertResult", "DeleteResult", "QueryResult", "FetchResult", "InfoResult",
]

logger.remove()
logger.add(sys.stdout, enqueue=True, level=(os.getenv("PINECONE_LOGGING") or "ERROR"))

UpsertResult = None
DeleteResult = None
QueryResult = None
FetchResult = None
InfoResult = None

@sentry
def init(project_name: str = None, api_key: str = None, host: str = None, environment: str = None, config: str = "~/.pinecone", **kwargs):
    """Initializes the Pinecone client.

    :param api_key: Required if not set in config file or by environment variable ``PINECONE_API_KEY``.
    :param host: Optional. Controller host.
    :param environment: Optional. Deployment environment.
    :param config: Optional. An INI configuration file.
    """
    Config.reset(project_name=project_name, api_key=api_key, controller_host=host, environment=environment, config_file=config, **kwargs)
    if not bool(Config.API_KEY):
        logger.warning("API key is required.")


# Init
init()
