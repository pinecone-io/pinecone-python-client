#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from collections import Iterable

from pinecone import Config
from pinecone.core.client import ApiClient, Configuration
from pinecone.core.utils.sentry import sentry_decorator as sentry
from .core.client.models import FetchResponse, ProtobufAny, QueryRequest, QueryResponse, QueryVector, RpcStatus, ScoredVector, SingleQueryResults, SummarizeResponse, UpsertRequest, Vector
from .core.utils.constants import CLIENT_VERSION_HEADER, CLIENT_ID
from pinecone.core.client.api.vector_operations_api import VectorOperationsApi
from pinecone.core.utils import fix_tuple_length

__all__ = [
    "Index", "FetchResponse", "ProtobufAny", "QueryRequest", "QueryResponse", "QueryVector", "RpcStatus", "ScoredVector", "SingleQueryResults", "SummarizeResponse", "UpsertRequest", "Vector"
]


class Index:

    def __init__(self, index_name: str, openapi_client_config: Configuration = None, pool_threads=1):
        openapi_client_config = openapi_client_config or Configuration.get_default_copy()
        openapi_client_config.api_key = openapi_client_config.api_key or {}
        openapi_client_config.api_key['ApiKeyAuth'] = openapi_client_config.api_key.get('ApiKeyAuth', Config.API_KEY)
        openapi_client_config.server_variables = openapi_client_config.server_variables or {}
        openapi_client_config.server_variables = {
            **{
                'environment': Config.ENVIRONMENT,
                'service_prefix': f'{index_name}-{Config.PROJECT_NAME}'
            },
            **openapi_client_config.server_variables
        }
        self._api_client = ApiClient(configuration=openapi_client_config, pool_threads=pool_threads)
        self._api_client.set_default_header(CLIENT_VERSION_HEADER, CLIENT_ID)
        self._vector_api = VectorOperationsApi(self)

    @sentry
    def upsert(self, vectors, **kwargs):
        def _vector_transform(item):
            if isinstance(item, Vector):
                return item
            if isinstance(item, tuple):
                id, values, metadata = fix_tuple_length(item, 3)
                return Vector(id=id, values=values, metadata=metadata or {})
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

        return self._vector_api.upsert(
            UpsertRequest(vectors=list(map(_vector_transform, vectors)), **kwargs)
        )

    @sentry
    def delete(self, *args, **kwargs):
        return self._vector_api.delete(*args, **kwargs)

    @sentry
    def fetch(self, *args, **kwargs):
        return self._vector_api.fetch(*args, **kwargs)

    @sentry
    def query(self, queries, **kwargs):
        def _query_transform(item):
            if isinstance(item, QueryVector):
                return item
            if isinstance(item, tuple):
                values, filter = fix_tuple_length(item, 2)
                return QueryVector(values=values, filter=filter)
            if isinstance(item, Iterable):
                return QueryVector(values=item)
            raise ValueError(f"Invalid query vector value passed: cannot interpret type {type(item)}")

        return self._vector_api.query(
            QueryRequest(queries=list(map(_query_transform, queries)), **kwargs)
        )

    @sentry
    def summarize(self, *args, **kwargs):
        return self._vector_api.summarize(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._api_client.__exit__(exc_type, exc_value, traceback)