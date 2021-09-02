from collections import Iterable

from pinecone import Config
from pinecone.core.openapi import ApiClient, Configuration
from pinecone.core.utils.sentry import sentry_decorator as sentry
from .core.openapi.models import FetchResponse, ListNamespacesResponse, ProtobufAny, QueryRequest, QueryResponse, QueryVector, RpcStatus, ScoredVector, SingleQueryResults, SummarizeResponse, UpsertRequest, Vector

__all__ = [
    "Index", "FetchResponse", "ListNamespacesResponse", "ProtobufAny", "QueryRequest", "QueryResponse", "QueryVector", "RpcStatus", "ScoredVector", "SingleQueryResults", "SummarizeResponse", "UpsertRequest", "Vector"
]





from pinecone.core.openapi.api.vector_service_api import VectorServiceApi
from pinecone.core.utils import fix_tuple_length


class Index(ApiClient):

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
        super().__init__(configuration=openapi_client_config, pool_threads=pool_threads)
        self._vector_api = VectorServiceApi(self)

    @sentry
    def upsert(self, vectors, **kwargs):
        def _vector_transform(item):
            if isinstance(item, Vector):
                return item
            if isinstance(item, tuple):
                id, values, metadata = fix_tuple_length(item, 3)
                return Vector(id=id, values=values, metadata=metadata or {})
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

        return self._vector_api.vector_service_upsert(
            UpsertRequest(vectors=list(map(_vector_transform, vectors)), **kwargs)
        )

    @sentry
    def delete(self, *args, **kwargs):
        return self._vector_api.vector_service_delete(*args, **kwargs)

    @sentry
    def fetch(self, *args, **kwargs):
        return self._vector_api.vector_service_fetch(*args, **kwargs)

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

        return self._vector_api.vector_service_query(
            QueryRequest(queries=list(map(_query_transform, queries)), **kwargs)
        )

    @sentry
    def list_namespaces(self, *args, **kwargs):
        return self._vector_api.vector_service_list_namespaces(*args, **kwargs)

    @sentry
    def summarize(self, *args, **kwargs):
        return self._vector_api.vector_service_summarize(*args, **kwargs)
