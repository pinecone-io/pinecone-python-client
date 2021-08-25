from pinecone import Config
from pinecone.experimental.openapi import ApiClient, Configuration
from pinecone.utils.sentry import sentry_decorator as sentry
from .experimental.openapi.models import QueryRequest, UpsertRequest

__all__ = [
    "Index",
]

from pinecone.experimental.openapi.api.vector_service_api import VectorServiceApi


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
    def upsert(self, *args, **kwargs):
        return self._vector_api.vector_service_upsert(UpsertRequest(*args, **kwargs))

    @sentry
    def delete(self, *args, **kwargs):
        return self._vector_api.vector_service_delete(*args, **kwargs)

    @sentry
    def fetch(self, *args, **kwargs):
        return self._vector_api.vector_service_fetch(*args, **kwargs)

    @sentry
    def query(self, *args, **kwargs):
        return self._vector_api.vector_service_query(QueryRequest(*args, **kwargs))

    @sentry
    def list(self, *args, **kwargs):
        return self._vector_api.vector_service_list(*args, **kwargs)

    @sentry
    def list_namespaces(self, *args, **kwargs):
        return self._vector_api.vector_service_list_namespaces(*args, **kwargs)

    @sentry
    def summarize(self, *args, **kwargs):
        return self._vector_api.vector_service_summarize(*args, **kwargs)
