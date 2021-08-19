from abc import ABC
from typing import NamedTuple

import pinecone.experimental.openapi
from pinecone import Config
from pinecone.constants import CLIENT_VERSION
from pinecone.experimental.openapi import ApiClient, Configuration
from pinecone.experimental.openapi.api import vector_service_api
from pinecone.utils.sentry import sentry_decorator as sentry


class ClientConfig(NamedTuple):
    environment: str = ""
    api_key: str = ""
    project_name: str = ""
    controller_host: str = ""
    openapi_client_config: Configuration = None
    pool_threads: int = 1


class PineconeApiClient(ApiClient):

    def __init__(self, index_name: str, config: ClientConfig = None):
        config = config or ClientConfig()
        config.openapi_client_config = config.openapi_client_config or Configuration.get_default_copy()
        config.openapi_client_config.api_key = {'ApiKeyAuth': (config.openapi_client_config.api_key or Config.API_KEY)}
        config.openapi_client_config.server_variables = config.openapi_client_config.server_variables or {}
        config.openapi_client_config.server_variables = {
            **{
                'environment': config.environment or Config.ENVIRONMENT,
                'index-name': index_name,
                'project-name': config.project_name or Config.PROJECT_NAME
            },
            **config.openapi_client_config.server_variables
        }
        super().__init__(configuration=config.openapi_client_config, pool_threads=config.pool_threads)

        # self.fixed_metadata = {
        #     "api-key": Config.API_KEY,
        #     "service-name": index_name,
        #     "client-version": CLIENT_VERSION
        # }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @sentry
    def close(self):
        """Closes the connection to the index."""
        try:
            self.api_client.close()
        except TypeError:
            pass
