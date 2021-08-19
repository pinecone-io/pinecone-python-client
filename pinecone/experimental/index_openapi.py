from pinecone import Config
from pinecone.experimental.openapi import ApiClient, Configuration


class PineconeApiClient(ApiClient):

    def __init__(self, index_name: str, openapi_client_config: Configuration = None, pool_threads=1):
        openapi_client_config = openapi_client_config or Configuration.get_default_copy()
        # openapi_client_config.host = openapi_client_config.host or \
        #                              f'https://{index_name}-{Config.PROJECT_NAME}.svc.{Config.ENVIRONMENT}.pinecone.io'
        openapi_client_config.api_key = openapi_client_config.api_key or {}
        openapi_client_config.api_key['ApiKeyAuth'] = openapi_client_config.api_key['ApiKeyAuth'] or Config.API_KEY
        openapi_client_config.server_variables = openapi_client_config.server_variables or {}
        openapi_client_config.server_variables = {
            **{
                'environment': Config.ENVIRONMENT,
                'index-name': index_name,
                'project-name': Config.PROJECT_NAME
            },
            **openapi_client_config.server_variables
        }
        super().__init__(configuration=openapi_client_config, pool_threads=pool_threads)

        # self.fixed_metadata = {
        #     "api-key": Config.API_KEY,
        #     "service-name": index_name,
        #     "client-version": CLIENT_VERSION
        # }
