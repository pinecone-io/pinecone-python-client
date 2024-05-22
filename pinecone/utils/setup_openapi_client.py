from pinecone.core.client.api_client import ApiClient
from .user_agent import get_user_agent

def setup_openapi_client(api_klass, config, openapi_config, pool_threads):
    api_client = ApiClient(
        configuration=openapi_config, 
        pool_threads=pool_threads
    )
    api_client.user_agent = get_user_agent(config)
    extra_headers = config.additional_headers or {}
    for key, value in extra_headers.items():
        api_client.set_default_header(key, value)
    client = api_klass(api_client)
    return client
