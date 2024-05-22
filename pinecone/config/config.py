from typing import NamedTuple, Optional, Dict
import os

from pinecone.exceptions import PineconeConfigurationError
from pinecone.config.openapi import OpenApiConfigFactory
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration
from pinecone.utils import normalize_host
from pinecone.utils.constants import SOURCE_TAG

class Config(NamedTuple):
    api_key: str = ""
    host: str = ""
    proxy_url: Optional[str] = None
    proxy_headers: Optional[Dict[str, str]] = None
    ssl_ca_certs: Optional[str] = None
    ssl_verify: Optional[bool] = None
    additional_headers: Optional[Dict[str, str]] = {}
    source_tag: Optional[str] = None

class ConfigBuilder:
    """

    Configurations are resolved in the following order:

    - configs passed as keyword parameters
    - configs specified in environment variables
    - default values (if applicable)
    """

    """Initializes the Pinecone client.

    :param api_key: Required if not set in config file or by environment variable ``PINECONE_API_KEY``.
    :param host: Optional. Controller host.
    :param openapi_config: Optional. Set OpenAPI client configuration.
    """

    @staticmethod
    def build(
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        **kwargs,
    ) -> Config:
        api_key = api_key or kwargs.pop("api_key", None) or os.getenv("PINECONE_API_KEY")
        host = host or kwargs.pop("host", None)
        host = normalize_host(host)
        source_tag = kwargs.pop(SOURCE_TAG, None)

        if not api_key:
            raise PineconeConfigurationError("You haven't specified an Api-Key.")
        if not host:
            raise PineconeConfigurationError("You haven't specified a host.")

        return Config(api_key, host, proxy_url, proxy_headers, ssl_ca_certs, ssl_verify, additional_headers, source_tag)
    
    @staticmethod
    def build_openapi_config(
        config: Config, openapi_config: Optional[OpenApiConfiguration] = None, **kwargs
    ) -> OpenApiConfiguration:
        if openapi_config:
            openapi_config = OpenApiConfigFactory.copy(openapi_config=openapi_config, api_key=config.api_key, host=config.host)
        elif openapi_config is None:
            openapi_config = OpenApiConfigFactory.build(api_key=config.api_key, host=config.host)

        # Check if value passed before overriding any values present
        # in the openapi_config. This means if the user has passed 
        # an openapi_config object and a kwarg for the same setting,
        # the kwarg will take precedence.
        if (config.proxy_url):
            openapi_config.proxy = config.proxy_url
        if (config.proxy_headers):
            openapi_config.proxy_headers = config.proxy_headers
        if (config.ssl_ca_certs):
            openapi_config.ssl_ca_cert = config.ssl_ca_certs
        if (config.ssl_verify != None):
            openapi_config.verify_ssl = config.ssl_verify
        
        return openapi_config