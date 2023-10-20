from typing import NamedTuple
import os

from pinecone.utils import check_kwargs
from pinecone.exceptions import PineconeConfigurationError
from pinecone.core.client.exceptions import ApiKeyError
from pinecone.config.openapi import OpenApiConfigFactory
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration


class ConfigBase(NamedTuple):
    api_key: str = ""
    host: str = ""
    openapi_config: OpenApiConfiguration = None


class Config:
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

    def __init__(
        self,
        api_key: str = None,
        host: str = None,
        openapi_config: OpenApiConfiguration = None,
        **kwargs,
    ):
        api_key = api_key or kwargs.pop("api_key", None) or os.getenv("PINECONE_API_KEY")
        host = host or kwargs.pop("host", None)
        openapi_config = (
            openapi_config
            or kwargs.pop("openapi_config", None)
            or OpenApiConfigFactory.build(api_key=api_key, host=host)
        )

        check_kwargs(self.__init__, kwargs)
        self._config = ConfigBase(api_key, host, openapi_config)
        self.validate()

    def validate(self):
        if not self._config.api_key:
            raise PineconeConfigurationError("You haven't specified an Api-Key.")
        if not self._config.host:
            raise PineconeConfigurationError("You haven't specified a host.")

    @property
    def API_KEY(self):
        return self._config.api_key

    @property
    def HOST(self):
        return self._config.host

    @property
    def OPENAPI_CONFIG(self):
        return self._config.openapi_config
