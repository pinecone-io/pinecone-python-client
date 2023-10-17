import logging
import sys
from typing import NamedTuple, List
import os

import certifi
import requests
import configparser
import socket

from urllib3.connection import HTTPConnection

from pinecone.core.client.exceptions import ApiKeyError
from pinecone.core.api_action import ActionAPI, WhoAmIResponse
from pinecone.core.utils import warn_deprecated, check_kwargs
from pinecone.core.utils.constants import (
    CLIENT_VERSION,
    PARENT_LOGGER_NAME,
    DEFAULT_PARENT_LOGGER_LEVEL,
    TCP_KEEPIDLE,
    TCP_KEEPINTVL,
    TCP_KEEPCNT,
)
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

__all__ = ["Config", "init"]

_logger = logging.getLogger(__name__)
_parent_logger = logging.getLogger(PARENT_LOGGER_NAME)
_parent_logger.setLevel(DEFAULT_PARENT_LOGGER_LEVEL)


class ConfigBase(NamedTuple):
    environment: str = ""
    api_key: str = ""
    project_name: str = ""
    controller_host: str = ""
    openapi_config: OpenApiConfiguration = None


class _CONFIG:
    """

    Order of configs to load:

    - configs specified explicitly in reset
    - environment variables
    - configs specified in the INI file
    - default configs
    """

    def __init__(self):
        self.reset()

    def validate(self):
        if not self._config.api_key:  # None or empty string invalid
            raise ApiKeyError("You haven't specified an Api-Key.")

    def reset(self, config_file=None, **kwargs):
        config = ConfigBase()

        # Load config from file
        file_config = self._load_config_file(config_file)

        # Get the environment first. Make sure that it is not overwritten in subsequent config objects.
        environment = (
            kwargs.pop("environment", None)
            or os.getenv("PINECONE_ENVIRONMENT")
            or file_config.pop("environment", None)
            or "us-west1-gcp"
        )
        config = config._replace(environment=environment)

        # Set default config
        default_config = ConfigBase(
            controller_host="https://controller.{0}.pinecone.io".format(config.environment),
        )
        config = config._replace(**self._preprocess_and_validate_config(default_config._asdict()))

        # Set INI file config
        config = config._replace(**self._preprocess_and_validate_config(file_config))

        # Set environment config
        env_config = ConfigBase(
            project_name=os.getenv("PINECONE_PROJECT_NAME"),
            api_key=os.getenv("PINECONE_API_KEY"),
            controller_host=os.getenv("PINECONE_CONTROLLER_HOST"),
        )
        config = config._replace(**self._preprocess_and_validate_config(env_config._asdict()))

        # Set explicit config
        config = config._replace(**self._preprocess_and_validate_config(kwargs))

        self._config = config

        # load project_name etc. from whoami api
        action_api = ActionAPI(host=config.controller_host, api_key=config.api_key)
        try:
            whoami_response = action_api.whoami()
        except requests.exceptions.RequestException:
            # proceed with default values; reset() may be called later w/ correct values
            whoami_response = WhoAmIResponse()

        if not self._config.project_name:
            config = config._replace(
                **self._preprocess_and_validate_config({"project_name": whoami_response.projectname})
            )

        self._config = config

        # Set OpenAPI client config
        default_openapi_config = OpenApiConfiguration.get_default_copy()
        default_openapi_config.ssl_ca_cert = certifi.where()
        openapi_config = kwargs.pop("openapi_config", None) or default_openapi_config

        openapi_config.socket_options = self._get_socket_options()

        config = config._replace(openapi_config=openapi_config)
        self._config = config

    def _preprocess_and_validate_config(self, config: dict) -> dict:
        """Normalize, filter, and validate config keys/values.

        Trims whitespace, removes invalid keys (and the "environment" key),
        and raises ValueError in case an invalid value was specified.
        """
        # general preprocessing and filtering
        result = {k: v for k, v in config.items() if k in ConfigBase._fields if v is not None}
        result.pop("environment", None)
        # validate api key
        api_key = result.get("api_key")
        # if api_key:
        #     try:
        #         uuid.UUID(api_key)
        #     except ValueError as e:
        #         raise ValueError(f"Pinecone API key \"{api_key}\" appears invalid. "
        #                          f"Did you specify it correctly?") from e
        return result

    def _load_config_file(self, config_file: str) -> dict:
        """Load from INI config file."""
        config_obj = {}
        if config_file:
            full_path = os.path.expanduser(config_file)
            if os.path.isfile(full_path):
                parser = configparser.ConfigParser()
                parser.read(full_path)
                if "default" in parser.sections():
                    config_obj = {**parser["default"]}
        return config_obj

    @staticmethod
    def _get_socket_options(
        do_keep_alive: bool = True,
        keep_alive_idle_sec: int = TCP_KEEPIDLE,
        keep_alive_interval_sec: int = TCP_KEEPINTVL,
        keep_alive_tries: int = TCP_KEEPCNT,
    ) -> List[tuple]:
        """
        Returns the socket options to pass to OpenAPI's Rest client
        Args:
            do_keep_alive: Whether to enable TCP keep alive mechanism
            keep_alive_idle_sec: Time in seconds of connection idleness before starting to send keep alive probes
            keep_alive_interval_sec: Interval time in seconds between keep alive probe messages
            keep_alive_tries: Number of failed keep alive tries (unanswered KA messages) before terminating the connection

        Returns:
            A list of socket options for the Rest client's connection pool
        """
        # Source: https://www.finbourne.com/blog/the-mysterious-hanging-client-tcp-keep-alives

        socket_params = HTTPConnection.default_socket_options
        if not do_keep_alive:
            return socket_params

        socket_params += [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]

        # TCP Keep Alive Probes for different platforms
        platform = sys.platform
        # TCP Keep Alive Probes for Linux
        if (
            platform == "linux"
            and hasattr(socket, "TCP_KEEPIDLE")
            and hasattr(socket, "TCP_KEEPINTVL")
            and hasattr(socket, "TCP_KEEPCNT")
        ):
            socket_params += [(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, keep_alive_idle_sec)]
            socket_params += [(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, keep_alive_interval_sec)]
            socket_params += [(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, keep_alive_tries)]

        # TCP Keep Alive Probes for Windows OS
        # NOTE: Changing TCP KA params on windows is done via a different mechanism which OpenAPI's Rest client doesn't expose.
        # Since the default values work well, it seems setting `(socket.SO_KEEPALIVE, 1)` is sufficient.
        # Leaving this code here for future reference.
        # elif platform == 'win32' and hasattr(socket, "SIO_KEEPALIVE_VALS"):
        #     socket.ioctl((socket.SIO_KEEPALIVE_VALS, (1, keep_alive_idle_sec * 1000, keep_alive_interval_sec * 1000)))

        # TCP Keep Alive Probes for Mac OS
        elif platform == "darwin":
            TCP_KEEPALIVE = 0x10
            socket_params += [(socket.IPPROTO_TCP, TCP_KEEPALIVE, keep_alive_interval_sec)]

        return socket_params

    @property
    def ENVIRONMENT(self):
        return self._config.environment

    @property
    def API_KEY(self):
        return self._config.api_key

    @property
    def PROJECT_NAME(self):
        return self._config.project_name

    @property
    def CONTROLLER_HOST(self):
        return self._config.controller_host

    @property
    def OPENAPI_CONFIG(self):
        return self._config.openapi_config

    @property
    def LOG_LEVEL(self):
        """
        Deprecated since v2.0.2 [Will be removed in v3.0.0]; use the standard logging module logger "pinecone" instead.
        """
        warn_deprecated(
            description='LOG_LEVEL is deprecated. Use the standard logging module logger "pinecone" instead.',
            deprecated_in="2.0.2",
            removal_in="3.0.0",
        )
        return logging.getLevelName(logging.getLogger("pinecone").level)


def init(
    api_key: str = None,
    host: str = None,
    environment: str = None,
    project_name: str = None,
    log_level: str = None,
    openapi_config: OpenApiConfiguration = None,
    config: str = "./.pinecone",
    **kwargs
):
    """Initializes configuration for the Pinecone client.

    The `pinecone` module is the main entrypoint to this sdk. You will use instances of it to create and manage indexes as well as 
    perform data operations on those indexes after they are created.

    **Initializing the client**

    There are two pieces of configuration required to use the Pinecone client: an API key and environment value. These values can
    be passed using environment variables, an INI configuration file, or explicitly as arguments to the ``init`` function. Find
    your configuration values in the console dashboard at [https://app.pinecone.io](https://app.pinecone.io).
    
    **Using environment variables**

    The environment variables used to configure the client are the following:
    
    ```python
    export PINECONE_API_KEY="your_api_key"
    export PINECONE_ENVIRONMENT="your_environment"
    export PINECONE_PROJECT_NAME="your_project_name"
    export PINECONE_CONTROLLER_HOST="your_controller_host"
    ```

    **Using an INI configuration file**

    You can use an INI configuration file to configure the client. The default location for this file is `./.pinecone`.
    You must place configuration values in the `default` group, and the keys must have the following format:

    ```python
    [default]
    api_key=your_api_key
    environment=your_environment
    project_name=your_project_name
    controller_host=your_controller_host
    ```

    When environment variables or a config file are provided, you do not need to initialize the client explicitly:

    ```python
    import pinecone
    pinecone.list_indexes()
    ```

    *Passing configuration values*

    If you prefer to pass configuration in code, the constructor accepts the following arguments. This could be useful if
    your application needs to interact with multiple projects, each with a different configuration. Explicitly passed values
    will override any existing environment or configuration file values.

    ```python
    pinecone.init(api_key="my-api-key", environment="my-environment")
    ```
    
    Args:
        api_key (str, optional): The API key for your Pinecone project. Required if not set in environment variables or the config file. 
            You can find this in the [Pinecone console](https://app.pinecone.io).
        host (str, optional): Custom controller host which will be used for API calls involving index operations.
        environment (str, optional): The environment for your Pinecone project. Required if not set in environment variables or the config file.
            You can find this in the [Pinecone console](https://app.pinecone.io).
        project_name (str, optional): The Pinecone project name. Overrides the value that is otherwise looked up and used from the Pinecone backend.
        openapi_config (`pinecone.core.client.configuration.Configuration`, optional): Sets a custom OpenAPI client configuration.
        config (str, optional): The path to an INI configuration file. Defaults to `./.pinecone`.
        log_level (str, optional): Deprecated since v2.0.2 [Will be removed in v3.0.0]; use the standard logging module to manage logger "pinecone" instead.
    """
    check_kwargs(init, kwargs)
    Config.reset(
        project_name=project_name,
        api_key=api_key,
        controller_host=host,
        environment=environment,
        openapi_config=openapi_config,
        config_file=config,
        **kwargs
    )
    if log_level:
        warn_deprecated(
            description='log_level is deprecated. Use the standard logging module to manage logger "pinecone" instead.',
            deprecated_in="2.0.2",
            removal_in="3.0.0",
        )


Config = _CONFIG()

# Init
init()
