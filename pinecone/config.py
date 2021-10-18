#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#
import logging
from typing import NamedTuple
import os

import certifi
import requests
import sentry_sdk
import configparser

from pinecone.core.client.exceptions import ApiKeyError
from pinecone.core.api_action import ActionAPI, WhoAmIResponse
from pinecone.core.utils import warn_deprecated
from pinecone.core.utils.constants import CLIENT_VERSION, PARENT_LOGGER_NAME, DEFAULT_PARENT_LOGGER_LEVEL
from pinecone.core.utils.sentry import sentry_decorator as sentry
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

__all__ = [
    "Config", "init"
]

_logger = logging.getLogger(__name__)
_parent_logger = logging.getLogger(PARENT_LOGGER_NAME)
_parent_logger.setLevel(DEFAULT_PARENT_LOGGER_LEVEL)


def _set_sentry_tags(config: dict):
    sentry_sdk.set_tag("package_version", CLIENT_VERSION)
    sentry_tag_names = ('environment', 'project_name', 'controller_host', 'username', 'user_label')
    for key, val in config.items():
        if key in sentry_tag_names:
            sentry_sdk.set_tag(key, val)


class ConfigBase(NamedTuple):
    environment: str = ""
    api_key: str = ""
    project_name: str = ""
    controller_host: str = ""
    openapi_config: OpenApiConfiguration = None


class _CONFIG:
    """

    Order of configs to load:

    - configs specified explictly in reset
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
                **self._preprocess_and_validate_config({'project_name': whoami_response.projectname}))

        self._config = config

        # Set OpenAPI client config
        default_openapi_config = OpenApiConfiguration.get_default_copy()
        default_openapi_config.ssl_ca_cert = certifi.where()
        openapi_config = (
                kwargs.pop("openapi_config", None)
                or default_openapi_config
        )

        config = config._replace(openapi_config=openapi_config)
        self._config = config

        # Sentry
        _set_sentry_tags({**whoami_response._asdict(), **self._config._asdict()})

    def _preprocess_and_validate_config(self, config: dict) -> dict:
        """Normalize, filter, and validate config keys/values.

        Trims whitespace, removes invalid keys (and the "environment" key),
        and raises ValueError in case an invalid value was specified.
        """
        # general preprocessing and filtering
        result = {k: v for k, v in config.items() if k in ConfigBase._fields if v is not None}
        result.pop('environment', None)
        # validate api key
        api_key = result.get('api_key')
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
            deprecated_in='2.0.2',
            removal_in='3.0.0'
        )
        return logging.getLevelName(logging.getLogger('pinecone').level)


@sentry
def init(api_key: str = None, host: str = None, environment: str = None, project_name: str = None,
         log_level: str = None, openapi_config: OpenApiConfiguration = None,
         config: str = "~/.pinecone", **kwargs):
    """Initializes the Pinecone client.

    :param api_key: Required if not set in config file or by environment variable ``PINECONE_API_KEY``.
    :param host: Optional. Controller host.
    :param environment: Optional. Deployment environment.
    :param project_name: Optional. Pinecone project name. Overrides the value that is otherwise looked up and used from the Pinecone backend.
    :param openapi_config: Optional. Set OpenAPI client configuration.
    :param config: Optional. An INI configuration file.
    :param log_level: Deprecated since v2.0.2 [Will be removed in v3.0.0]; use the standard logging module to manage logger "pinecone" instead.
    """
    Config.reset(project_name=project_name, api_key=api_key, controller_host=host, environment=environment,
                 openapi_config=openapi_config, config_file=config, **kwargs)
    if log_level:
        warn_deprecated(
            description='log_level is deprecated. Use the standard logging module to manage logger "pinecone" instead.',
            deprecated_in='2.0.2',
            removal_in='3.0.0'
        )


Config = _CONFIG()

# Init
init()
