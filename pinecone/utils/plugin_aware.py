from typing import Any
from .setup_openapi_client import build_plugin_setup_client
from pinecone.config import Config
from pinecone.openapi_support.configuration import Configuration as OpenApiConfig

from pinecone_plugin_interface import load_and_install as install_plugins
import logging

logger = logging.getLogger(__name__)
""" @private """


class PluginAware:
    """
    Base class for classes that support plugin loading.

    This class provides functionality to lazily load plugins when they are first accessed.
    Subclasses must set the following attributes before calling super().__init__():
    - config: Config
    - openapi_config: OpenApiConfig
    - pool_threads: int
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the PluginAware class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            AttributeError: If required attributes are not set in the subclass.
        """
        logger.debug("PluginAware __init__ called for %s", self.__class__.__name__)

        # Check for required attributes after super().__init__ has been called
        missing_attrs = []
        if not hasattr(self, "config"):
            missing_attrs.append("config")
        if not hasattr(self, "openapi_config"):
            missing_attrs.append("openapi_config")
        if not hasattr(self, "pool_threads"):
            missing_attrs.append("pool_threads")

        if missing_attrs:
            raise AttributeError(
                f"PluginAware class requires the following attributes: {', '.join(missing_attrs)}. "
                f"These must be set in the {self.__class__.__name__} class's __init__ method "
                f"before calling super().__init__()."
            )

        self._plugins_loaded = False
        """ @private """

    def __getattr__(self, name: str) -> Any:
        """
        Called when an attribute is not found through the normal lookup process.
        This allows for lazy loading of plugins when they are first accessed.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The requested attribute.

        Raises:
            AttributeError: If the attribute cannot be found after loading plugins.
        """
        if not self._plugins_loaded:
            logger.debug("Loading plugins for %s", self.__class__.__name__)
            self.load_plugins(
                config=self.config,
                openapi_config=self.openapi_config,
                pool_threads=self.pool_threads,
            )
            self._plugins_loaded = True
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def load_plugins(
        self, config: Config, openapi_config: OpenApiConfig, pool_threads: int
    ) -> None:
        """
        Load plugins for the parent class.

        Args:
            config: The Pinecone configuration.
            openapi_config: The OpenAPI configuration.
            pool_threads: The number of threads in the pool.
        """
        try:
            # Build the OpenAPI client for plugin setup
            openapi_client_builder = build_plugin_setup_client(
                config=config, openapi_config=openapi_config, pool_threads=pool_threads
            )
            # Install plugins
            install_plugins(self, openapi_client_builder)
            logger.debug("Plugins loaded successfully for %s", self.__class__.__name__)
        except ImportError as e:
            logger.warning("Failed to import plugin module: %s", e)
        except Exception as e:
            logger.error("Error loading plugins: %s", e, exc_info=True)
