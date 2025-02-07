from .setup_openapi_client import build_plugin_setup_client
from pinecone.config import Config
from pinecone.openapi_support.configuration import Configuration as OpenApiConfig


from pinecone_plugin_interface import load_and_install as install_plugins
import logging

logger = logging.getLogger(__name__)
""" @private """


class PluginAware:
    def load_plugins(
        self, config: Config, openapi_config: OpenApiConfig, pool_threads: int
    ) -> None:
        """@private"""
        try:
            # I don't expect this to ever throw, but wrapping this in a
            # try block just in case to make sure a bad plugin doesn't
            # halt client initialization.
            openapi_client_builder = build_plugin_setup_client(
                config=config, openapi_config=openapi_config, pool_threads=pool_threads
            )
            install_plugins(self, openapi_client_builder)
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")
