from .setup_openapi_client import build_plugin_setup_client
from pinecone_plugin_interface import load_and_install as install_plugins
import logging

logger = logging.getLogger(__name__)


class PluginAware:
    def load_plugins(self):
        """@private"""
        try:
            # I don't expect this to ever throw, but wrapping this in a
            # try block just in case to make sure a bad plugin doesn't
            # halt client initialization.
            openapi_client_builder = build_plugin_setup_client(
                config=self.config,
                openapi_config=self.openapi_config,
                pool_threads=self.pool_threads,
            )
            install_plugins(self, openapi_client_builder)
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")
