import logging
from typing import Optional, TYPE_CHECKING

from pinecone.core.openapi.repository_control.api.manage_repositories_api import (
    ManageRepositoriesApi,
)
from pinecone.openapi_support.api_client import ApiClient

from pinecone.utils import setup_openapi_client, PluginAware
from pinecone.core.openapi.repository_control import API_VERSION


logger = logging.getLogger(__name__)
""" :meta private: """

if TYPE_CHECKING:
    from .resources.sync.repository import RepositoryResource
    from pinecone.config import Config, OpenApiConfiguration


class RepositoryControl(PluginAware):
    def __init__(
        self, config: "Config", openapi_config: "OpenApiConfiguration", pool_threads: int
    ) -> None:
        self.config = config
        """ :meta private: """

        self._openapi_config = openapi_config
        """ :meta private: """

        self._pool_threads = pool_threads
        """ :meta private: """

        self._repository_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=ManageRepositoriesApi,
            config=self.config,
            openapi_config=self._openapi_config,
            pool_threads=self._pool_threads,
            api_version=API_VERSION,
        )
        """ :meta private: """

        self._repository_resource: Optional["RepositoryResource"] = None
        """ :meta private: """

        super().__init__()  # Initialize PluginAware

    @property
    def repository(self) -> "RepositoryResource":
        if self._repository_resource is None:
            from .resources.sync.repository import RepositoryResource

            self._repository_resource = RepositoryResource(
                repository_api=self._repository_api,
                config=self.config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._repository_resource
