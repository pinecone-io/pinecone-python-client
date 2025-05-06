import logging
from typing import Optional, TYPE_CHECKING

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
from pinecone.openapi_support.api_client import ApiClient

from pinecone.utils import setup_openapi_client
from pinecone.core.openapi.db_control import API_VERSION


logger = logging.getLogger(__name__)
""" @private """

if TYPE_CHECKING:
    from .resources.sync.index import IndexResource
    from .resources.sync.collection import CollectionResource


class DBControl:
    def __init__(self, config, openapi_config, pool_threads):
        self.config = config
        """ @private """

        self.openapi_config = openapi_config
        """ @private """

        self.pool_threads = pool_threads
        """ @private """

        self._index_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=ManageIndexesApi,
            config=self.config,
            openapi_config=self.openapi_config,
            pool_threads=pool_threads,
            api_version=API_VERSION,
        )
        """ @private """

        self._index_resource: Optional["IndexResource"] = None
        """ @private """

        self._collection_resource: Optional["CollectionResource"] = None
        """ @private """

    @property
    def index(self) -> "IndexResource":
        if self._index_resource is None:
            from .resources.sync.index import IndexResource

            self._index_resource = IndexResource(index_api=self._index_api, config=self.config)
        return self._index_resource

    @property
    def collection(self) -> "CollectionResource":
        if self._collection_resource is None:
            from .resources.sync.collection import CollectionResource

            self._collection_resource = CollectionResource(self._index_api)
        return self._collection_resource
