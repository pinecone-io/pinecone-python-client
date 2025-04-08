import logging
from typing import Optional, TYPE_CHECKING

from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
from pinecone.openapi_support import AsyncioApiClient

from pinecone.utils import setup_async_openapi_client
from pinecone.core.openapi.db_control import API_VERSION

logger = logging.getLogger(__name__)
""" @private """


if TYPE_CHECKING:
    from .resources_asyncio.index import IndexResourceAsyncio
    from .resources_asyncio.collection import CollectionResourceAsyncio


class DBControlAsyncio:
    def __init__(self, config, openapi_config, pool_threads):
        self.config = config
        """ @private """

        self.index_api = setup_async_openapi_client(
            api_client_klass=AsyncioApiClient,
            api_klass=AsyncioManageIndexesApi,
            config=self.config,
            openapi_config=self.openapi_config,
            api_version=API_VERSION,
        )
        """ @private """

        self._index_resource: Optional["IndexResourceAsyncio"] = None
        """ @private """

        self._collection_resource: Optional["CollectionResourceAsyncio"] = None
        """ @private """

    @property
    def index(self) -> "IndexResourceAsyncio":
        if self._index_resource is None:
            from .resources_asyncio.index import IndexResourceAsyncio

            self._index_resource = IndexResourceAsyncio(
                index_api=self.index_api, config=self.config
            )
        return self._index_resource

    @property
    def collection(self) -> "CollectionResourceAsyncio":
        if self._collection_resource is None:
            from .resources_asyncio.collection import CollectionResourceAsyncio

            self._collection_resource = CollectionResourceAsyncio(self.index_api)
        return self._collection_resource
