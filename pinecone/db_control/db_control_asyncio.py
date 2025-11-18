from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
from pinecone.openapi_support import AsyncioApiClient

from pinecone.utils import setup_async_openapi_client
from pinecone.core.openapi.db_control import API_VERSION

logger = logging.getLogger(__name__)
""" :meta private: """


if TYPE_CHECKING:
    from .resources.asyncio.index import IndexResourceAsyncio
    from .resources.asyncio.collection import CollectionResourceAsyncio
    from .resources.asyncio.restore_job import RestoreJobResourceAsyncio
    from .resources.asyncio.backup import BackupResourceAsyncio
    from pinecone.config import Config, OpenApiConfiguration


class DBControlAsyncio:
    def __init__(self, config: "Config", openapi_config: "OpenApiConfiguration") -> None:
        self._config = config
        """ :meta private: """

        self._openapi_config = openapi_config
        """ :meta private: """

        self._index_api: AsyncioManageIndexesApi = setup_async_openapi_client(
            api_client_klass=AsyncioApiClient,
            api_klass=AsyncioManageIndexesApi,
            config=self._config,
            openapi_config=self._openapi_config,
            api_version=API_VERSION,
        )
        """ :meta private: """

        self._index_resource: "IndexResourceAsyncio" | None = None
        """ :meta private: """

        self._collection_resource: "CollectionResourceAsyncio" | None = None
        """ :meta private: """

        self._restore_job_resource: "RestoreJobResourceAsyncio" | None = None
        """ :meta private: """

        self._backup_resource: "BackupResourceAsyncio" | None = None
        """ :meta private: """

    @property
    def index(self) -> "IndexResourceAsyncio":
        if self._index_resource is None:
            from .resources.asyncio.index import IndexResourceAsyncio

            self._index_resource = IndexResourceAsyncio(
                index_api=self._index_api, config=self._config
            )
        return self._index_resource

    @property
    def collection(self) -> "CollectionResourceAsyncio":
        if self._collection_resource is None:
            from .resources.asyncio.collection import CollectionResourceAsyncio

            self._collection_resource = CollectionResourceAsyncio(self._index_api)
        return self._collection_resource

    @property
    def restore_job(self) -> "RestoreJobResourceAsyncio":
        if self._restore_job_resource is None:
            from .resources.asyncio.restore_job import RestoreJobResourceAsyncio

            self._restore_job_resource = RestoreJobResourceAsyncio(self._index_api)
        return self._restore_job_resource

    @property
    def backup(self) -> "BackupResourceAsyncio":
        if self._backup_resource is None:
            from .resources.asyncio.backup import BackupResourceAsyncio

            self._backup_resource = BackupResourceAsyncio(self._index_api)
        return self._backup_resource
