import logging
from typing import Union, Optional, Dict, Any

from pinecone.config import ConfigBuilder


from pinecone.core.openapi.ckb_knowledge_data.api.document_operations_api import (
    DocumentOperationsApi,
)
from pinecone.core.openapi.ckb_knowledge_data import API_VERSION

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.ckb_knowledge_data.models import (
    DocumentForUpsert,
    UpsertDocumentResponse,
)

from .interfaces import RepositoryInterface

from pinecone.utils import setup_openapi_client

from multiprocessing import cpu_count


logger = logging.getLogger(__name__)
""" :meta private: """


class Repository(RepositoryInterface):
    """
    A client for interacting with a Pinecone Repository API.
    """

    def __init__(
        self,
        api_key: str,
        host: str,
        pool_threads: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        openapi_config=None,
        **kwargs,
    ):
        self._config = ConfigBuilder.build(
            api_key=api_key, host=host, additional_headers=additional_headers, **kwargs
        )
        """ :meta private: """
        self._openapi_config = ConfigBuilder.build_openapi_config(self._config, openapi_config)
        """ :meta private: """

        if pool_threads is None:
            self._pool_threads = 5 * cpu_count()
            """ :meta private: """
        else:
            self._pool_threads = pool_threads
            """ :meta private: """

        if kwargs.get("connection_pool_maxsize", None):
            self._openapi_config.connection_pool_maxsize = kwargs.get("connection_pool_maxsize")

        self._repository_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=DocumentOperationsApi,
            config=self._config,
            openapi_config=self._openapi_config,
            pool_threads=self._pool_threads,
            api_version=API_VERSION,
        )

        self._api_client = self._vector_api.api_client

        self._bulk_import_resource = None
        """ :meta private: """

        self._namespace_resource = None
        """ :meta private: """

        # Pass the same api_client to the ImportFeatureMixin
        super().__init__(api_client=self._api_client)

    def upsert(
        self, namespace: str, document: Union[Dict[str, Any], DocumentForUpsert], **kwargs
    ) -> UpsertDocumentResponse:
        self._repository_api.upsert_document(namespace=namespace, document=document, **kwargs)
