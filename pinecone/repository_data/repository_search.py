import warnings
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

from pinecone.config import ConfigBuilder

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.repository_data.api.document_operations_api import DocumentOperationsApi
from pinecone.core.openapi.repository_data import API_VERSION
from pinecone.core.openapi.repository_data.models import SearchDocumentsResponse
from .request_factory import RepositoryRequestFactory
from ..utils import setup_openapi_client, validate_and_convert_errors, filter_dict
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS

from multiprocessing import cpu_count


if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration


logger = logging.getLogger(__name__)
""" :meta private: """


class RepositorySearch:
    """
    A client for interacting with a Pinecone Repository search endpoint.
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

        self._repo_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=DocumentOperationsApi,
            config=self._config,
            openapi_config=self._openapi_config,
            pool_threads=self._pool_threads,
            api_version=API_VERSION,
        )

        self._api_client = self._repo_api.api_client

    @property
    def config(self) -> "Config":
        """:meta private:"""
        return self._config

    @property
    def openapi_config(self) -> "OpenApiConfiguration":
        """:meta private:"""
        warnings.warn(
            "The `openapi_config` property has been renamed to `_openapi_config`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._openapi_config

    @property
    def pool_threads(self) -> int:
        """:meta private:"""
        warnings.warn(
            "The `pool_threads` property has been renamed to `_pool_threads`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pool_threads

    def _openapi_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return filter_dict(kwargs, OPENAPI_ENDPOINT_PARAMS)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._repo_api.api_client.close()

    def close(self):
        self._repo_api.api_client.close()

    @validate_and_convert_errors
    def search(
        self,
        namespace: str,
        query_str: str,
        top_k: Optional[int] = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchDocumentsResponse:
        if namespace is None:
            raise Exception("Namespace is required when searching documents")

        request = RepositoryRequestFactory.search_request(
            query_str=query_str, top_k=top_k, filter=filter
        )

        return self._repo_api.search_documents(namespace, request)
