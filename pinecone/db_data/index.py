from pinecone.utils.tqdm import tqdm
import warnings
import logging
import json
from typing import Union, List, Optional, Dict, Any, Literal, Iterator, TYPE_CHECKING

from pinecone.config import ConfigBuilder

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.db_data.api.vector_operations_api import VectorOperationsApi
from pinecone.core.openapi.db_data import API_VERSION
from pinecone.core.openapi.db_data.models import (
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertResponse,
    ListResponse,
    SearchRecordsResponse,
    ListNamespacesResponse,
    NamespaceDescription,
)
from .dataclasses import (
    Vector,
    SparseValues,
    FetchResponse,
    FetchByMetadataResponse,
    Pagination,
    SearchQuery,
    SearchRerank,
)
from .interfaces import IndexInterface
from .request_factory import IndexRequestFactory
from .types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    FilterTypedDict,
    SearchRerankTypedDict,
    SearchQueryTypedDict,
)
from ..utils import (
    setup_openapi_client,
    parse_non_empty_args,
    validate_and_convert_errors,
    filter_dict,
    PluginAware,
    require_kwargs,
)
from .query_results_aggregator import QueryResultsAggregator, QueryNamespacesResults
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS

from multiprocessing.pool import ApplyResult
from multiprocessing import cpu_count
from concurrent.futures import as_completed


if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from .resources.sync.bulk_import import BulkImportResource
    from .resources.sync.namespace import NamespaceResource
    from .resources.sync.vector import VectorResource
    from .resources.sync.record import RecordResource

    from pinecone.core.openapi.db_data.models import (
        StartImportResponse,
        ListImportsResponse,
        ImportModel,
    )

    from .resources.sync.bulk_import import ImportErrorMode

logger = logging.getLogger(__name__)
""" :meta private: """


def parse_query_response(response: QueryResponse):
    """:meta private:"""
    response._data_store.pop("results", None)
    return response


class Index(PluginAware, IndexInterface):
    """
    A client for interacting with a Pinecone index via REST API.
    For improved performance, use the Pinecone GRPC index client.
    """

    _bulk_import_resource: Optional["BulkImportResource"]
    """ :meta private: """

    _namespace_resource: Optional["NamespaceResource"]
    """ :meta private: """

    def __init__(
        self,
        api_key: str,
        host: str,
        pool_threads: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        openapi_config=None,
        **kwargs,
    ):
        if additional_headers is None:
            additional_headers = {}
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

        self._vector_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=VectorOperationsApi,
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

        self._vector_resource = None
        """ :meta private: """

        self._record_resource = None
        """ :meta private: """

        # Initialize PluginAware parent class
        super().__init__()

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

    @property
    def bulk_import(self) -> "BulkImportResource":
        """:meta private:"""
        if self._bulk_import_resource is None:
            from .resources.sync.bulk_import import BulkImportResource

            self._bulk_import_resource = BulkImportResource(api_client=self._api_client)
        return self._bulk_import_resource

    @property
    def namespace(self) -> "NamespaceResource":
        """:meta private:"""
        if self._namespace_resource is None:
            from .resources.sync.namespace import NamespaceResource

            self._namespace_resource = NamespaceResource(
                api_client=self._api_client,
                config=self._config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._namespace_resource

    @property
    def vector(self) -> "VectorResource":
        """:meta private:"""
        if self._vector_resource is None:
            from .resources.sync.vector import VectorResource

            self._vector_resource = VectorResource(
                vector_api=self._vector_api,
                api_client=self._api_client,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._vector_resource

    @property
    def record(self) -> "RecordResource":
        """:meta private:"""
        if self._record_resource is None:
            from .resources.sync.record import RecordResource

            self._record_resource = RecordResource(
                vector_api=self._vector_api,
                openapi_config=self._openapi_config,
            )
        return self._record_resource

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vector_api.api_client.close()

    def close(self):
        self._vector_api.api_client.close()

    @validate_and_convert_errors
    def upsert(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse:
        return self.vector.upsert(
            vectors=vectors,
            namespace=namespace,
            batch_size=batch_size,
            show_progress=show_progress,
            **kwargs,
        )

    @validate_and_convert_errors
    def upsert_from_dataframe(
        self, df, namespace: Optional[str] = None, batch_size: int = 500, show_progress: bool = True
    ) -> UpsertResponse:
        return self.vector.upsert_from_dataframe(
            df=df, namespace=namespace, batch_size=batch_size, show_progress=show_progress
        )

    @validate_and_convert_errors
    def upsert_records(self, namespace: str, records: List[Dict[str, Any]]):
        """Upsert records to a namespace.

        Upsert records to a namespace. A record is a dictionary that contains either an ``id`` or ``_id``
        field along with other fields that will be stored as metadata. The ``id`` or ``_id`` field is used
        as the unique identifier for the record. At least one field in the record should correspond to
        a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and upserts them into
        the specified namespace of the index.

        Args:
            namespace (str): The namespace of the index to upsert records to.
            records (List[Dict[str, Any]]): The records to upsert into the index.
                Each record should contain either an ``id`` or ``_id`` field.

        Examples:

        .. code-block:: python

            >>> from pinecone import Pinecone, CloudProvider, AwsRegion, EmbedModel, IndexEmbed
            >>> pc = Pinecone(api_key="<<PINECONE_API_KEY>>")
            >>> index_model = pc.create_index_for_model(
            ...     name="my-model-index",
            ...     cloud=CloudProvider.AWS,
            ...     region=AwsRegion.US_WEST_2,
            ...     embed=IndexEmbed(
            ...         model=EmbedModel.Multilingual_E5_Large,
            ...         field_map={"text": "my_text_field"}
            ...     )
            ... )
            >>> idx = pc.Index(host=index_model.host)
            >>> idx.upsert_records(
            ...     namespace="my-namespace",
            ...     records=[
            ...         {"_id": "test1", "my_text_field": "Apple is a popular fruit."},
            ...         {"_id": "test2", "my_text_field": "The tech company Apple is innovative."},
            ...     ],
            ... )
        """
        return self.record.upsert_records(namespace=namespace, records=records)

    @validate_and_convert_errors
    def search(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        return self.record.search(namespace=namespace, query=query, rerank=rerank, fields=fields)

    @validate_and_convert_errors
    def search_records(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        return self.record.search_records(namespace=namespace, query=query, rerank=rerank, fields=fields)

    @validate_and_convert_errors
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self.vector.delete(
            ids=ids, delete_all=delete_all, namespace=namespace, filter=filter, **kwargs
        )

    @validate_and_convert_errors
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> FetchResponse:
        return self.vector.fetch(ids=ids, namespace=namespace, **kwargs)

    @validate_and_convert_errors
    def fetch_by_metadata(
        self,
        filter: FilterTypedDict,
        namespace: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        **kwargs,
    ) -> FetchByMetadataResponse:
        """Fetch vectors by metadata filter.

        Look up and return vectors by metadata filter from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:

        .. code-block:: python

            >>> index.fetch_by_metadata(
            ...     filter={'genre': {'$in': ['comedy', 'drama']}, 'year': {'$eq': 2019}},
            ...     namespace='my_namespace',
            ...     limit=50
            ... )
            >>> index.fetch_by_metadata(
            ...     filter={'status': 'active'},
            ...     pagination_token='token123'
            ... )

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                Metadata filter expression to select vectors.
                See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`
            namespace (str): The namespace to fetch vectors from.
                            If not specified, the default namespace is used. [optional]
            limit (int): Max number of vectors to return. Defaults to 100. [optional]
            pagination_token (str): Pagination token to continue a previous listing operation. [optional]

        Returns:
            FetchByMetadataResponse: Object containing the fetched vectors, namespace, usage, and pagination token.
        """
        return self.vector.fetch_by_metadata(
            filter=filter,
            namespace=namespace,
            limit=limit,
            pagination_token=pagination_token,
            **kwargs,
        )

    @validate_and_convert_errors
    def query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Union[QueryResponse, ApplyResult]:
        return self.vector.query(
            *args,
            top_k=top_k,
            vector=vector,
            id=id,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
            **kwargs,
        )

    @validate_and_convert_errors
    def query_namespaces(
        self,
        vector: Optional[List[float]],
        namespaces: List[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, Dict[str, Union[List[float], List[int]]]]
        ] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        return self.vector.query_namespaces(
            vector=vector,
            namespaces=namespaces,
            metric=metric,
            top_k=top_k,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
            **kwargs,
        )

    @validate_and_convert_errors
    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self.vector.update(
            id=id,
            values=values,
            set_metadata=set_metadata,
            namespace=namespace,
            sparse_values=sparse_values,
            **kwargs,
        )

    @validate_and_convert_errors
    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        return self.vector.describe_index_stats(filter=filter, **kwargs)

    @validate_and_convert_errors
    def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> ListResponse:
        return self.vector.list_paginated(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            namespace=namespace,
            **kwargs,
        )

    @validate_and_convert_errors
    def list(self, **kwargs):
        return self.vector.list(**kwargs)

    @validate_and_convert_errors
    def start_import(
        self,
        uri: str,
        integration_id: Optional[str] = None,
        error_mode: Optional[
            Union["ImportErrorMode", Literal["CONTINUE", "ABORT"], str]
        ] = "CONTINUE",
    ) -> "StartImportResponse":
        """
        Args:
            uri (str): The URI of the data to import. The URI must start with the scheme of a supported storage provider.
            integration_id (Optional[str], optional): If your bucket requires authentication to access, you need to pass the id of your storage integration using this property. Defaults to None.
            error_mode: Defaults to "CONTINUE". If set to "CONTINUE", the import operation will continue even if some
                records fail to import. Pass "ABORT" to stop the import operation if any records fail to import.

        Returns:
            `StartImportResponse`: Contains the id of the import operation.

        Import data from a storage provider into an index. The uri must start with the scheme of a supported
        storage provider. For buckets that are not publicly readable, you will also need to separately configure
        a storage integration and pass the integration id.

        Examples:
            >>> from pinecone import Pinecone
            >>> index = Pinecone().Index('my-index')
            >>> index.start_import(uri="s3://bucket-name/path/to/data.parquet")
            { id: "1" }
        """
        return self.bulk_import.start(uri=uri, integration_id=integration_id, error_mode=error_mode)

    @validate_and_convert_errors
    def list_imports(self, **kwargs) -> Iterator["ImportModel"]:
        """
        Args:
            limit (Optional[int]): The maximum number of operations to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
                to fetch the next page of results. [optional]

        Returns:
            Returns a generator that yields each import operation. It automatically handles pagination tokens on your behalf so you can
            easily iterate over all results. The `list_imports` method accepts all of the same arguments as list_imports_paginated

        .. code-block:: python

            for op in index.list_imports():
                print(op)


        You can convert the generator into a list by wrapping the generator in a call to the built-in `list` function:

        .. code-block:: python

            operations = list(index.list_imports())

        You should be cautious with this approach because it will fetch all operations at once, which could be a large number
        of network calls and a lot of memory to hold the results.
        """
        for i in self.bulk_import.list(**kwargs):
            yield i

    @validate_and_convert_errors
    def list_imports_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> "ListImportsResponse":
        """
        Args:
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns: ListImportsResponse object which contains the list of operations as ImportModel objects, pagination information,
            and usage showing the number of read_units consumed.

        The list_imports_paginated() operation returns information about import operations.
        It returns operations in a paginated form, with a pagination token to fetch the next page of results.

        Consider using the `list_imports` method to avoid having to handle pagination tokens manually.

        Examples:

        .. code-block:: python

            >>> results = index.list_imports_paginated(limit=5)
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> results.data[0]
            {
                "id": "6",
                "uri": "s3://dev-bulk-import-datasets-pub/10-records-dim-10/",
                "status": "Completed",
                "percent_complete": 100.0,
                "records_imported": 10,
                "created_at": "2024-09-06T14:52:02.567776+00:00",
                "finished_at": "2024-09-06T14:52:28.130717+00:00"
            }
            >>> next_results = index.list_imports_paginated(limit=5, pagination_token=results.pagination.next)
        """
        return self.bulk_import.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )

    @validate_and_convert_errors
    def describe_import(self, id: str) -> "ImportModel":
        """
        Args:
            id (str): The id of the import operation. This value is returned when
                starting an import, and can be looked up using list_imports.

        Returns:
            `ImportModel`: An object containing operation id, status, and other details.

        describe_import is used to get detailed information about a specific import operation.
        """
        return self.bulk_import.describe(id=id)

    @validate_and_convert_errors
    def cancel_import(self, id: str):
        """Cancel an import operation.

        Args:
            id (str): The id of the import operation to cancel.
        """
        return self.bulk_import.cancel(id=id)

    @validate_and_convert_errors
    @require_kwargs
    def create_namespace(
        self, name: str, schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> "NamespaceDescription":
        return self.namespace.create(name=name, schema=schema, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def describe_namespace(self, namespace: str, **kwargs) -> "NamespaceDescription":
        return self.namespace.describe(namespace=namespace, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def delete_namespace(self, namespace: str, **kwargs) -> Dict[str, Any]:
        return self.namespace.delete(namespace=namespace, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def list_namespaces(
        self, limit: Optional[int] = None, **kwargs
    ) -> Iterator[ListNamespacesResponse]:
        return self.namespace.list(limit=limit, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def list_namespaces_paginated(
        self, limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> ListNamespacesResponse:
        return self.namespace.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )
