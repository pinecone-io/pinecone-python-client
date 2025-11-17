from __future__ import annotations

from pinecone.utils.tqdm import tqdm
import warnings
import logging
import json
from typing import Any, Literal, Iterator, TYPE_CHECKING

from pinecone.config import ConfigBuilder

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.db_data.api.vector_operations_api import VectorOperationsApi
from pinecone.core.openapi.db_data import API_VERSION
from pinecone.core.openapi.db_data.models import (
    QueryResponse as OpenAPIQueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
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
    QueryResponse,
    UpsertResponse,
    UpdateResponse,
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

    from pinecone.core.openapi.db_data.models import (
        StartImportResponse,
        ListImportsResponse,
        ImportModel,
    )

    from .resources.sync.bulk_import import ImportErrorMode

logger = logging.getLogger(__name__)
""" :meta private: """


def parse_query_response(response: OpenAPIQueryResponse) -> QueryResponse:
    """:meta private:"""
    # Convert OpenAPI QueryResponse to dataclass QueryResponse
    from pinecone.utils.response_info import extract_response_info

    response_info = None
    if hasattr(response, "_response_info"):
        response_info = response._response_info

    if response_info is None:
        response_info = extract_response_info({})

    # Remove deprecated 'results' field if present
    if hasattr(response, "_data_store"):
        response._data_store.pop("results", None)

    return QueryResponse(
        matches=response.matches,
        namespace=response.namespace or "",
        usage=response.usage if hasattr(response, "usage") and response.usage else None,
        _response_info=response_info,
    )


class UpsertResponseTransformer:
    """Transformer for converting ApplyResult[OpenAPIUpsertResponse] to UpsertResponse.

    This wrapper transforms the OpenAPI response to our dataclass when .get() is called,
    while delegating other methods to the underlying ApplyResult.
    """

    _apply_result: ApplyResult
    """ :meta private: """

    def __init__(self, apply_result: ApplyResult) -> None:
        self._apply_result = apply_result

    def get(self, timeout: float | None = None) -> UpsertResponse:
        openapi_response = self._apply_result.get(timeout)
        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if hasattr(openapi_response, "_response_info"):
            response_info = openapi_response._response_info
        if response_info is None:
            response_info = extract_response_info({})
        return UpsertResponse(
            upserted_count=openapi_response.upserted_count, _response_info=response_info
        )

    def __getattr__(self, name: str) -> Any:
        # Delegate other methods to the underlying ApplyResult
        return getattr(self._apply_result, name)


class Index(PluginAware, IndexInterface):
    """
    A client for interacting with a Pinecone index via REST API.
    For improved performance, use the Pinecone GRPC index client.
    """

    _config: "Config"
    """ :meta private: """

    _openapi_config: "OpenApiConfiguration"
    """ :meta private: """

    _pool_threads: int
    """ :meta private: """

    _vector_api: VectorOperationsApi
    """ :meta private: """

    _api_client: ApiClient
    """ :meta private: """

    _bulk_import_resource: "BulkImportResource" | None
    """ :meta private: """

    _namespace_resource: "NamespaceResource" | None
    """ :meta private: """

    def __init__(
        self,
        api_key: str,
        host: str,
        pool_threads: int | None = None,
        additional_headers: dict[str, str] | None = {},
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

        connection_pool_maxsize = kwargs.get("connection_pool_maxsize", None)
        if connection_pool_maxsize is not None:
            self._openapi_config.connection_pool_maxsize = connection_pool_maxsize

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

        # Pass the same api_client to the ImportFeatureMixin
        super().__init__(api_client=self._api_client)

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

    def _openapi_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return filter_dict(kwargs, OPENAPI_ENDPOINT_PARAMS)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vector_api.api_client.close()

    def close(self):
        self._vector_api.api_client.close()

    @validate_and_convert_errors
    def upsert(
        self,
        vectors: (
            list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict]
        ),
        namespace: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse | ApplyResult:
        """Upsert vectors into a namespace of your index.

        The upsert operation writes vectors into a namespace of your index.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        Args:
            vectors: A list of vectors to upsert. Can be a list of Vector objects, tuples, or dictionaries.
            namespace: The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size: The number of vectors to upsert in each batch.
                       If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress: Whether to show a progress bar using tqdm.
                          Applied only if batch_size is provided. Default is True.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            UpsertResponse: Includes the number of vectors upserted. If async_req=True, returns ApplyResult instead.

        **Upserting dense vectors**

        When working with dense vectors, the dimension of each vector must match the dimension configured for the
        index. A vector can be represented in a variety of ways.

        .. code-block:: python
            :caption: Upserting a dense vector using the Vector object

            from pinecone import Pinecone, Vector

            pc = Pinecone()
            idx = pc.Index(host="example-index-host")

            idx.upsert(
                namespace='my-namespace',
                vectors=[
                    Vector(
                        id='id1',
                        values=[0.1, 0.2, 0.3, 0.4],
                        metadata={'metadata_key': 'metadata_value'}
                    ),
                ]
            )

        .. code-block:: python
            :caption: Upserting a dense vector as a two-element tuple (no metadata)

            idx.upsert(
                namespace='my-namespace',
                vectors=[
                    ('id1', [0.1, 0.2, 0.3, 0.4]),
                ]
            )

        .. code-block:: python
            :caption: Upserting a dense vector as a three-element tuple with metadata

            idx.upsert(
                namespace='my-namespace',
                vectors=[
                    (
                        'id1',
                        [0.1, 0.2, 0.3, 0.4],
                        {'metadata_key': 'metadata_value'}
                    ),
                ]
            )

        .. code-block:: python
            :caption: Upserting a dense vector using a vector dictionary

            idx.upsert(
                namespace='my-namespace',
                vectors=[
                    {
                        "id": "id1",
                        "values": [0.1, 0.2, 0.3, 0.4],
                        "metadata": {"metadata_key": "metadata_value"}
                    },
                ]
            )

        **Upserting sparse vectors**

        .. code-block:: python
            :caption: Upserting a sparse vector

            from pinecone import (
                Pinecone,
                Vector,
                SparseValues,
            )

            pc = Pinecone()
            idx = pc.Index(host="example-index-host")

            idx.upsert(
                namespace='my-namespace',
                vectors=[
                    Vector(
                        id='id1',
                        sparse_values=SparseValues(
                            indices=[1, 2],
                            values=[0.2, 0.4]
                        )
                    ),
                ]
            )

        .. code-block:: python
            :caption: Upserting a sparse vector using a dictionary

            idx.upsert(
                namespace='my-namespace',
                vectors=[
                    {
                        "id": "id1",
                        "sparse_values": {
                            "indices": [1, 2],
                            "values": [0.2, 0.4]
                        }
                    },
                ]
            )

        **Batch upsert**

        If you have a large number of vectors, you can upsert them in batches.

        .. code-block:: python
            :caption: Upserting in batches

            from pinecone import Pinecone, Vector
            import random

            pc = Pinecone()
            idx = pc.Index(host="example-index-host")

            num_vectors = 100000
            vectors = [
                Vector(
                    id=f'id{i}',
                    values=[random.random() for _ in range(1536)])
                for i in range(num_vectors)
            ]

            idx.upsert(
                namespace='my-namespace',
                vectors=vectors,
                batch_size=50
            )

        **Visual progress bar with tqdm**

        To see a progress bar when upserting in batches, you will need to separately install `tqdm <https://tqdm.github.io/>`_.
        If ``tqdm`` is present, the client will detect and use it to display progress when ``show_progress=True``.

        To upsert in parallel, follow `this link <https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel>`_.

        """
        _check_type = kwargs.pop("_check_type", True)

        if kwargs.get("async_req", False) and batch_size is not None:
            raise ValueError(
                "async_req is not supported when batch_size is provided."
                "To upsert in parallel, please follow: "
                "https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel"
            )

        if batch_size is None:
            result = self._upsert_batch(vectors, namespace, _check_type, **kwargs)
            # If async_req=True, result is an ApplyResult[OpenAPIUpsertResponse]
            # We need to wrap it to convert to our dataclass when .get() is called
            if kwargs.get("async_req", False):
                # result is ApplyResult when async_req=True
                return UpsertResponseTransformer(result)  # type: ignore[arg-type, return-value]
            # result is UpsertResponse when async_req=False
            # _upsert_batch already returns UpsertResponse when async_req=False
            return result

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        pbar = tqdm(total=len(vectors), disable=not show_progress, desc="Upserted vectors")
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch_result = self._upsert_batch(
                vectors[i : i + batch_size], namespace, _check_type, **kwargs
            )
            # When batch_size is provided, async_req cannot be True (checked above),
            # so batch_result is always UpsertResponse, not ApplyResult
            assert isinstance(
                batch_result, UpsertResponse
            ), "batch_result must be UpsertResponse when batch_size is provided"
            pbar.update(batch_result.upserted_count)
            # we can't use here pbar.n for the case show_progress=False
            total_upserted += batch_result.upserted_count

        # _response_info may be attached if LSN headers were present in the last batch
        # Create dataclass UpsertResponse from the last batch result
        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if batch_result and hasattr(batch_result, "_response_info"):
            response_info = batch_result._response_info
        if response_info is None:
            response_info = extract_response_info({})

        return UpsertResponse(upserted_count=total_upserted, _response_info=response_info)

    def _upsert_batch(
        self,
        vectors: (
            list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict]
        ),
        namespace: str | None,
        _check_type: bool,
        **kwargs,
    ) -> UpsertResponse | ApplyResult:
        # Convert OpenAPI UpsertResponse to dataclass UpsertResponse
        result = self._vector_api.upsert_vectors(
            IndexRequestFactory.upsert_request(vectors, namespace, _check_type, **kwargs),
            **self._openapi_kwargs(kwargs),
        )

        # If async_req=True, result is an ApplyResult[OpenAPIUpsertResponse]
        # We need to wrap it in a transformer that converts to our dataclass
        if kwargs.get("async_req", False):
            # Return ApplyResult - it will be unwrapped by the caller
            # The ApplyResult contains OpenAPIUpsertResponse which will be converted when .get() is called
            return result  # type: ignore[no-any-return]  # ApplyResult is not tracked through OpenAPI layers

        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if hasattr(result, "_response_info"):
            response_info = result._response_info
        if response_info is None:
            response_info = extract_response_info({})

        return UpsertResponse(upserted_count=result.upserted_count, _response_info=response_info)

    @staticmethod
    def _iter_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size].to_dict(orient="records")
            yield batch

    @validate_and_convert_errors
    def upsert_from_dataframe(
        self, df, namespace: str | None = None, batch_size: int = 500, show_progress: bool = True
    ) -> UpsertResponse:
        """Upsert vectors from a pandas DataFrame into the index.

        Args:
            df: A pandas DataFrame with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into. If not specified, the default namespace is used. [optional]
            batch_size: The number of rows to upsert in a single batch. Defaults to 500.
            show_progress: Whether to show a progress bar. Defaults to True.

        Returns:
            UpsertResponse: Object containing the number of vectors upserted.

        Examples:

        .. code-block:: python

            import pandas as pd
            from pinecone import Pinecone

            pc = Pinecone()
            idx = pc.Index(host="example-index-host")

            # Create a DataFrame with vector data
            df = pd.DataFrame({
                'id': ['id1', 'id2', 'id3'],
                'values': [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9]
                ],
                'metadata': [
                    {'key1': 'value1'},
                    {'key2': 'value2'},
                    {'key3': 'value3'}
                ]
            })

            # Upsert from DataFrame
            response = idx.upsert_from_dataframe(
                df=df,
                namespace='my-namespace',
                batch_size=100,
                show_progress=True
            )

        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError(
                "The `pandas` package is not installed. Please install pandas to use `upsert_from_dataframe()`"
            )

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Only pandas dataframes are supported. Found: {type(df)}")

        pbar = tqdm(total=len(df), disable=not show_progress, desc="sending upsert requests")
        results = []
        for chunk in self._iter_dataframe(df, batch_size=batch_size):
            res = self.upsert(vectors=chunk, namespace=namespace)
            pbar.update(len(chunk))
            results.append(res)

        upserted_count = 0
        last_result = None
        for res in results:
            # upsert_from_dataframe doesn't use async_req, so res is always UpsertResponse
            assert isinstance(
                res, UpsertResponse
            ), "Expected UpsertResponse when not using async_req"
            upserted_count += res.upserted_count
            last_result = res

        # Create aggregated response with metadata from final batch
        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if last_result and hasattr(last_result, "_response_info"):
            response_info = last_result._response_info
        if response_info is None:
            response_info = extract_response_info({})

        return UpsertResponse(upserted_count=upserted_count, _response_info=response_info)

    def upsert_records(self, namespace: str, records: list[dict]) -> UpsertResponse:
        """Upsert records to a namespace.

        A record is a dictionary that contains either an ``id`` or ``_id``
        field along with other fields that will be stored as metadata. The ``id`` or ``_id`` field is used
        as the unique identifier for the record. At least one field in the record should correspond to
        a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and upserts them into
        the specified namespace of the index.

        Args:
            namespace: The namespace of the index to upsert records to.
            records: The records to upsert into the index. Each record should contain an ``id`` or ``_id``
                    field and fields that match the index's embed configuration field mappings.

        Returns:
            UpsertResponse: Object which contains the number of records upserted.

        Examples:

        .. code-block:: python
            :caption: Upserting records to be embedded with Pinecone's integrated inference models

            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel,
                IndexEmbed
            )

            pc = Pinecone(api_key="<<PINECONE_API_KEY>>")

            # Create an index configured for the multilingual-e5-large model
            index_model = pc.create_index_for_model(
                name="my-model-index",
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_WEST_2,
                embed=IndexEmbed(
                    model=EmbedModel.Multilingual_E5_Large,
                    field_map={"text": "my_text_field"}
                )
            )

            # Instantiate the index client
            idx = pc.Index(host=index_model.host)

            # Upsert records
            idx.upsert_records(
                namespace="my-namespace",
                records=[
                    {
                        "_id": "test1",
                        "my_text_field": "Apple is a popular fruit known for its sweetness and crisp texture.",
                    },
                    {
                        "_id": "test2",
                        "my_text_field": "The tech company Apple is known for its innovative products like the iPhone.",
                    },
                    {
                        "_id": "test3",
                        "my_text_field": "Many people enjoy eating apples as a healthy snack.",
                    },
                ],
            )

        """
        args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=records)
        # Use _return_http_data_only=False to get headers for LSN extraction
        result = self._vector_api.upsert_records_namespace(_return_http_data_only=False, **args)
        # result is a tuple: (data, status, headers) when _return_http_data_only=False
        response_info = None
        if isinstance(result, tuple) and len(result) >= 3:
            headers = result[2]
            if headers:
                from pinecone.utils.response_info import extract_response_info

                response_info = extract_response_info(headers)
                # response_info may contain raw_headers even without LSN values

        # Ensure response_info is always present
        if response_info is None:
            from pinecone.utils.response_info import extract_response_info

            response_info = extract_response_info({})

        # Count records (could be len(records) but we don't know if any failed)
        # For now, assume all succeeded
        return UpsertResponse(upserted_count=len(records), _response_info=response_info)

    @validate_and_convert_errors
    def search(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: SearchRerankTypedDict | SearchRerank | None = None,
        fields: list[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Search for records in a namespace.

        This operation converts a query to a vector embedding and then searches a namespace. You
        can optionally provide a reranking operation as part of the search.

        Args:
            namespace: The namespace in the index to search.
            query: The SearchQuery to use for the search. The query can include a ``match_terms`` field
                   to specify which terms must be present in the text of each search hit. The match_terms
                   should be a dict with ``strategy`` (str) and ``terms`` (list[str]) keys, e.g.
                   ``{"strategy": "all", "terms": ["term1", "term2"]}``. Currently only "all" strategy
                   is supported, which means all specified terms must be present.
                   **Note:** match_terms is only supported for sparse indexes with integrated embedding
                   configured to use the pinecone-sparse-english-v0 model.
            rerank: The SearchRerank to use with the search request. [optional]
            fields: List of fields to return in the response. Defaults to ["*"] to return all fields. [optional]

        Returns:
            SearchRecordsResponse: The records that match the search.

        Examples:

        .. code-block:: python

            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel,
                IndexEmbed,
                SearchQuery,
                SearchRerank,
                RerankModel
            )

            pc = Pinecone(api_key="<<PINECONE_API_KEY>>")

            # Create an index for your embedding model
            index_model = pc.create_index_for_model(
                name="my-model-index",
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_WEST_2,
                embed=IndexEmbed(
                    model=EmbedModel.Multilingual_E5_Large,
                    field_map={"text": "my_text_field"}
                )
            )

            # Instantiate the index client
            idx = pc.Index(host=index_model.host)

            # Search for similar records
            response = idx.search(
                namespace="my-namespace",
                query=SearchQuery(
                    inputs={
                        "text": "Apple corporation",
                    },
                    top_k=3,
                ),
                rerank=SearchRerank(
                    model=RerankModel.Bge_Reranker_V2_M3,
                    rank_fields=["my_text_field"],
                    top_n=3,
                ),
            )

        """
        if namespace is None:
            raise Exception("Namespace is required when searching records")

        request = IndexRequestFactory.search_request(query=query, rerank=rerank, fields=fields)

        from typing import cast

        result = self._vector_api.search_records_namespace(namespace, request)
        return cast(SearchRecordsResponse, result)

    @validate_and_convert_errors
    def search_records(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: SearchRerankTypedDict | SearchRerank | None = None,
        fields: list[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Alias of the search() method.

        See :meth:`search` for full documentation and examples.

        """
        return self.search(namespace, query=query, rerank=rerank, fields=fields)

    @validate_and_convert_errors
    def delete(
        self,
        ids: list[str] | None = None,
        delete_all: bool | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Delete vectors from the index, from a single namespace.

        The Delete operation deletes vectors from the index, from a single namespace.
        No error is raised if the vector id does not exist.

        Note: For any delete call, if namespace is not specified, the default namespace ``""`` is used.
        Since the delete operation does not error when ids are not present, this means you may not receive
        an error if you delete from the wrong namespace.

        Delete can occur in the following mutually exclusive ways:

        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete_all must be set to False)

        Args:
            ids: Vector ids to delete. [optional]
            delete_all: This indicates that all vectors in the index namespace should be deleted.
                       Default is False. [optional]
            namespace: The namespace to delete vectors from. If not specified, the default namespace is used. [optional]
            filter: If specified, the metadata filter here will be used to select the vectors to delete.
                   This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                   See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            dict[str, Any]: An empty dictionary if the delete operation was successful.

        Examples:

        .. code-block:: python

            >>> # Delete specific vectors by ID
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            {}

            >>> # Delete all vectors from a namespace
            >>> index.delete(delete_all=True, namespace='my_namespace')
            {}

            >>> # Delete vectors matching a metadata filter
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace')
            {}

        """
        from typing import cast

        result = self._vector_api.delete_vectors(
            IndexRequestFactory.delete_request(
                ids=ids, delete_all=delete_all, namespace=namespace, filter=filter, **kwargs
            ),
            **self._openapi_kwargs(kwargs),
        )
        return cast(dict[str, Any], result)

    @validate_and_convert_errors
    def fetch(self, ids: list[str], namespace: str | None = None, **kwargs) -> FetchResponse:
        """Fetch vectors by ID from a single namespace.

        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Args:
            ids: The vector IDs to fetch.
            namespace: The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            FetchResponse: Object which contains the list of Vector objects, and namespace name.

        Examples:

        .. code-block:: python

            >>> # Fetch vectors from a specific namespace
            >>> response = index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> for vector_id, vector in response.vectors.items():
            ...     print(f"{vector_id}: {vector.values}")

            >>> # Fetch vectors from the default namespace
            >>> response = index.fetch(ids=['id1', 'id2'])

        """
        args_dict = parse_non_empty_args([("namespace", namespace)])
        result = self._vector_api.fetch_vectors(ids=ids, **args_dict, **kwargs)
        # Copy response info from OpenAPI response if present
        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if hasattr(result, "_response_info"):
            response_info = result._response_info
        if response_info is None:
            response_info = extract_response_info({})

        fetch_response = FetchResponse(
            namespace=result.namespace,
            vectors={k: Vector.from_dict(v) for k, v in result.vectors.items()},
            usage=result.usage,
            _response_info=response_info,
        )
        return fetch_response

    @validate_and_convert_errors
    def fetch_by_metadata(
        self,
        filter: FilterTypedDict,
        namespace: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        **kwargs,
    ) -> FetchByMetadataResponse:
        """Fetch vectors by metadata filter.

        Look up and return vectors by metadata filter from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Args:
            filter: Metadata filter expression to select vectors.
                   See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`
            namespace: The namespace to fetch vectors from.
                      If not specified, the default namespace is used. [optional]
            limit: Max number of vectors to return. Defaults to 100. [optional]
            pagination_token: Pagination token to continue a previous listing operation. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            FetchByMetadataResponse: Object containing the fetched vectors, namespace, usage, and pagination token.

        Examples:

        .. code-block:: python

            >>> # Fetch vectors matching a complex filter
            >>> response = index.fetch_by_metadata(
            ...     filter={'genre': {'$in': ['comedy', 'drama']}, 'year': {'$eq': 2019}},
            ...     namespace='my_namespace',
            ...     limit=50
            ... )
            >>> print(f"Found {len(response.vectors)} vectors")

            >>> # Fetch vectors with pagination
            >>> response = index.fetch_by_metadata(
            ...     filter={'status': 'active'},
            ...     pagination_token='token123'
            ... )
            >>> if response.pagination:
            ...     print(f"Next page token: {response.pagination.next}")

        """
        request = IndexRequestFactory.fetch_by_metadata_request(
            filter=filter,
            namespace=namespace,
            limit=limit,
            pagination_token=pagination_token,
            **kwargs,
        )
        result = self._vector_api.fetch_vectors_by_metadata(request, **self._openapi_kwargs(kwargs))

        pagination = None
        if result.pagination and result.pagination.next:
            pagination = Pagination(next=result.pagination.next)

        # Copy response info from OpenAPI response if present
        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if hasattr(result, "_response_info"):
            response_info = result._response_info
        if response_info is None:
            response_info = extract_response_info({})

        fetch_by_metadata_response = FetchByMetadataResponse(
            namespace=result.namespace or "",
            vectors={k: Vector.from_dict(v) for k, v in result.vectors.items()},
            usage=result.usage,
            pagination=pagination,
            _response_info=response_info,
        )
        return fetch_by_metadata_response

    @validate_and_convert_errors
    def query(
        self,
        *args,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: SparseValues | SparseVectorTypedDict | None = None,
        **kwargs,
    ) -> QueryResponse | ApplyResult:
        """Query a namespace using a query vector.

        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        Args:
            top_k: The number of results to return for each query. Must be an integer greater than 1.
            vector: The query vector. This should be the same length as the dimension of the index
                   being queried. Each ``query()`` request can contain only one of the parameters
                   ``id`` or ``vector``. [optional]
            id: The unique ID of the vector to be used as a query vector.
               Each ``query()`` request can contain only one of the parameters
               ``vector`` or ``id``. [optional]
            namespace: The namespace to query vectors from. If not specified, the default namespace is used. [optional]
            filter: The filter to apply. You can use vector metadata to limit your search.
                   See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            include_values: Indicates whether vector values are included in the response.
                           If omitted the server will use the default value of False [optional]
            include_metadata: Indicates whether metadata is included in the response as well as the ids.
                             If omitted the server will use the default value of False [optional]
            sparse_vector: Sparse values of the query vector. Expected to be either a SparseValues object or a dict
                          of the form: ``{'indices': list[int], 'values': list[float]}``, where the lists each have
                          the same length. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            QueryResponse: Object which contains the list of the closest vectors as ScoredVector objects,
                          and namespace name. If async_req=True, returns ApplyResult instead.

        Examples:

        .. code-block:: python

            >>> # Query with a vector
            >>> response = index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> for match in response.matches:
            ...     print(f"ID: {match.id}, Score: {match.score}")

            >>> # Query using an existing vector ID
            >>> response = index.query(id='id1', top_k=10, namespace='my_namespace')

            >>> # Query with metadata filter
            >>> response = index.query(
            ...     vector=[1, 2, 3],
            ...     top_k=10,
            ...     namespace='my_namespace',
            ...     filter={'key': 'value'}
            ... )

            >>> # Query with include_values and include_metadata
            >>> response = index.query(
            ...     id='id1',
            ...     top_k=10,
            ...     namespace='my_namespace',
            ...     include_metadata=True,
            ...     include_values=True
            ... )

            >>> # Query with sparse vector (hybrid search)
            >>> response = index.query(
            ...     vector=[1, 2, 3],
            ...     sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            ...     top_k=10,
            ...     namespace='my_namespace'
            ... )

            >>> # Query with sparse vector using SparseValues object
            >>> from pinecone import SparseValues
            >>> response = index.query(
            ...     vector=[1, 2, 3],
            ...     sparse_vector=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
            ...     top_k=10,
            ...     namespace='my_namespace'
            ... )

        """
        response = self._query(
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

        if kwargs.get("async_req", False) or kwargs.get("async_threadpool_executor", False):
            # For async requests, the OpenAPI client wraps the response in ApplyResult
            # The response is already an ApplyResult[OpenAPIQueryResponse]
            return response  # type: ignore[return-value]  # ApplyResult is not tracked through OpenAPI layers
        else:
            return parse_query_response(response)

    def _query(
        self,
        *args,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: SparseValues | SparseVectorTypedDict | None = None,
        **kwargs,
    ) -> OpenAPIQueryResponse:
        if len(args) > 0:
            raise ValueError(
                "The argument order for `query()` has changed; please use keyword arguments instead of positional arguments. Example: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace='my_namespace')"
            )

        if top_k < 1:
            raise ValueError("top_k must be a positive integer")

        request = IndexRequestFactory.query_request(
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
        from typing import cast

        result = self._vector_api.query_vectors(request, **self._openapi_kwargs(kwargs))
        # When async_req=False, result is QueryResponse, not ApplyResult
        return cast(OpenAPIQueryResponse, result)

    @validate_and_convert_errors
    def query_namespaces(
        self,
        vector: list[float] | None,
        namespaces: list[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: int | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: SparseValues | SparseVectorTypedDict | None = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        """Query multiple namespaces in parallel and combine the results.

        The ``query_namespaces()`` method is used to make a query to multiple namespaces in parallel and combine
        the results into one result set.

        .. admonition:: Note

            Since several asynchronous calls are made on your behalf when calling this method, you will need to tune
            the **pool_threads** and **connection_pool_maxsize** parameter of the Index constructor to suit your workload.
            If these values are too small in relation to your workload, you will experience performance issues as
            requests queue up while waiting for a request thread to become available.

        Args:
            vector: The query vector, must be the same length as the dimension of the index being queried.
            namespaces: The list of namespaces to query.
            metric: Must be one of 'cosine', 'euclidean', 'dotproduct'. This is needed in order to merge results
                   across namespaces, since the interpretation of score depends on the index metric type.
            top_k: The number of results you would like to request from each namespace. Defaults to 10. [optional]
            filter: Pass an optional filter to filter results based on metadata. Defaults to None. [optional]
            include_values: Boolean field indicating whether vector values should be included with results. Defaults to None. [optional]
            include_metadata: Boolean field indicating whether vector metadata should be included with results. Defaults to None. [optional]
            sparse_vector: If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            QueryNamespacesResults: A QueryNamespacesResults object containing the combined results from all namespaces,
                                   as well as the combined usage cost in read units.

        Examples:

        .. code-block:: python

            from pinecone import Pinecone

            pc = Pinecone()

            index = pc.Index(
                host="index-name",
                pool_threads=32,
                connection_pool_maxsize=32
            )

            query_vec = [0.1, 0.2, 0.3]  # An embedding that matches the index dimension
            combined_results = index.query_namespaces(
                vector=query_vec,
                namespaces=['ns1', 'ns2', 'ns3', 'ns4'],
                metric="cosine",
                top_k=10,
                filter={'genre': {"$eq": "drama"}},
                include_values=True,
                include_metadata=True
            )

            for vec in combined_results.matches:
                print(vec.id, vec.score)
            print(combined_results.usage)

        """
        if namespaces is None or len(namespaces) == 0:
            raise ValueError("At least one namespace must be specified")
        if sparse_vector is None and vector is not None and len(vector) == 0:
            # If querying with a vector, it must not be empty
            raise ValueError("Query vector must not be empty")

        overall_topk = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(top_k=overall_topk, metric=metric)

        target_namespaces = set(namespaces)  # dedup namespaces
        async_futures = [
            self.query(
                vector=vector,
                namespace=ns,
                top_k=overall_topk,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                async_threadpool_executor=True,
                _preload_content=False,
                **kwargs,
            )
            for ns in target_namespaces
        ]

        from typing import cast
        from concurrent.futures import Future

        # async_futures is a list of ApplyResult, but as_completed expects Future
        futures: list[Future[Any]] = cast(list[Future[Any]], async_futures)
        for result in as_completed(futures):
            raw_result = result.result()
            response = json.loads(raw_result.data.decode("utf-8"))
            aggregator.add_results(response)

        final_results = aggregator.get_results()
        return final_results

    @validate_and_convert_errors
    def update(
        self,
        id: str | None = None,
        values: list[float] | None = None,
        set_metadata: VectorMetadataTypedDict | None = None,
        namespace: str | None = None,
        sparse_values: SparseValues | SparseVectorTypedDict | None = None,
        filter: FilterTypedDict | None = None,
        dry_run: bool | None = None,
        **kwargs,
    ) -> UpdateResponse:
        """Update vectors in a namespace.

        The Update operation updates vectors in a namespace.

        This method supports two update modes:

        1. **Single vector update by ID**: Provide ``id`` to update a specific vector.
           - Updates the vector with the given ID
           - If ``values`` is included, it will overwrite the previous vector values
           - If ``set_metadata`` is included, the metadata will be merged with existing metadata on the vector.
             Fields specified in ``set_metadata`` will overwrite existing fields with the same key, while
             fields not in ``set_metadata`` will remain unchanged.

        2. **Bulk update by metadata filter**: Provide ``filter`` to update all vectors matching the filter criteria.
           - Updates all vectors in the namespace that match the filter expression
           - Useful for updating metadata across multiple vectors at once
           - If ``set_metadata`` is included, the metadata will be merged with existing metadata on each vector.
             Fields specified in ``set_metadata`` will overwrite existing fields with the same key, while
             fields not in ``set_metadata`` will remain unchanged.
           - The response includes ``matched_records`` indicating how many vectors were updated

        Either ``id`` or ``filter`` must be provided (but not both in the same call).

        Args:
            id: Vector's unique id. Required for single vector updates. Must not be provided when using filter. [optional]
            values: Vector values to set. [optional]
            set_metadata: Metadata to merge with existing metadata on the vector(s). Fields specified will overwrite
                         existing fields with the same key, while fields not specified will remain unchanged. [optional]
            namespace: Namespace name where to update the vector(s). [optional]
            sparse_values: Sparse values to update for the vector. Expected to be either a SparseValues object or a dict
                          of the form: ``{'indices': list[int], 'values': list[float]}`` where the lists each have
                          the same length. [optional]
            filter: A metadata filter expression. When provided, updates all vectors in the namespace that match
                   the filter criteria. See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`.
                   Must not be provided when using id. Either ``id`` or ``filter`` must be provided. [optional]
            dry_run: If ``True``, return the number of records that match the ``filter`` without executing
                    the update. Only meaningful when using ``filter`` (not with ``id``). Useful for previewing
                    the impact of a bulk update before applying changes. Defaults to ``False``. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            UpdateResponse: An UpdateResponse object. When using filter-based updates, the response includes
                           ``matched_records`` indicating the number of vectors that were updated (or would be updated if
                           ``dry_run=True``).

        Examples:

        **Single vector update by ID:**

        .. code-block:: python

            >>> # Update vector values
            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')

            >>> # Update vector metadata
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace')

            >>> # Update vector values and sparse values
            >>> index.update(
            ...     id='id1',
            ...     values=[1, 2, 3],
            ...     sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            ...     namespace='my_namespace'
            ... )

            >>> # Update with SparseValues object
            >>> from pinecone import SparseValues
            >>> index.update(
            ...     id='id1',
            ...     values=[1, 2, 3],
            ...     sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
            ...     namespace='my_namespace'
            ... )

        **Bulk update by metadata filter:**

        .. code-block:: python

            >>> # Update metadata for all vectors matching the filter
            >>> response = index.update(
            ...     set_metadata={'status': 'active'},
            ...     filter={'genre': {'$eq': 'drama'}},
            ...     namespace='my_namespace'
            ... )
            >>> print(f"Updated {response.matched_records} vectors")

            >>> # Preview how many vectors would be updated (dry run)
            >>> response = index.update(
            ...     set_metadata={'status': 'active'},
            ...     filter={'genre': {'$eq': 'drama'}},
            ...     namespace='my_namespace',
            ...     dry_run=True
            ... )
            >>> print(f"Would update {response.matched_records} vectors")

        """
        # Validate that exactly one of id or filter is provided
        if id is None and filter is None:
            raise ValueError("Either 'id' or 'filter' must be provided to update vectors.")
        if id is not None and filter is not None:
            raise ValueError(
                "Cannot provide both 'id' and 'filter' in the same update call. Use 'id' for single vector updates or 'filter' for bulk updates."
            )
        result = self._vector_api.update_vector(
            IndexRequestFactory.update_request(
                id=id,
                values=values,
                set_metadata=set_metadata,
                namespace=namespace,
                sparse_values=sparse_values,
                filter=filter,
                dry_run=dry_run,
                **kwargs,
            ),
            **self._openapi_kwargs(kwargs),
        )
        # Extract response info from result if it's an OpenAPI model with _response_info
        response_info = None
        matched_records = None
        if hasattr(result, "_response_info"):
            response_info = result._response_info
        else:
            # If result is a dict or empty, create default response_info
            from pinecone.utils.response_info import extract_response_info

            response_info = extract_response_info({})

        # Extract matched_records from OpenAPI model
        if hasattr(result, "matched_records"):
            matched_records = result.matched_records
        # Check _data_store for fields not in the OpenAPI spec
        if hasattr(result, "_data_store"):
            if matched_records is None:
                matched_records = result._data_store.get(
                    "matchedRecords"
                ) or result._data_store.get("matched_records")

        return UpdateResponse(matched_records=matched_records, _response_info=response_info)

    @validate_and_convert_errors
    def describe_index_stats(
        self, filter: FilterTypedDict | None = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """Get statistics about the index's contents.

        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Args:
            filter: If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
                   See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            DescribeIndexStatsResponse: Object which contains stats about the index.

        Examples:

        .. code-block:: python

            >>> pc = Pinecone()
            >>> index = pc.Index(host="example-index-host")
            >>> stats = index.describe_index_stats()
            >>> print(f"Total vectors: {stats.total_vector_count}")
            >>> print(f"Dimension: {stats.dimension}")
            >>> print(f"Namespaces: {list(stats.namespaces.keys())}")

            >>> # Get stats for vectors matching a filter
            >>> filtered_stats = index.describe_index_stats(
            ...     filter={'genre': {'$eq': 'drama'}}
            ... )

        """
        from typing import cast

        result = self._vector_api.describe_index_stats(
            IndexRequestFactory.describe_index_stats_request(filter, **kwargs),
            **self._openapi_kwargs(kwargs),
        )
        # When async_req=False, result is IndexDescription, not ApplyResult
        return cast(DescribeIndexStatsResponse, result)

    @validate_and_convert_errors
    def list_paginated(
        self,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str | None = None,
        **kwargs,
    ) -> ListResponse:
        """List vector IDs based on an id prefix within a single namespace (paginated).

        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the ``list`` method to avoid having to handle pagination tokens manually.

        Args:
            prefix: The id prefix to match. If unspecified, an empty string prefix will
                   be used with the effect of listing all ids in a namespace [optional]
            limit: The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token: A token needed to fetch the next page of results. This token is returned
                            in the response if additional results are available. [optional]
            namespace: The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            ListResponse: Object which contains the list of ids, the namespace name, pagination information,
                         and usage showing the number of read_units consumed.

        Examples:

        .. code-block:: python

            >>> # List vectors with a prefix
            >>> results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace')
            >>> [v.id for v in results.vectors]
            ['99', '990', '991', '992', '993']
            >>> # Get next page
            >>> if results.pagination and results.pagination.next:
            ...     next_results = index.list_paginated(
            ...         prefix='99',
            ...         limit=5,
            ...         namespace='my_namespace',
            ...         pagination_token=results.pagination.next
            ...     )

        """
        args_dict = IndexRequestFactory.list_paginated_args(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            namespace=namespace,
            **kwargs,
        )
        from typing import cast

        result = self._vector_api.list_vectors(**args_dict, **kwargs)
        # When async_req=False, result is ListResponse, not ApplyResult
        return cast(ListResponse, result)

    @validate_and_convert_errors
    def list(self, **kwargs):
        """List vector IDs based on an id prefix within a single namespace (generator).

        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Args:
            prefix: The id prefix to match. If unspecified, an empty string prefix will
                   be used with the effect of listing all ids in a namespace [optional]
            limit: The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token: A token needed to fetch the next page of results. This token is returned
                            in the response if additional results are available. [optional]
            namespace: The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Yields:
            list[str]: A list of vector IDs for each page of results.

        Examples:

        .. code-block:: python

            >>> # Iterate over all vector IDs with a prefix
            >>> for ids in index.list(prefix='99', limit=5, namespace='my_namespace'):
            ...     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
            ['999']

            >>> # Convert generator to list (be cautious with large datasets)
            >>> all_ids = []
            >>> for ids in index.list(prefix='99', namespace='my_namespace'):
            ...     all_ids.extend(ids)

        """
        done = False
        while not done:
            results = self.list_paginated(**kwargs)
            if len(results.vectors) > 0:
                yield [v.id for v in results.vectors]

            if results.pagination:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    @validate_and_convert_errors
    def start_import(
        self,
        uri: str,
        integration_id: str | None = None,
        error_mode: ("ImportErrorMode" | Literal["CONTINUE", "ABORT"] | str) | None = "CONTINUE",
    ) -> "StartImportResponse":
        """
        Args:
            uri (str): The URI of the data to import. The URI must start with the scheme of a supported storage provider.
            integration_id (str | None, optional): If your bucket requires authentication to access, you need to pass the id of your storage integration using this property. Defaults to None.
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
            limit (int | None): The maximum number of operations to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (str | None): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
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
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> "ListImportsResponse":
        """
        Args:
            limit (int | None): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (str | None): A token needed to fetch the next page of results. This token is returned
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
            id: The id of the import operation to cancel.

        Returns:
            The response from the cancel operation.

        Examples:

        .. code-block:: python

            >>> # Cancel an import operation
            >>> index.cancel_import(id="import-123")

        """
        return self.bulk_import.cancel(id=id)

    @validate_and_convert_errors
    @require_kwargs
    def create_namespace(
        self, name: str, schema: dict[str, Any] | None = None, **kwargs
    ) -> "NamespaceDescription":
        """Create a namespace in a serverless index.

        Create a namespace in a serverless index. For guidance and examples, see
        `Manage namespaces <https://docs.pinecone.io/guides/manage-data/manage-namespaces>`_.

        **Note:** This operation is not supported for pod-based indexes.

        Args:
            name: The name of the namespace to create.
            schema: Optional schema configuration for the namespace as a dictionary. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            NamespaceDescription: Information about the created namespace including vector count.

        Examples:

        .. code-block:: python

            >>> # Create a namespace with just a name
            >>> namespace = index.create_namespace(name="my-namespace")
            >>> print(f"Created namespace: {namespace.name}, Vector count: {namespace.vector_count}")

            >>> # Create a namespace with schema configuration
            >>> from pinecone.core.openapi.db_data.model.create_namespace_request_schema import CreateNamespaceRequestSchema
            >>> schema = CreateNamespaceRequestSchema(fields={...})
            >>> namespace = index.create_namespace(name="my-namespace", schema=schema)

        """
        return self.namespace.create(name=name, schema=schema, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def describe_namespace(self, namespace: str, **kwargs) -> "NamespaceDescription":
        """Describe a namespace within an index, showing the vector count within the namespace.

        Args:
            namespace: The namespace to describe.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            NamespaceDescription: Information about the namespace including vector count.

        Examples:

        .. code-block:: python

            >>> namespace_info = index.describe_namespace(namespace="my-namespace")
            >>> print(f"Namespace: {namespace_info.name}")
            >>> print(f"Vector count: {namespace_info.vector_count}")

        """
        return self.namespace.describe(namespace=namespace, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def delete_namespace(self, namespace: str, **kwargs) -> dict[str, Any]:
        """Delete a namespace from an index.

        Args:
            namespace: The namespace to delete.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            dict[str, Any]: Response from the delete operation.

        Examples:

        .. code-block:: python

            >>> result = index.delete_namespace(namespace="my-namespace")
            >>> print("Namespace deleted successfully")

        """
        from typing import cast

        result = self.namespace.delete(namespace=namespace, **kwargs)
        return cast(dict[str, Any], result)

    @validate_and_convert_errors
    @require_kwargs
    def list_namespaces(
        self, limit: int | None = None, **kwargs
    ) -> Iterator[ListNamespacesResponse]:
        """List all namespaces in an index.

        This method automatically handles pagination to return all results.

        Args:
            limit: The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            Iterator[ListNamespacesResponse]: An iterator that yields ListNamespacesResponse objects containing the list of namespaces.

        Examples:

        .. code-block:: python

            >>> # Iterate over all namespaces
            >>> for namespace_response in index.list_namespaces(limit=5):
            ...     for namespace in namespace_response.namespaces:
            ...         print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")

            >>> # Convert to list (be cautious with large datasets)
            >>> results = list(index.list_namespaces(limit=5))
            >>> for namespace_response in results:
            ...     for namespace in namespace_response.namespaces:
            ...         print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")

        """
        return self.namespace.list(limit=limit, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def list_namespaces_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> ListNamespacesResponse:
        """List all namespaces in an index with pagination support.

        The response includes pagination information if there are more results available.

        Consider using the ``list_namespaces`` method to avoid having to handle pagination tokens manually.

        Args:
            limit: The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            pagination_token: A token needed to fetch the next page of results. This token is returned
                            in the response if additional results are available. [optional]
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            ListNamespacesResponse: Object containing the list of namespaces and pagination information.

        Examples:

        .. code-block:: python

            >>> # Get first page of namespaces
            >>> results = index.list_namespaces_paginated(limit=5)
            >>> for namespace in results.namespaces:
            ...     print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")

            >>> # Get next page if available
            >>> if results.pagination and results.pagination.next:
            ...     next_results = index.list_namespaces_paginated(
            ...         limit=5,
            ...         pagination_token=results.pagination.next
            ...     )

        """
        return self.namespace.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )
