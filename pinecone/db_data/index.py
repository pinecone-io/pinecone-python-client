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

    def __init__(self, apply_result: ApplyResult):
        self._apply_result = apply_result

    def get(self, timeout=None):
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

    def __getattr__(self, name):
        # Delegate other methods to the underlying ApplyResult
        return getattr(self._apply_result, name)


class Index(PluginAware, IndexInterface):
    """
    A client for interacting with a Pinecone index via REST API.
    For improved performance, use the Pinecone GRPC index client.
    """

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
            filter (dict[str, str | float | int | bool | List | dict]):
                Metadata filter expression to select vectors.
                See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`
            namespace (str): The namespace to fetch vectors from.
                            If not specified, the default namespace is used. [optional]
            limit (int): Max number of vectors to return. Defaults to 100. [optional]
            pagination_token (str): Pagination token to continue a previous listing operation. [optional]

        Returns:
            FetchByMetadataResponse: Object containing the fetched vectors, namespace, usage, and pagination token.
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
            id (str): The id of the import operation to cancel.
        """
        return self.bulk_import.cancel(id=id)

    @validate_and_convert_errors
    @require_kwargs
    def create_namespace(
        self, name: str, schema: dict[str, Any] | None = None, **kwargs
    ) -> "NamespaceDescription":
        return self.namespace.create(name=name, schema=schema, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def describe_namespace(self, namespace: str, **kwargs) -> "NamespaceDescription":
        return self.namespace.describe(namespace=namespace, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def delete_namespace(self, namespace: str, **kwargs) -> dict[str, Any]:
        from typing import cast

        result = self.namespace.delete(namespace=namespace, **kwargs)
        return cast(dict[str, Any], result)

    @validate_and_convert_errors
    @require_kwargs
    def list_namespaces(
        self, limit: int | None = None, **kwargs
    ) -> Iterator[ListNamespacesResponse]:
        return self.namespace.list(limit=limit, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    def list_namespaces_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> ListNamespacesResponse:
        return self.namespace.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )
