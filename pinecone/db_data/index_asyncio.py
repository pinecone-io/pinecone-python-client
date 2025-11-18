from __future__ import annotations

from pinecone.utils.tqdm import tqdm


import logging
import asyncio
import json

from .index_asyncio_interface import IndexAsyncioInterface
from .query_results_aggregator import QueryResultsAggregator
from typing import List, Dict, Any, Literal, AsyncIterator, TYPE_CHECKING
from typing_extensions import Self

from pinecone.config import ConfigBuilder

from pinecone.openapi_support import AsyncioApiClient
from pinecone.core.openapi.db_data.api.vector_operations_api import AsyncioVectorOperationsApi
from pinecone.core.openapi.db_data import API_VERSION
from pinecone.core.openapi.db_data.models import (
    QueryResponse as OpenAPIQueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertRequest,
    DeleteRequest,
    ListResponse,
    SearchRecordsResponse,
    ListNamespacesResponse,
    NamespaceDescription,
)

from ..utils import (
    setup_async_openapi_client,
    parse_non_empty_args,
    validate_and_convert_errors,
    filter_dict,
    require_kwargs,
)
from .types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    FilterTypedDict,
    SearchQueryTypedDict,
    SearchRerankTypedDict,
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

from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS
from .index import IndexRequestFactory

from .vector_factory import VectorFactory
from .query_results_aggregator import QueryNamespacesResults

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from .resources.asyncio.bulk_import_asyncio import BulkImportResourceAsyncio
    from .resources.asyncio.namespace_asyncio import NamespaceResourceAsyncio

    from pinecone.core.openapi.db_data.models import (
        StartImportResponse,
        ListImportsResponse,
        ImportModel,
    )


logger = logging.getLogger(__name__)
""" :meta private: """

__all__ = ["_IndexAsyncio", "IndexAsyncio"]

_OPENAPI_ENDPOINT_PARAMS = (
    "_return_http_data_only",
    "_preload_content",
    "_request_timeout",
    "_check_input_type",
    "_check_return_type",
)
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


class _IndexAsyncio(IndexAsyncioInterface):
    """
    The `IndexAsyncio` class provides an asynchronous interface to interact with Pinecone indexes.

    Like the `Index` class, it provides methods to upsert, delete, fetch, and query vectors in a Pinecone index.

    The `IndexAsyncio` class is instantiated through a helper method of the `Pinecone` class. It is not meant to be instantiated directly.
    This is to ensure that configuration is handled consistently across all Pinecone objects.

    ## Managing the async context

    The `IndexAsyncio` class relies on an underlying `aiohttp` `ClientSession` to make asynchronous HTTP requests. To ensure that the session is properly closed, you
    should use the `async with` syntax when creating a `IndexAsyncio` object to use it as an async context manager. This will ensure that the session is properly
    closed when the context is exited.

    ```python
    import asyncio
    from pinecone import Pinecone

    async def main():
        pc = Pinecone(api_key='YOUR_API_KEY')
        async with pc.IndexAsyncio(host='example-index-dojoi3u.svc.eu-west1-gcp.pinecone.io') as idx:
            # Do async things
            await idx.upsert(
                vectors=[
                    ...
                ]
            )

    asyncio.run(main())
    ```

    As an alternative, if you prefer to avoid code with a nested appearance and are willing to manage cleanup yourself, you can await the `close()` method to close the session when you are done.

    ```python
    import asyncio
    from pinecone import Pinecone

    async def main():
        pc = Pinecone(api_key='YOUR_API_KEY')
        idx = pc.IndexAsyncio(host='example-index-dojoi3u.svc.eu-west1-gcp.pinecone.io')

        # Do async things
        await idx.describe_index_stats()

        # After you're done, you're responsible for calling this yourself
        await pc.close()

    asyncio.run(main())
    ```

    Failing to do this may result in error messages appearing from the underlyling aiohttp library.
    """

    config: "Config"
    """ :meta private: """

    _openapi_config: "OpenApiConfiguration"
    """ :meta private: """

    _vector_api: AsyncioVectorOperationsApi
    """ :meta private: """

    _api_client: AsyncioApiClient
    """ :meta private: """

    _bulk_import_resource: "BulkImportResourceAsyncio" | None
    """ :meta private: """

    _namespace_resource: "NamespaceResourceAsyncio" | None
    """ :meta private: """

    def __init__(
        self,
        api_key: str,
        host: str,
        additional_headers: dict[str, str] | None = {},
        openapi_config=None,
        **kwargs,
    ) -> None:
        self.config = ConfigBuilder.build(
            api_key=api_key, host=host, additional_headers=additional_headers, **kwargs
        )
        """ :meta private: """
        self._openapi_config = ConfigBuilder.build_openapi_config(self.config, openapi_config)
        """ :meta private: """

        connection_pool_maxsize = kwargs.get("connection_pool_maxsize", None)
        if connection_pool_maxsize is not None:
            self._openapi_config.connection_pool_maxsize = connection_pool_maxsize

        self._vector_api = setup_async_openapi_client(
            api_client_klass=AsyncioApiClient,
            api_klass=AsyncioVectorOperationsApi,
            config=self.config,
            openapi_config=self._openapi_config,
            api_version=API_VERSION,
        )
        """ :meta private: """

        self._api_client = self._vector_api.api_client
        """ :meta private: """

        self._bulk_import_resource = None
        """ :meta private: """

        self._namespace_resource = None
        """ :meta private: """

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_value: Exception | None, traceback: Any | None
    ) -> bool | None:
        await self._api_client.close()
        return None

    async def close(self) -> None:
        """Cleanup resources used by the Pinecone Index client.

        This method should be called when the client is no longer needed so that
        it can cleanup the aioahttp session and other resources.

        After close has been called, the client instance should not be used.

        ```python
        import asyncio
        from pinecone import Pinecone

        async def main():
            pc = Pinecone()
            idx = pc.IndexAsyncio(host='example-index-dojoi3u.svc.eu-west1-gcp.pinecone.io')
            await idx.upsert_records(
                namespace='my-namespace',
                records=[
                    ...
                ]
            )

            # Close the client when done
            await idx.close()

        asyncio.run(main())
        ```

        If you are using the client as a context manager, the close method is called automatically
        when exiting.

        ```python
        import asyncio
        from pinecone import Pinecone

        async def main():
            pc = Pinecone()
            async with pc.IndexAscynio(host='example-index-dojoi3u.svc.eu-west1-gcp.pinecone.io') as idx:
                await idx.upsert_records(
                    namespace='my-namespace',
                    records=[
                        ...
                    ]
                )

        # No need to call close in this case because the "async with" syntax
        # automatically calls close when exiting the block.
        asyncio.run(main())
        ```

        """
        await self._api_client.close()

    @property
    def bulk_import(self) -> "BulkImportResourceAsyncio":
        """:meta private:"""
        if self._bulk_import_resource is None:
            from .resources.asyncio.bulk_import_asyncio import BulkImportResourceAsyncio

            self._bulk_import_resource = BulkImportResourceAsyncio(api_client=self._api_client)
        return self._bulk_import_resource

    @property
    def namespace(self) -> "NamespaceResourceAsyncio":
        """:meta private:"""
        if self._namespace_resource is None:
            from .resources.asyncio.namespace_asyncio import NamespaceResourceAsyncio

            self._namespace_resource = NamespaceResourceAsyncio(api_client=self._api_client)
        return self._namespace_resource

    @validate_and_convert_errors
    async def upsert(
        self,
        vectors: (
            list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict]
        ),
        namespace: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse:
        _check_type = kwargs.pop("_check_type", True)

        if batch_size is None:
            return await self._upsert_batch(vectors, namespace, _check_type, **kwargs)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        upsert_tasks = [
            self._upsert_batch(vectors[i : i + batch_size], namespace, _check_type, **kwargs)
            for i in range(0, len(vectors), batch_size)
        ]

        total_upserted = 0
        last_result = None
        with tqdm(total=len(vectors), desc="Upserted vectors", disable=not show_progress) as pbar:
            for task in asyncio.as_completed(upsert_tasks):
                res = await task
                pbar.update(res.upserted_count)
                total_upserted += res.upserted_count
                last_result = res

        # Create aggregated response with metadata from last completed batch
        # Note: For parallel batches, this uses the last completed result (order may vary)
        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if last_result and hasattr(last_result, "_response_info"):
            response_info = last_result._response_info
        if response_info is None:
            response_info = extract_response_info({})

        return UpsertResponse(upserted_count=total_upserted, _response_info=response_info)

    @validate_and_convert_errors
    async def _upsert_batch(
        self,
        vectors: (
            list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict]
        ),
        namespace: str | None,
        _check_type: bool,
        **kwargs,
    ) -> UpsertResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])

        def vec_builder(v):
            return VectorFactory.build(v, check_type=_check_type)

        # Convert OpenAPI UpsertResponse to dataclass UpsertResponse
        result = await self._vector_api.upsert_vectors(
            UpsertRequest(
                vectors=list(map(vec_builder, vectors)),
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )

        from pinecone.utils.response_info import extract_response_info

        response_info = None
        if hasattr(result, "_response_info"):
            response_info = result._response_info
        if response_info is None:
            response_info = extract_response_info({})

        return UpsertResponse(upserted_count=result.upserted_count, _response_info=response_info)

    @validate_and_convert_errors
    async def upsert_from_dataframe(
        self, df, namespace: str | None = None, batch_size: int = 500, show_progress: bool = True
    ):
        raise NotImplementedError("upsert_from_dataframe is not implemented for asyncio")

    @validate_and_convert_errors
    async def delete(
        self,
        ids: list[str] | None = None,
        delete_all: bool | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args(
            [("ids", ids), ("delete_all", delete_all), ("namespace", namespace), ("filter", filter)]
        )

        from typing import cast

        result = await self._vector_api.delete_vectors(
            DeleteRequest(
                **args_dict,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in _OPENAPI_ENDPOINT_PARAMS and v is not None
                },
                _check_type=_check_type,
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )
        return cast(dict[str, Any], result)

    @validate_and_convert_errors
    async def fetch(self, ids: list[str], namespace: str | None = None, **kwargs) -> FetchResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])
        result = await self._vector_api.fetch_vectors(ids=ids, **args_dict, **kwargs)
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
    async def fetch_by_metadata(
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

            import asyncio
            from pinecone import Pinecone

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-host") as idx:
                    result = await idx.fetch_by_metadata(
                        filter={'genre': {'$in': ['comedy', 'drama']}, 'year': {'$eq': 2019}},
                        namespace='my_namespace',
                        limit=50
                    )
                    for vec_id in result.vectors:
                        vector = result.vectors[vec_id]
                        print(vector.id)
                        print(vector.metadata)

            asyncio.run(main())

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
        result = await self._vector_api.fetch_vectors_by_metadata(
            request, **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )

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
    async def query(
        self,
        *args,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> QueryResponse:
        response = await self._query(
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
        return parse_query_response(response)

    async def _query(
        self,
        *args,
        top_k: int,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> OpenAPIQueryResponse:
        if len(args) > 0:
            raise ValueError(
                "Please use keyword arguments instead of positional arguments. Example: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace='my_namespace')"
            )

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

        result = await self._vector_api.query_vectors(
            request, **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )
        return cast(OpenAPIQueryResponse, result)

    @validate_and_convert_errors
    async def query_namespaces(
        self,
        namespaces: list[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: int | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        vector: list[float] | None = None,
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
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
        tasks = [
            self._query(
                top_k=overall_topk,
                vector=vector,
                namespace=ns,
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

        for task in asyncio.as_completed(tasks):
            raw_result = await task
            # When _preload_content=False, _query returns a RESTResponse object
            from pinecone.openapi_support.rest_utils import RESTResponse

            if isinstance(raw_result, RESTResponse):
                response = json.loads(raw_result.data.decode("utf-8"))
                aggregator.add_results(response)
            else:
                # Fallback: if somehow we got an OpenAPIQueryResponse, parse it
                response = json.loads(raw_result.to_dict())
                aggregator.add_results(response)

        final_results = aggregator.get_results()
        return final_results

    @validate_and_convert_errors
    async def update(
        self,
        id: str | None = None,
        values: list[float] | None = None,
        set_metadata: VectorMetadataTypedDict | None = None,
        namespace: str | None = None,
        sparse_values: (SparseValues | SparseVectorTypedDict) | None = None,
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
        result = await self._vector_api.update_vector(
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
    async def describe_index_stats(
        self, filter: FilterTypedDict | None = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        from typing import cast

        result = await self._vector_api.describe_index_stats(
            IndexRequestFactory.describe_index_stats_request(filter, **kwargs),
            **self._openapi_kwargs(kwargs),
        )
        return cast(DescribeIndexStatsResponse, result)

    @validate_and_convert_errors
    async def list_paginated(
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

        result = await self._vector_api.list_vectors(**args_dict, **kwargs)
        return cast(ListResponse, result)

    @validate_and_convert_errors
    async def list(self, **kwargs) -> AsyncIterator[list[str]]:
        done = False
        while not done:
            results = await self.list_paginated(**kwargs)
            if len(results.vectors) > 0:
                yield [v.id for v in results.vectors]

            if results.pagination:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    async def upsert_records(self, namespace: str, records: List[Dict]) -> UpsertResponse:
        args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=records)
        # Use _return_http_data_only=False to get headers for LSN extraction
        result = await self._vector_api.upsert_records_namespace(
            _return_http_data_only=False, **args
        )
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

    async def search(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: (SearchRerankTypedDict | SearchRerank) | None = None,
        fields: List[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        if namespace is None:
            raise Exception("Namespace is required when searching records")

        request = IndexRequestFactory.search_request(query=query, rerank=rerank, fields=fields)

        from typing import cast

        result = await self._vector_api.search_records_namespace(namespace, request)
        return cast(SearchRecordsResponse, result)

    async def search_records(
        self,
        namespace: str,
        query: SearchQueryTypedDict | SearchQuery,
        rerank: (SearchRerankTypedDict | SearchRerank) | None = None,
        fields: List[str] | None = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        return await self.search(namespace, query=query, rerank=rerank, fields=fields)

    def _openapi_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return filter_dict(kwargs, OPENAPI_ENDPOINT_PARAMS)

    async def start_import(
        self,
        uri: str,
        integration_id: str | None = None,
        error_mode: Literal["CONTINUE", "ABORT"] | None = "CONTINUE",
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
            >>> index = Pinecone().IndexAsyncio(host="example-index.svc.aped-4627-b74a.pinecone.io")
            >>> await index.start_import(uri="s3://bucket-name/path/to/data.parquet")
            { id: "1" }

        """
        return await self.bulk_import.start(
            uri=uri, integration_id=integration_id, error_mode=error_mode
        )

    async def list_imports(self, **kwargs) -> AsyncIterator["ImportModel"]:
        """
        Args:
            limit (Optional[int]): The maximum number of operations to fetch in each network call. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): When there are multiple pages of results, a pagination token is returned in the response. The token can be used
                to fetch the next page of results. [optional]

        Returns an async generator that yields each import operation. It automatically handles pagination tokens on your behalf so you can
        easily iterate over all results. The `list_imports` method accepts all of the same arguments as `list_imports_paginated`

        ```python
        async for op in index.list_imports():
            print(op)
        ```
        """
        async for op in self.bulk_import.list(**kwargs):
            yield op

    async def list_imports_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> "ListImportsResponse":
        """
        Args:
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns:
            `ListImportsResponse` object which contains the list of operations as ImportModel objects, pagination information,
                and usage showing the number of read_units consumed.

        The `list_imports_paginated` operation returns information about import operations.
        It returns operations in a paginated form, with a pagination token to fetch the next page of results.

        Consider using the `list_imports` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = await index.list_imports_paginated(limit=5)
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
            >>> next_results = await index.list_imports_paginated(limit=5, pagination_token=results.pagination.next)

        """
        return await self.bulk_import.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )

    async def describe_import(self, id: str) -> "ImportModel":
        """
        Args:
            id (str): The id of the import operation. This value is returned when
            starting an import, and can be looked up using list_imports.

        Returns:
            ImportModel: An object containing operation id, status, and other details.

        `describe_import` is used to get detailed information about a specific import operation.
        """
        return await self.bulk_import.describe(id=id)

    async def cancel_import(self, id: str):
        """Cancel an import operation.

        Args:
            id (str): The id of the import operation to cancel.
        """
        return await self.bulk_import.cancel(id=id)

    @validate_and_convert_errors
    @require_kwargs
    async def create_namespace(
        self, name: str, schema: dict[str, Any] | None = None, **kwargs
    ) -> "NamespaceDescription":
        return await self.namespace.create(name=name, schema=schema, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    async def describe_namespace(self, namespace: str, **kwargs) -> "NamespaceDescription":
        return await self.namespace.describe(namespace=namespace, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    async def delete_namespace(self, namespace: str, **kwargs) -> dict[str, Any]:
        from typing import cast

        result = await self.namespace.delete(namespace=namespace, **kwargs)
        return cast(dict[str, Any], result)

    @validate_and_convert_errors
    @require_kwargs
    async def list_namespaces(  # type: ignore[override, misc]  # mypy limitation: async generators in abstract methods
        self, limit: int | None = None, **kwargs
    ) -> AsyncIterator[ListNamespacesResponse]:
        async for namespace in self.namespace.list(limit=limit, **kwargs):
            yield namespace

    @validate_and_convert_errors
    @require_kwargs
    async def list_namespaces_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> ListNamespacesResponse:
        return await self.namespace.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )


IndexAsyncio = _IndexAsyncio
