from pinecone.utils.tqdm import tqdm


import logging
import asyncio
import json

from .index_asyncio_interface import IndexAsyncioInterface
from .query_results_aggregator import QueryResultsAggregator
from typing import Union, List, Optional, Dict, Any, Literal

from pinecone.config import ConfigBuilder

from pinecone.openapi_support import AsyncioApiClient
from pinecone.core.openapi.db_data.api.vector_operations_api import AsyncioVectorOperationsApi
from pinecone.core.openapi.db_data import API_VERSION
from pinecone.core.openapi.db_data.models import (
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertRequest,
    UpsertResponse,
    DeleteRequest,
    ListResponse,
    SearchRecordsResponse,
)

from ..utils import (
    setup_async_openapi_client,
    parse_non_empty_args,
    validate_and_convert_errors,
    filter_dict,
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
from .dataclasses import Vector, SparseValues, FetchResponse, SearchQuery, SearchRerank

from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS
from .index import IndexRequestFactory

from .vector_factory import VectorFactory
from .query_results_aggregator import QueryNamespacesResults
from .features.bulk_import import ImportFeatureMixinAsyncio


logger = logging.getLogger(__name__)
""" @private """

__all__ = ["_IndexAsyncio"]

_OPENAPI_ENDPOINT_PARAMS = (
    "_return_http_data_only",
    "_preload_content",
    "_request_timeout",
    "_check_input_type",
    "_check_return_type",
)
""" @private """


def parse_query_response(response: QueryResponse):
    if hasattr(response, "_data_store"):
        # I'm not sure, but I think this is no longer needed. At some point
        # in the past the query response returned "results" instead of matches
        # and then for some time it returned both keys even though "results"
        # was always empty. I'm leaving this here just in case.
        response._data_store.pop("results", None)
    return response


class _IndexAsyncio(IndexAsyncioInterface, ImportFeatureMixinAsyncio):
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

    def __init__(
        self,
        api_key: str,
        host: str,
        additional_headers: Optional[Dict[str, str]] = {},
        openapi_config=None,
        **kwargs,
    ):
        self.config = ConfigBuilder.build(
            api_key=api_key, host=host, additional_headers=additional_headers, **kwargs
        )
        """ @private """
        self._openapi_config = ConfigBuilder.build_openapi_config(self.config, openapi_config)
        """ @private """

        if kwargs.get("connection_pool_maxsize", None):
            self._openapi_config.connection_pool_maxsize = kwargs.get("connection_pool_maxsize")

        self._vector_api = setup_async_openapi_client(
            api_client_klass=AsyncioApiClient,
            api_klass=AsyncioVectorOperationsApi,
            config=self.config,
            openapi_config=self._openapi_config,
            api_version=API_VERSION,
        )
        """ @private """

        self._api_client = self._vector_api.api_client
        """ @private """

        # Pass the same api_client to the ImportFeatureMixinAsyncio
        # This is important for async context management to work correctly
        super().__init__(api_client=self._api_client)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._api_client.close()

    async def close(self):
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

    @validate_and_convert_errors
    async def upsert(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
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
        with tqdm(total=len(vectors), desc="Upserted vectors", disable=not show_progress) as pbar:
            for task in asyncio.as_completed(upsert_tasks):
                res = await task
                pbar.update(res.upserted_count)
                total_upserted += res.upserted_count

        return UpsertResponse(upserted_count=total_upserted)

    @validate_and_convert_errors
    async def _upsert_batch(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str],
        _check_type: bool,
        **kwargs,
    ) -> UpsertResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])

        def vec_builder(v):
            return VectorFactory.build(v, check_type=_check_type)

        return await self._vector_api.upsert_vectors(
            UpsertRequest(
                vectors=list(map(vec_builder, vectors)),
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )

    @validate_and_convert_errors
    async def upsert_from_dataframe(
        self, df, namespace: Optional[str] = None, batch_size: int = 500, show_progress: bool = True
    ):
        raise NotImplementedError("upsert_from_dataframe is not implemented for asyncio")

    @validate_and_convert_errors
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args(
            [("ids", ids), ("delete_all", delete_all), ("namespace", namespace), ("filter", filter)]
        )

        return await self._vector_api.delete_vectors(
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

    @validate_and_convert_errors
    async def fetch(
        self, ids: List[str], namespace: Optional[str] = None, **kwargs
    ) -> FetchResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])
        return await self._vector_api.fetch_vectors(ids=ids, **args_dict, **kwargs)

    @validate_and_convert_errors
    async def query(
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
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryResponse:
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
        return await self._vector_api.query_vectors(
            request, **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )

    @validate_and_convert_errors
    async def query_namespaces(
        self,
        namespaces: List[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        vector: Optional[List[float]] = None,
        sparse_vector: Optional[
            Union[SparseValues, Dict[str, Union[List[float], List[int]]]]
        ] = None,
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

        for task in asyncio.as_completed(tasks):
            raw_result = await task
            response = json.loads(raw_result.data.decode("utf-8"))
            aggregator.add_results(response)

        final_results = aggregator.get_results()
        return final_results

    @validate_and_convert_errors
    async def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return await self._vector_api.update_vector(
            IndexRequestFactory.update_request(
                id=id,
                values=values,
                set_metadata=set_metadata,
                namespace=namespace,
                sparse_values=sparse_values,
                **kwargs,
            ),
            **self._openapi_kwargs(kwargs),
        )

    @validate_and_convert_errors
    async def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        return await self._vector_api.describe_index_stats(
            IndexRequestFactory.describe_index_stats_request(filter, **kwargs),
            **self._openapi_kwargs(kwargs),
        )

    @validate_and_convert_errors
    async def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> ListResponse:
        args_dict = IndexRequestFactory.list_paginated_args(
            prefix=prefix,
            limit=limit,
            pagination_token=pagination_token,
            namespace=namespace,
            **kwargs,
        )
        return await self._vector_api.list_vectors(**args_dict, **kwargs)

    @validate_and_convert_errors
    async def list(self, **kwargs):
        done = False
        while not done:
            results = await self.list_paginated(**kwargs)
            if len(results.vectors) > 0:
                yield [v.id for v in results.vectors]

            if results.pagination:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    async def upsert_records(self, namespace: str, records: List[Dict]):
        args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=records)
        await self._vector_api.upsert_records_namespace(**args)

    async def search(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        if namespace is None:
            raise Exception("Namespace is required when searching records")

        request = IndexRequestFactory.search_request(query=query, rerank=rerank, fields=fields)

        return await self._vector_api.search_records_namespace(namespace, request)

    async def search_records(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        return await self.search(namespace, query=query, rerank=rerank, fields=fields)

    def _openapi_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return filter_dict(kwargs, OPENAPI_ENDPOINT_PARAMS)
