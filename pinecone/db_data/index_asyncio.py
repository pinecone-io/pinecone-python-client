from __future__ import annotations

from pinecone.utils.tqdm import tqdm


import logging
import asyncio
import json

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
from pinecone.adapters.response_adapters import adapt_query_response, adapt_fetch_response
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
    """:meta private:

    Deprecated: Use adapt_query_response from pinecone.adapters instead.
    This function is kept for backward compatibility.
    """
    return adapt_query_response(response)


class _IndexAsyncio:
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
        """
        Args:
            vectors (Union[list[Vector], list[VectorTuple], list[VectorTupleWithMetadata], list[VectorTypedDict]]): A list of vectors to upsert.
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns:
            `UpsertResponse`, includes the number of vectors upserted.


        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        To upsert in parallel follow `this link <https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel>`_.

        **Upserting dense vectors**

        .. admonition:: Note

            The dimension of each dense vector must match the dimension of the index.

        A vector can be represented in a variety of ways.

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    # A Vector object
                    await idx.upsert(
                        namespace = 'my-namespace',
                        vectors = [
                            Vector(id='id1', values=[0.1, 0.2, 0.3, 0.4], metadata={'metadata_key': 'metadata_value'}),
                        ]
                    )

                    # A vector tuple
                    await idx.upsert(
                        namespace = 'my-namespace',
                        vectors = [
                            ('id1', [0.1, 0.2, 0.3, 0.4]),
                        ]
                    )

                    # A vector tuple with metadata
                    await idx.upsert(
                        namespace = 'my-namespace',
                        vectors = [
                            ('id1', [0.1, 0.2, 0.3, 0.4], {'metadata_key': 'metadata_value'}),
                        ]
                    )

                    # A vector dictionary
                    await idx.upsert(
                        namespace = 'my-namespace',
                        vectors = [
                            {"id": 1, "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"metadata_key": "metadata_value"}},
                        ]
                    )

            asyncio.run(main())


        **Upserting sparse vectors**

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    # A Vector object
                    await idx.upsert(
                        namespace = 'my-namespace',
                        vectors = [
                            Vector(id='id1', sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4])),
                        ]
                    )

                    # A dictionary
                    await idx.upsert(
                        namespace = 'my-namespace',
                        vectors = [
                            {"id": 1, "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]}},
                        ]
                    )

            asyncio.run(main())


        **Batch upsert**

        If you have a large number of vectors, you can upsert them in batches.

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:

                await idx.upsert(
                    namespace = 'my-namespace',
                    vectors = [
                        {'id': 'id1', 'values': [0.1, 0.2, 0.3, 0.4]},
                        {'id': 'id2', 'values': [0.2, 0.3, 0.4, 0.5]},
                        {'id': 'id3', 'values': [0.3, 0.4, 0.5, 0.6]},
                        {'id': 'id4', 'values': [0.4, 0.5, 0.6, 0.7]},
                        {'id': 'id5', 'values': [0.5, 0.6, 0.7, 0.8]},
                        # More vectors here
                    ],
                    batch_size = 50
                )

            asyncio.run(main())


        **Visual progress bar with tqdm**

        To see a progress bar when upserting in batches, you will need to separately install the `tqdm` package.
        If `tqdm` is present, the client will detect and use it to display progress when `show_progress=True`.
        """
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
        """This method has not been implemented yet for the IndexAsyncio class."""
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
        """
        Args:
            ids (list[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                                Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                            If not specified, the default namespace is used.
            filter (dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                    See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]


        The Delete operation deletes vectors from the index, from a single namespace.

        No error is raised if the vector id does not exist.

        Note: For any delete call, if namespace is not specified, the default namespace `""` is used.
        Since the delete operation does not error when ids are not present, this means you may not receive
        an error if you delete from the wrong namespace.

        Delete can occur in the following mutual exclusive ways:

        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
            (note that for this option delete all must be set to False)

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    # Delete specific ids
                    await idx.delete(
                        ids=['id1', 'id2'],
                        namespace='my_namespace'
                    )

                    # Delete everything in a namespace
                    await idx.delete(
                        delete_all=True,
                        namespace='my_namespace'
                    )

                    # Delete by metadata filter
                    await idx.delete(
                        filter={'key': 'value'},
                        namespace='my_namespace'
                    )

            asyncio.run(main())

        Returns: An empty dictionary if the delete operation was successful.
        """
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
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    # Fetch specific ids in namespace
                    fetched = await idx.fetch(
                        ids=['id1', 'id2'],
                        namespace='my_namespace'
                    )
                    for vec_id in fetched.vectors:
                        vector = fetched.vectors[vec_id]
                        print(vector.id)
                        print(vector.metadata)
                        print(vector.values)

            asyncio.run(main())

        Args:
            ids (list[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        args_dict = parse_non_empty_args([("namespace", namespace)])
        result = await self._vector_api.fetch_vectors(ids=ids, **args_dict, **kwargs)
        return adapt_fetch_response(result)

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
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        **Querying with dense vectors**

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    query_embedding = [0.1, 0.2, 0.3, ...] # An embedding that matches the index dimension

                    # Query by vector values
                    results = await idx.query(
                        vector=query_embedding,
                        top_k=10,
                        filter={'genre': {"$eq": "drama"}}, # Optionally filter by metadata
                        namespace='my_namespace',
                        include_values=False,
                        include_metadata=True
                    )

                    # Query using vector id (the values from this stored vector will be used to query)
                    results = await idx.query(
                        id='1',
                        top_k=10,
                        filter={"year": {"$gt": 2000}},
                        namespace='my_namespace',
                    )

            asyncio.run(main())


        **Query with sparse vectors**

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    query_embedding = [0.1, 0.2, 0.3, ...] # An embedding that matches the index dimension

                    # Query by vector values
                    results = await idx.query(
                        vector=query_embedding,
                        top_k=10,
                        filter={'genre': {"$eq": "drama"}}, # Optionally filter by metadata
                        namespace='my_namespace',
                        include_values=False,
                        include_metadata=True
                    )

                    # Query using vector id (the values from this stored vector will be used to query)
                    results = await idx.query(
                        id='1',
                        top_k=10,
                        filter={"year": {"$gt": 2000}},
                        namespace='my_namespace',
                    )

            asyncio.run(main())

        Examples:

        .. code-block:: python

            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=SparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (list[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (dict[str, Union[str, float, int, bool, List, dict]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, dict[str, Union[list[float], list[int]]]]): sparse values of the query vector.
                            Expected to be either a SparseValues object or a dict of the form:
                             {'indices': list[int], 'values': list[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
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
        """The query_namespaces() method is used to make a query to multiple namespaces in parallel and combine the results into one result set.

        Args:
            vector (list[float]): The query vector, must be the same length as the dimension of the index being queried.
            namespaces (list[str]): The list of namespaces to query.
            top_k (Optional[int], optional): The number of results you would like to request from each namespace. Defaults to 10.
            filter (Optional[dict[str, Union[str, float, int, bool, List, dict]]], optional): Pass an optional filter to filter results based on metadata. Defaults to None.
            include_values (Optional[bool], optional): Boolean field indicating whether vector values should be included with results. Defaults to None.
            include_metadata (Optional[bool], optional): Boolean field indicating whether vector metadata should be included with results. Defaults to None.
            sparse_vector (Optional[ Union[SparseValues, dict[str, Union[list[float], list[int]]]] ], optional): If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None.

        Returns:
            QueryNamespacesResults: A QueryNamespacesResults object containing the combined results from all namespaces, as well as the combined usage cost in read units.

        Examples:

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone

            async def main():
                pc = Pinecone(api_key="your-api-key")
                idx = pc.IndexAsyncio(
                    host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io",
                )

                query_vec = [0.1, 0.2, 0.3] # An embedding that matches the index dimension
                combined_results = await idx.query_namespaces(
                    vector=query_vec,
                    namespaces=['ns1', 'ns2', 'ns3', 'ns4'],
                    top_k=10,
                    filter={'genre': {"$eq": "drama"}},
                    include_values=True,
                    include_metadata=True
                )
                for vec in combined_results.matches:
                    print(vec.id, vec.score)
                print(combined_results.usage)

                await idx.close()

            asyncio.run(main())

        """
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
        """
        The Update operation updates vectors in a namespace.

        This method supports two update modes:

        1. **Single vector update by ID**: Provide `id` to update a specific vector.
           - Updates the vector with the given ID
           - If `values` is included, it will overwrite the previous vector values
           - If `set_metadata` is included, the metadata will be merged with existing metadata on the vector.
             Fields specified in `set_metadata` will overwrite existing fields with the same key, while
             fields not in `set_metadata` will remain unchanged.

        2. **Bulk update by metadata filter**: Provide `filter` to update all vectors matching the filter criteria.
           - Updates all vectors in the namespace that match the filter expression
           - Useful for updating metadata across multiple vectors at once
           - If `set_metadata` is included, the metadata will be merged with existing metadata on each vector.
             Fields specified in `set_metadata` will overwrite existing fields with the same key, while
             fields not in `set_metadata` will remain unchanged.
           - The response includes `matched_records` indicating how many vectors were updated

        Either `id` or `filter` must be provided (but not both in the same call).

        Examples:

        **Single vector update by ID:**

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    # Update vector values
                    await idx.update(
                        id='id1',
                        values=[0.1, 0.2, 0.3, ...],
                        namespace='my_namespace'
                    )

                    # Update metadata
                    await idx.update(
                        id='id1',
                        set_metadata={'key': 'value'},
                        namespace='my_namespace'
                    )

                    # Update sparse values
                    await idx.update(
                        id='id1',
                        sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
                        namespace='my_namespace'
                    )

                    # Update sparse values with SparseValues object
                    await idx.update(
                        id='id1',
                        sparse_values=SparseValues(indices=[234781, 5432], values=[0.2, 0.4]),
                        namespace='my_namespace'
                    )

        **Bulk update by metadata filter:**

        .. code-block:: python

                    # Update metadata for all vectors matching the filter
                    response = await idx.update(
                        set_metadata={'status': 'active'},
                        filter={'genre': {'$eq': 'drama'}},
                        namespace='my_namespace'
                    )
                    print(f"Updated {response.matched_records} vectors")
                    # Preview how many vectors would be updated (dry run)
                    response = await idx.update(
                        set_metadata={'status': 'active'},
                        filter={'genre': {'$eq': 'drama'}},
                        namespace='my_namespace',
                        dry_run=True
                    )
                    print(f"Would update {response.matched_records} vectors")

            asyncio.run(main())

        Args:
            id (str): Vector's unique id. Required for single vector updates. Must not be provided when using filter. [optional]
            values (list[float]): Vector values to set. [optional]
            set_metadata (dict[str, Union[str, float, int, bool, list[int], list[float], list[str]]]]):
                Metadata to merge with existing metadata on the vector(s). Fields specified will overwrite
                existing fields with the same key, while fields not specified will remain unchanged. [optional]
            namespace (str): Namespace name where to update the vector(s). [optional]
            sparse_values: (dict[str, Union[list[float], list[int]]]): Sparse values to update for the vector.
                           Expected to be either a SparseValues object or a dict of the form:
                           {'indices': list[int], 'values': list[float]} where the lists each have the same length. [optional]
            filter (dict[str, Union[str, float, int, bool, List, dict]]): A metadata filter expression.
                    When provided, updates all vectors in the namespace that match the filter criteria.
                    See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`.
                    Must not be provided when using id. Either `id` or `filter` must be provided. [optional]
            dry_run (bool): If `True`, return the number of records that match the `filter` without executing
                    the update. Only meaningful when using `filter` (not with `id`). Useful for previewing
                    the impact of a bulk update before applying changes. Defaults to `False`. [optional]

        Returns:
            UpdateResponse: An UpdateResponse object. When using filter-based updates, the response includes
            `matched_records` indicating the number of vectors that were updated (or would be updated if
            `dry_run=True`).
        """
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
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Args:
            filter (dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.

        .. code-block:: python

            import asyncio
            from pinecone import Pinecone, Vector, SparseValues

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    print(await idx.describe_index_stats())

            asyncio.run(main())

        """
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
        """
        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the `list` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace')
            >>> [v.id for v in results.vectors]
            ['99', '990', '991', '992', '993']
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace', pagination_token=results.pagination.next)

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]

        Returns: ListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
        """
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
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:
            >>> for ids in index.list(prefix='99', limit=5, namespace='my_namespace'):
            >>>     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
            ['999']

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
        """
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
        """
        :param namespace: The namespace of the index to upsert records to.
        :type namespace: str, required
        :param records: The records to upsert into the index.
        :type records: list[Dict], required

        Upsert records to a namespace. A record is a dictionary that contains eitiher an `id` or `_id`
        field along with other fields that will be stored as metadata. The `id` or `_id` field is used
        as the unique identifier for the record. At least one field in the record should correspond to
        a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and upserts them into
        the specified namespacce of the index.

        .. code-block:: python

            import asyncio
            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel
                IndexEmbed
            )

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    # upsert records
                    await idx.upsert_records(
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
                            {
                                "_id": "test4",
                                "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                            },
                            {
                                "_id": "test5",
                                "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
                            },
                            {
                                "_id": "test6",
                                "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
                            },
                        ],
                    )

                    from pinecone import SearchQuery, SearchRerank, RerankModel

                    # search for similar records
                    response = await idx.search_records(
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

            asyncio.run(main())

        """
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
        """
        :param namespace: The namespace in the index to search.
        :type namespace: str, required
        :param query: The SearchQuery to use for the search. The query can include a ``match_terms`` field
                      to specify which terms must be present in the text of each search hit. The match_terms
                      should be a dict with ``strategy`` (str) and ``terms`` (list[str]) keys, e.g.
                      ``{"strategy": "all", "terms": ["term1", "term2"]}``. Currently only "all" strategy
                      is supported, which means all specified terms must be present.
                      **Note:** match_terms is only supported for sparse indexes with integrated embedding
                      configured to use the pinecone-sparse-english-v0 model.
        :type query: Union[Dict, SearchQuery], required
        :param rerank: The SearchRerank to use with the search request.
        :type rerank: Union[Dict, SearchRerank], optional
        :return: The records that match the search.

        Search for records.

        This operation converts a query to a vector embedding and then searches a namespace. You
        can optionally provide a reranking operation as part of the search.

        .. code-block:: python

            import asyncio
            from pinecone import (
                Pinecone,
                CloudProvider,
                AwsRegion,
                EmbedModel
                IndexEmbed
            )

            async def main():
                pc = Pinecone()
                async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                    # upsert records
                    await idx.upsert_records(
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
                            {
                                "_id": "test4",
                                "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                            },
                            {
                                "_id": "test5",
                                "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
                            },
                            {
                                "_id": "test6",
                                "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
                            },
                        ],
                    )

                    from pinecone import SearchQuery, SearchRerank, RerankModel

                    # search for similar records
                    response = await idx.search_records(
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

            asyncio.run(main())

        """
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
        """Alias of the search() method."""
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
        """Create a namespace in a serverless index.

        Args:
            name (str): The name of the namespace to create
            schema (Optional[dict[str, Any]]): Optional schema configuration for the namespace as a dictionary. [optional]

        Returns:
            NamespaceDescription: Information about the created namespace including vector count

        Create a namespace in a serverless index. For guidance and examples, see
        `Manage namespaces <https://docs.pinecone.io/guides/manage-data/manage-namespaces>`_.

        **Note:** This operation is not supported for pod-based indexes.

        Examples:

            .. code-block:: python

                >>> # Create a namespace with just a name
                >>> import asyncio
                >>> from pinecone import Pinecone
                >>>
                >>> async def main():
                ...     pc = Pinecone()
                ...     async with pc.IndexAsyncio(host="example-index-dojoi3u.svc.eu-west1-gcp.pinecone.io") as idx:
                ...         namespace = await idx.create_namespace(name="my-namespace")
                ...         print(f"Created namespace: {namespace.name}, Vector count: {namespace.vector_count}")
                >>>
                >>> asyncio.run(main())

                >>> # Create a namespace with schema configuration
                >>> from pinecone.core.openapi.db_data.model.create_namespace_request_schema import CreateNamespaceRequestSchema
                >>> schema = CreateNamespaceRequestSchema(fields={...})
                >>> namespace = await idx.create_namespace(name="my-namespace", schema=schema)
        """
        return await self.namespace.create(name=name, schema=schema, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    async def describe_namespace(self, namespace: str, **kwargs) -> "NamespaceDescription":
        """Describe a namespace within an index, showing the vector count within the namespace.

        Args:
            namespace (str): The namespace to describe

        Returns:
            NamespaceDescription: Information about the namespace including vector count
        """
        return await self.namespace.describe(namespace=namespace, **kwargs)

    @validate_and_convert_errors
    @require_kwargs
    async def delete_namespace(self, namespace: str, **kwargs) -> dict[str, Any]:
        """Delete a namespace from an index.

        Args:
            namespace (str): The namespace to delete

        Returns:
            dict[str, Any]: Response from the delete operation
        """
        from typing import cast

        result = await self.namespace.delete(namespace=namespace, **kwargs)
        return cast(dict[str, Any], result)

    @validate_and_convert_errors
    @require_kwargs
    async def list_namespaces(
        self, limit: int | None = None, **kwargs
    ) -> AsyncIterator[ListNamespacesResponse]:
        """List all namespaces in an index. This method automatically handles pagination to return all results.

        Args:
            limit (Optional[int]): The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]

        Returns:
            ``ListNamespacesResponse``: Object containing the list of namespaces.

        Examples:
            .. code-block:: python
                >>> async for namespace in index.list_namespaces(limit=5):
                ...     print(f"Namespace: {namespace.name}, Vector count: {namespace.vector_count}")
                Namespace: namespace1, Vector count: 1000
                Namespace: namespace2, Vector count: 2000
        """
        async for namespace in self.namespace.list(limit=limit, **kwargs):
            yield namespace

    @validate_and_convert_errors
    @require_kwargs
    async def list_namespaces_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> ListNamespacesResponse:
        """List all namespaces in an index with pagination support. The response includes pagination information if there are more results available.

        Consider using the ``list_namespaces`` method to avoid having to handle pagination tokens manually.

        Args:
            limit (Optional[int]): The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns:
            ``ListNamespacesResponse``: Object containing the list of namespaces and pagination information.

        Examples:
            .. code-block:: python
                >>> results = await index.list_namespaces_paginated(limit=5)
                >>> results.pagination.next
                eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
                >>> next_results = await index.list_namespaces_paginated(limit=5, pagination_token=results.pagination.next)
        """
        return await self.namespace.list_paginated(
            limit=limit, pagination_token=pagination_token, **kwargs
        )


IndexAsyncio = _IndexAsyncio
