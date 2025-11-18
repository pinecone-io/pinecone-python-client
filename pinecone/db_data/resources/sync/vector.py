from __future__ import annotations

from pinecone.utils.tqdm import tqdm
import logging
from typing import Any, Literal

import orjson
from multiprocessing.pool import ApplyResult
from concurrent.futures import as_completed

from pinecone.core.openapi.db_data.api.vector_operations_api import VectorOperationsApi
from pinecone.core.openapi.db_data.models import (
    QueryResponse as OpenAPIQueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    ListResponse,
)
from pinecone.db_data.dataclasses import (
    Vector,
    SparseValues,
    FetchResponse,
    FetchByMetadataResponse,
    Pagination,
    QueryResponse,
    UpsertResponse,
    UpdateResponse,
)
from pinecone.db_data.request_factory import IndexRequestFactory
from pinecone.db_data.types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    FilterTypedDict,
)
from pinecone.utils import (
    validate_and_convert_errors,
    filter_dict,
    parse_non_empty_args,
    PluginAware,
)
from pinecone.db_data.query_results_aggregator import QueryResultsAggregator, QueryNamespacesResults
from pinecone.openapi_support import OPENAPI_ENDPOINT_PARAMS

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


class VectorResource(PluginAware):
    """Resource for vector operations on a Pinecone index."""

    def __init__(self, vector_api: VectorOperationsApi, config, openapi_config, pool_threads: int):
        self._vector_api = vector_api
        """ :meta private: """
        self._config = config
        """ :meta private: """
        self._openapi_config = openapi_config
        """ :meta private: """
        self._pool_threads = pool_threads
        """ :meta private: """
        super().__init__()

    def _openapi_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return filter_dict(kwargs, OPENAPI_ENDPOINT_PARAMS)

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
        """Upsert vectors into the index.

        The upsert operation writes vectors into a namespace. If a new value is upserted
        for an existing vector id, it will overwrite the previous value.

        Args:
            vectors: A list of vectors to upsert. Each vector can be a Vector object,
                tuple, or dictionary.
            namespace: The namespace to write to. If not specified, the default namespace
                is used. [optional]
            batch_size: The number of vectors to upsert in each batch. If not specified,
                all vectors will be upserted in a single batch. [optional]
            show_progress: Whether to show a progress bar using tqdm. Applied only if
                batch_size is provided. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse containing the number of vectors upserted, or ApplyResult if
            async_req=True.

        Examples:
            >>> index.vector.upsert(
            ...     vectors=[
            ...         ('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
            ...         ('id2', [1.0, 2.0, 3.0])
            ...     ],
            ...     namespace='ns1'
            ... )
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
                from typing import cast

                return cast(UpsertResponse, UpsertResponseTransformer(result))  # type: ignore[arg-type]
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
            from typing import cast

            return cast(UpsertResponse, result)  # ApplyResult is not tracked through OpenAPI layers

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
        """Upsert vectors from a pandas DataFrame.

        Args:
            df: A pandas DataFrame with vector data.
            namespace: The namespace to write to. If not specified, the default namespace
                is used. [optional]
            batch_size: The number of rows to upsert in each batch. Default is 500.
            show_progress: Whether to show a progress bar. Default is True.

        Returns:
            UpsertResponse containing the number of vectors upserted.

        Raises:
            RuntimeError: If pandas is not installed.
            ValueError: If df is not a pandas DataFrame.
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
            # res is always UpsertResponse when not using async_req
            # upsert() doesn't use async_req, so res is always UpsertResponse
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

    @validate_and_convert_errors
    def delete(
        self,
        ids: list[str] | None = None,
        delete_all: bool | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Delete vectors from the index.

        The Delete operation deletes vectors from the index, from a single namespace.
        No error is raised if the vector id does not exist.

        Args:
            ids: Vector ids to delete. [optional]
            delete_all: If True, all vectors in the index namespace will be deleted.
                Default is False. [optional]
            namespace: The namespace to delete vectors from. If not specified, the default
                namespace is used. [optional]
            filter: Metadata filter expression to select vectors to delete. This is mutually
                exclusive with specifying ids or using delete_all=True. [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            Dict containing the deletion response.

        Examples:
            >>> index.vector.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.vector.delete(delete_all=True, namespace='my_namespace')
            >>> index.vector.delete(filter={'key': 'value'}, namespace='my_namespace')
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
        """Fetch vectors by ID.

        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Args:
            ids: The vector IDs to fetch.
            namespace: The namespace to fetch vectors from. If not specified, the default
                namespace is used. [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            FetchResponse object containing the fetched vectors and namespace name.

        Examples:
            >>> index.vector.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.vector.fetch(ids=['id1', 'id2'])
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
            namespace: The namespace to fetch vectors from. If not specified, the default
                namespace is used. [optional]
            limit: Max number of vectors to return. Defaults to 100. [optional]
            pagination_token: Pagination token to continue a previous listing operation.
                [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            FetchByMetadataResponse: Object containing the fetched vectors, namespace,
                usage, and pagination token.

        Examples:
            >>> index.vector.fetch_by_metadata(
            ...     filter={'genre': {'$in': ['comedy', 'drama']}, 'year': {'$eq': 2019}},
            ...     namespace='my_namespace',
            ...     limit=50
            ... )
            >>> index.vector.fetch_by_metadata(
            ...     filter={'status': 'active'},
            ...     pagination_token='token123'
            ... )
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
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> QueryResponse | ApplyResult:
        """Query the index.

        The Query operation searches a namespace, using a query vector. It retrieves the
        ids of the most similar items in a namespace, along with their similarity scores.

        Args:
            top_k: The number of results to return for each query. Must be an integer
                greater than 1.
            vector: The query vector. This should be the same length as the dimension of
                the index being queried. Each query request can contain only one of the
                parameters id or vector. [optional]
            id: The unique ID of the vector to be used as a query vector. Each query request
                can contain only one of the parameters vector or id. [optional]
            namespace: The namespace to query. If not specified, the default namespace is
                used. [optional]
            filter: The filter to apply. You can use vector metadata to limit your search.
                See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`
                [optional]
            include_values: Indicates whether vector values are included in the response.
                If omitted the server will use the default value of False. [optional]
            include_metadata: Indicates whether metadata is included in the response as well
                as the ids. If omitted the server will use the default value of False.
                [optional]
            sparse_vector: Sparse values of the query vector. Expected to be either a
                SparseValues object or a dict of the form {'indices': list[int],
                'values': list[float]}, where the lists each have the same length.
                [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            QueryResponse object which contains the list of the closest vectors as
            ScoredVector objects, and namespace name, or ApplyResult if async_req=True.

        Examples:
            >>> index.vector.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.vector.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.vector.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace',
            ...                    filter={'key': 'value'})
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
            # parse_query_response already returns QueryResponse
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
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
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
        sparse_vector: (SparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        """Query across multiple namespaces.

        Performs a query operation across multiple namespaces and aggregates the results.

        Args:
            vector: The query vector. [optional]
            namespaces: List of namespace names to query.
            metric: The similarity metric to use for aggregation. Must be one of "cosine",
                "euclidean", or "dotproduct".
            top_k: The number of results to return. If not specified, defaults to 10.
                [optional]
            filter: The filter to apply. You can use vector metadata to limit your search.
                [optional]
            include_values: Indicates whether vector values are included in the response.
                [optional]
            include_metadata: Indicates whether metadata is included in the response.
                [optional]
            sparse_vector: Sparse values of the query vector. [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            QueryNamespacesResults containing aggregated results from all namespaces.

        Raises:
            ValueError: If no namespaces are specified or if vector is empty.

        Examples:
            >>> index.vector.query_namespaces(
            ...     vector=[1, 2, 3],
            ...     namespaces=['ns1', 'ns2'],
            ...     metric='cosine',
            ...     top_k=10
            ... )
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

        # async_futures is list[QueryResponse | ApplyResult]
        # When async_threadpool_executor=True, query returns ApplyResult
        # as_completed expects Iterable[Future], so we need to cast
        futures: list[Future[Any]] = cast(list[Future[Any]], async_futures)
        for result in as_completed(futures):
            raw_result = result.result()
            response = orjson.loads(raw_result.data)
            aggregator.add_results(response)

        final_results = aggregator.get_results()
        return final_results

    @validate_and_convert_errors
    def update(
        self,
        id: str,
        values: list[float] | None = None,
        set_metadata: VectorMetadataTypedDict | None = None,
        namespace: str | None = None,
        sparse_values: (SparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> UpdateResponse:
        """Update a vector in the index.

        The Update operation updates vector in a namespace. If a value is included, it
        will overwrite the previous value. If a set_metadata is included, the values of
        the fields specified in it will be added or overwrite the previous value.

        Args:
            id: Vector's unique id.
            values: Vector values to set. [optional]
            set_metadata: Metadata to set for vector. [optional]
            namespace: Namespace name where to update the vector. If not specified, the
                default namespace is used. [optional]
            sparse_values: Sparse values to update for the vector. Expected to be either
                a SparseValues object or a dict of the form {'indices': list[int],
                'values': list[float]} where the lists each have the same length.
                [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            UpdateResponse (contains no data).

        Examples:
            >>> index.vector.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.vector.update(id='id1', set_metadata={'key': 'value'},
            ...                     namespace='my_namespace')
        """
        result = self._vector_api.update_vector(
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
        # Extract response info from result if it's an OpenAPI model with _response_info
        response_info = None
        if hasattr(result, "_response_info"):
            response_info = result._response_info
        else:
            # If result is a dict or empty, create default response_info
            from pinecone.utils.response_info import extract_response_info

            response_info = extract_response_info({})

        return UpdateResponse(_response_info=response_info)

    @validate_and_convert_errors
    def describe_index_stats(
        self, filter: FilterTypedDict | None = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """Describe index statistics.

        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Args:
            filter: If this parameter is present, the operation only returns statistics
                for vectors that satisfy the filter. See `metadata filtering
                <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            DescribeIndexStatsResponse object which contains stats about the index.

        Examples:
            >>> index.vector.describe_index_stats()
            >>> index.vector.describe_index_stats(filter={'key': 'value'})
        """
        from typing import cast

        result = self._vector_api.describe_index_stats(
            IndexRequestFactory.describe_index_stats_request(filter, **kwargs),
            **self._openapi_kwargs(kwargs),
        )
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
        """List vectors with pagination.

        The list_paginated operation finds vectors based on an id prefix within a single
        namespace. It returns matching ids in a paginated form, with a pagination token to
        fetch the next page of results.

        Args:
            prefix: The id prefix to match. If unspecified, an empty string prefix will
                be used with the effect of listing all ids in a namespace. [optional]
            limit: The maximum number of ids to return. If unspecified, the server will
                use a default value. [optional]
            pagination_token: A token needed to fetch the next page of results. This token
                is returned in the response if additional results are available. [optional]
            namespace: The namespace to list vectors from. If not specified, the default
                namespace is used. [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            ListResponse object which contains the list of ids, the namespace name,
            pagination information, and usage showing the number of read_units consumed.

        Examples:
            >>> results = index.vector.list_paginated(prefix='99', limit=5,
            ...                                       namespace='my_namespace')
            >>> results.pagination.next
            'eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9'
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
        return cast(ListResponse, result)

    @validate_and_convert_errors
    def list(self, **kwargs):
        """List vectors.

        The list operation accepts all of the same arguments as list_paginated, and returns
        a generator that yields a list of the matching vector ids in each page of results.
        It automatically handles pagination tokens on your behalf.

        Args:
            **kwargs: Same arguments as list_paginated (prefix, limit, pagination_token,
                namespace).

        Yields:
            List of vector ids for each page of results.

        Examples:
            >>> for ids in index.vector.list(prefix='99', limit=5,
            ...                              namespace='my_namespace'):
            ...     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
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
