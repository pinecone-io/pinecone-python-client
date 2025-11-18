from __future__ import annotations

import logging
from typing import Any, Iterable, cast, Literal


from pinecone.utils.tqdm import tqdm
from concurrent.futures import as_completed, Future

from ..utils import (
    dict_to_proto_struct,
    parse_fetch_response,
    parse_fetch_by_metadata_response,
    parse_query_response,
    query_response_to_dict,
    parse_stats_response,
    parse_upsert_response,
    parse_update_response,
    parse_delete_response,
)
from ..vector_factory_grpc import VectorFactoryGRPC
from ..sparse_values_factory import SparseValuesFactory

from pinecone.core.openapi.db_data.models import IndexDescription as DescribeIndexStatsResponse
from pinecone.db_data.dataclasses import (
    FetchByMetadataResponse,
    UpdateResponse,
    UpsertResponse,
    FetchResponse,
    QueryResponse,
)
from pinecone.db_control.models.list_response import ListResponse as SimpleListResponse, Pagination
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    Vector as GRPCVector,
    QueryResponse as ProtoQueryResponse,
    UpsertRequest,
    DeleteRequest,
    QueryRequest,
    FetchRequest,
    FetchByMetadataRequest,
    UpdateRequest,
    ListRequest,
    DescribeIndexStatsRequest,
    SparseValues as GRPCSparseValues,
)
from pinecone import Vector, SparseValues
from pinecone.db_data.query_results_aggregator import QueryNamespacesResults, QueryResultsAggregator
from ..future import PineconeGrpcFuture
from ...db_data.types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorTuple,
    FilterTypedDict,
    VectorMetadataTypedDict,
)
from ...utils import PluginAware

logger = logging.getLogger(__name__)
""" :meta private: """


class VectorResourceGRPC(PluginAware):
    """Resource for vector operations on a Pinecone index (GRPC)."""

    def __init__(self, stub, runner, threadpool_executor):
        self._stub = stub
        """ :meta private: """
        self._runner = runner
        """ :meta private: """
        self._threadpool_executor = threadpool_executor
        """ :meta private: """
        super().__init__()

    @staticmethod
    def _parse_non_empty_args(args: list[tuple[str, Any]]) -> dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}

    def upsert(
        self,
        vectors: list[Vector] | list[GRPCVector] | list[VectorTuple] | list[VectorTypedDict],
        async_req: bool = False,
        namespace: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse | PineconeGrpcFuture:
        """Upsert vectors into the index.

        The upsert operation writes vectors into a namespace. If a new value is upserted
        for an existing vector id, it will overwrite the previous value.

        Args:
            vectors: A list of vectors to upsert. Each vector can be a GRPCVector object,
                tuple, or dictionary.
            async_req: If True, the upsert operation will be performed asynchronously.
                Cannot be used with batch_size. Defaults to False.
            namespace: The namespace to write to. If not specified, the default namespace
                is used. [optional]
            batch_size: The number of vectors to upsert in each batch. Cannot be used
                with async_req=True. If not specified, all vectors will be upserted in
                a single batch. [optional]
            show_progress: Whether to show a progress bar using tqdm. Applied only if
                batch_size is provided. Default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse containing the number of vectors upserted, or
            PineconeGrpcFuture if async_req=True.

        Examples:
            >>> index.vector.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
            ...                      ('id2', [1.0, 2.0, 3.0])],
            ...                     namespace='ns1', async_req=True)
        """
        if async_req and batch_size is not None:
            raise ValueError(
                "async_req is not supported when batch_size is provided."
                "To upsert in parallel, please follow: "
                "https://docs.pinecone.io/docs/performance-tuning"
            )

        timeout = kwargs.pop("timeout", None)

        vectors = list(map(VectorFactoryGRPC.build, vectors))
        if async_req:
            args_dict = self._parse_non_empty_args([("namespace", namespace)])
            request = UpsertRequest(vectors=vectors, **args_dict, **kwargs)
            future_result = self._runner.run(self._stub.Upsert.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            # The future itself will provide metadata when it completes
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_upsert_response
            )

        if batch_size is None:
            return self._upsert_batch(vectors, namespace, timeout=timeout, **kwargs)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        pbar = tqdm(total=len(vectors), disable=not show_progress, desc="Upserted vectors")
        total_upserted = 0
        last_batch_result = None
        for i in range(0, len(vectors), batch_size):
            batch_result = self._upsert_batch(
                vectors[i : i + batch_size], namespace, timeout=timeout, **kwargs
            )
            pbar.update(batch_result.upserted_count)
            # we can't use here pbar.n for the case show_progress=False
            total_upserted += batch_result.upserted_count
            last_batch_result = batch_result

        # Create aggregated response with metadata from final batch
        from pinecone.db_data.dataclasses import UpsertResponse

        response_info = None
        if last_batch_result and hasattr(last_batch_result, "_response_info"):
            response_info = last_batch_result._response_info
        else:
            from pinecone.utils.response_info import extract_response_info

            response_info = extract_response_info({})

        return UpsertResponse(upserted_count=total_upserted, _response_info=response_info)

    def _upsert_batch(
        self, vectors: list[GRPCVector], namespace: str | None, timeout: int | None, **kwargs
    ) -> UpsertResponse:
        args_dict = self._parse_non_empty_args([("namespace", namespace)])
        request = UpsertRequest(vectors=vectors, **args_dict)
        response, initial_metadata = self._runner.run(
            self._stub.Upsert, request, timeout=timeout, **kwargs
        )
        return parse_upsert_response(response, initial_metadata=initial_metadata)

    def upsert_from_dataframe(
        self,
        df,
        namespace: str | None = None,
        batch_size: int = 500,
        use_async_requests: bool = True,
        show_progress: bool = True,
    ) -> UpsertResponse:
        """Upsert vectors from a pandas DataFrame.

        Args:
            df: A pandas DataFrame with vector data.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            use_async_requests: Whether to upsert multiple requests at the same time
                using asynchronous request mechanism.
            show_progress: Whether to show a progress bar.

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
            res = self.upsert(vectors=chunk, namespace=namespace, async_req=use_async_requests)
            pbar.update(len(chunk))
            results.append(res)

        if use_async_requests:
            cast_results = cast(list[PineconeGrpcFuture], results)
            results = [
                async_result.result()
                for async_result in tqdm(
                    iterable=cast_results,
                    disable=not show_progress,
                    desc="collecting async responses",
                )
            ]

        upserted_count = 0
        last_result = None
        for res in results:
            if hasattr(res, "upserted_count") and isinstance(res.upserted_count, int):
                upserted_count += res.upserted_count
                last_result = res

        response_info = None
        if last_result and hasattr(last_result, "_response_info"):
            response_info = last_result._response_info
        else:
            from pinecone.utils.response_info import extract_response_info

            response_info = extract_response_info({})

        return UpsertResponse(upserted_count=upserted_count, _response_info=response_info)

    @staticmethod
    def _iter_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size].to_dict(orient="records")
            yield batch

    def delete(
        self,
        ids: list[str] | None = None,
        delete_all: bool | None = None,
        namespace: str | None = None,
        filter: FilterTypedDict | None = None,
        async_req: bool = False,
        **kwargs,
    ) -> dict[str, Any] | PineconeGrpcFuture:
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
            async_req: If True, the delete operation will be performed asynchronously.
                Defaults to False. [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            DeleteResponse (contains no data) or a PineconeGrpcFuture object if
            async_req is True.

        Examples:
            >>> index.vector.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.vector.delete(delete_all=True, namespace='my_namespace')
            >>> index.vector.delete(filter={'key': 'value'}, namespace='my_namespace', async_req=True)
        """
        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        args_dict = self._parse_non_empty_args(
            [
                ("ids", ids),
                ("delete_all", delete_all),
                ("namespace", namespace),
                ("filter", filter_struct),
            ]
        )
        timeout = kwargs.pop("timeout", None)

        request = DeleteRequest(**args_dict, **kwargs)
        if async_req:
            future_result = self._runner.run(self._stub.Delete.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_delete_response
            )
        else:
            response, initial_metadata = self._runner.run(
                self._stub.Delete, request, timeout=timeout
            )
            return parse_delete_response(response, initial_metadata=initial_metadata)

    def fetch(
        self,
        ids: list[str] | None,
        namespace: str | None = None,
        async_req: bool | None = False,
        **kwargs,
    ) -> FetchResponse | PineconeGrpcFuture:
        """Fetch vectors by ID.

        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Args:
            ids: The vector IDs to fetch.
            namespace: The namespace to fetch vectors from. If not specified, the default
                namespace is used. [optional]
            async_req: If True, the fetch operation will be performed asynchronously.
                Defaults to False. [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            FetchResponse object which contains the list of Vector objects, and namespace name.

        Examples:
            >>> index.vector.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.vector.fetch(ids=['id1', 'id2'])
        """
        timeout = kwargs.pop("timeout", None)

        args_dict = self._parse_non_empty_args([("namespace", namespace)])

        request = FetchRequest(ids=ids, **args_dict, **kwargs)

        if async_req:
            future_result = self._runner.run(self._stub.Fetch.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, result_transformer=parse_fetch_response, timeout=timeout
            )
        else:
            response, initial_metadata = self._runner.run(
                self._stub.Fetch, request, timeout=timeout
            )
            return parse_fetch_response(response, initial_metadata=initial_metadata)

    def fetch_by_metadata(
        self,
        filter: FilterTypedDict,
        namespace: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        async_req: bool | None = False,
        **kwargs,
    ) -> FetchByMetadataResponse | PineconeGrpcFuture:
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
            async_req: If True, the fetch operation will be performed asynchronously.
                Defaults to False. [optional]
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
        timeout = kwargs.pop("timeout", None)

        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        args_dict = self._parse_non_empty_args(
            [
                ("namespace", namespace),
                ("filter", filter_struct),
                ("limit", limit),
                ("pagination_token", pagination_token),
            ]
        )

        request = FetchByMetadataRequest(**args_dict, **kwargs)

        if async_req:
            future_result = self._runner.run(
                self._stub.FetchByMetadata.future, request, timeout=timeout
            )
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, result_transformer=parse_fetch_by_metadata_response, timeout=timeout
            )
        else:
            response, initial_metadata = self._runner.run(
                self._stub.FetchByMetadata, request, timeout=timeout
            )
            return parse_fetch_by_metadata_response(response, initial_metadata=initial_metadata)

    def _query(
        self,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        top_k: int | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: (SparseValues | GRPCSparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> tuple[ProtoQueryResponse, dict[str, str] | None]:
        """
        Low-level query method that returns protobuf Message and initial metadata without parsing.
        Used internally by query() and query_namespaces() for performance.

        Returns:
            Tuple of (protobuf_message, initial_metadata). initial_metadata may be None.
        """
        if vector is not None and id is not None:
            raise ValueError("Cannot specify both `id` and `vector`")

        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        sparse_vector = SparseValuesFactory.build(sparse_vector)
        args_dict = self._parse_non_empty_args(
            [
                ("vector", vector),
                ("id", id),
                ("namespace", namespace),
                ("top_k", top_k),
                ("filter", filter_struct),
                ("include_values", include_values),
                ("include_metadata", include_metadata),
                ("sparse_vector", sparse_vector),
            ]
        )

        request = QueryRequest(**args_dict)

        timeout = kwargs.pop("timeout", None)
        response, initial_metadata = self._runner.run(self._stub.Query, request, timeout=timeout)
        return response, initial_metadata

    def query(
        self,
        vector: list[float] | None = None,
        id: str | None = None,
        namespace: str | None = None,
        top_k: int | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: (SparseValues | GRPCSparseValues | SparseVectorTypedDict) | None = None,
        async_req: bool | None = False,
        **kwargs,
    ) -> "QueryResponse" | PineconeGrpcFuture:
        """Query the index.

        The Query operation searches a namespace, using a query vector. It retrieves the
        ids of the most similar items in a namespace, along with their similarity scores.

        Args:
            vector: The query vector. This should be the same length as the dimension of
                the index being queried. Each query request can contain only one of the
                parameters id or vector. [optional]
            id: The unique ID of the vector to be used as a query vector. Each query request
                can contain only one of the parameters vector or id. [optional]
            top_k: The number of results to return for each query. Must be an integer
                greater than 1.
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
            async_req: If True, the query operation will be performed asynchronously.
                Defaults to False. [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            QueryResponse object which contains the list of the closest vectors as
            ScoredVector objects, and namespace name, or PineconeGrpcFuture if
            async_req=True.

        Examples:
            >>> index.vector.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.vector.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.vector.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace',
            ...                    filter={'key': 'value'})
        """
        timeout = kwargs.pop("timeout", None)

        if async_req:
            # For async requests, we need to build the request manually
            if vector is not None and id is not None:
                raise ValueError("Cannot specify both `id` and `vector`")

            if filter is not None:
                filter_struct = dict_to_proto_struct(filter)
            else:
                filter_struct = None

            sparse_vector = SparseValuesFactory.build(sparse_vector)
            args_dict = self._parse_non_empty_args(
                [
                    ("vector", vector),
                    ("id", id),
                    ("namespace", namespace),
                    ("top_k", top_k),
                    ("filter", filter_struct),
                    ("include_values", include_values),
                    ("include_metadata", include_metadata),
                    ("sparse_vector", sparse_vector),
                ]
            )

            request = QueryRequest(**args_dict)
            future_result = self._runner.run(self._stub.Query.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, result_transformer=parse_query_response, timeout=timeout
            )
        else:
            # For sync requests, use _query to get protobuf Message and metadata, then parse it
            response, initial_metadata = self._query(
                vector=vector,
                id=id,
                namespace=namespace,
                top_k=top_k,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                timeout=timeout,
                **kwargs,
            )
            return parse_query_response(
                response, _check_type=False, initial_metadata=initial_metadata
            )

    def query_namespaces(
        self,
        vector: list[float],
        namespaces: list[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: int | None = None,
        filter: FilterTypedDict | None = None,
        include_values: bool | None = None,
        include_metadata: bool | None = None,
        sparse_vector: (GRPCSparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        """Query across multiple namespaces.

        Performs a query operation across multiple namespaces and aggregates the results.

        Args:
            vector: The query vector.
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
        if len(vector) == 0:
            raise ValueError("Query vector must not be empty")

        overall_topk = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(top_k=overall_topk, metric=metric)

        target_namespaces = set(namespaces)  # dedup namespaces
        futures = [
            self._threadpool_executor.submit(
                self._query,
                vector=vector,
                namespace=ns,
                top_k=overall_topk,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                **kwargs,
            )
            for ns in target_namespaces
        ]

        only_futures = cast(Iterable[Future], futures)
        for response in as_completed(only_futures):
            proto_response, _ = response.result()  # Ignore initial_metadata for query_namespaces
            # Convert protobuf Message to dict format for aggregator using optimized helper
            json_response = query_response_to_dict(proto_response)
            aggregator.add_results(json_response)

        final_results = aggregator.get_results()
        return final_results

    def update(
        self,
        id: str,
        async_req: bool = False,
        values: list[float] | None = None,
        set_metadata: VectorMetadataTypedDict | None = None,
        namespace: str | None = None,
        sparse_values: (GRPCSparseValues | SparseVectorTypedDict) | None = None,
        **kwargs,
    ) -> UpdateResponse | PineconeGrpcFuture:
        """Update a vector in the index.

        The Update operation updates vector in a namespace. If a value is included, it
        will overwrite the previous value. If a set_metadata is included, the values of
        the fields specified in it will be added or overwrite the previous value.

        Args:
            id: Vector's unique id.
            async_req: If True, the update operation will be performed asynchronously.
                Defaults to False.
            values: Vector values to set. [optional]
            set_metadata: Metadata to set for vector. [optional]
            namespace: Namespace name where to update the vector. If not specified, the
                default namespace is used. [optional]
            sparse_values: Sparse values to update for the vector. Expected to be either
                a GRPCSparseValues object or a dict of the form {'indices': list[int],
                'values': list[float]} where the lists each have the same length.
                [optional]
            **kwargs: Additional keyword arguments.

        Returns:
            UpdateResponse (contains no data), or PineconeGrpcFuture if async_req=True.

        Examples:
            >>> index.vector.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.vector.update(id='id1', set_metadata={'key': 'value'},
            ...                     namespace='my_namespace')
        """
        timeout = kwargs.pop("timeout", None)

        sparse_values = SparseValuesFactory.build(sparse_values)
        args_dict = self._parse_non_empty_args(
            [
                ("id", id),
                ("values", values),
                ("set_metadata", dict_to_proto_struct(set_metadata) if set_metadata else None),
                ("namespace", namespace),
                ("sparse_values", sparse_values),
            ]
        )

        request = UpdateRequest(**args_dict, **kwargs)

        if async_req:
            future_result = self._runner.run(self._stub.Update.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_update_response
            )
        else:
            response, initial_metadata = self._runner.run(
                self._stub.Update, request, timeout=timeout
            )
            return parse_update_response(response, initial_metadata=initial_metadata)

    def list_paginated(
        self,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str | None = None,
        **kwargs,
    ) -> SimpleListResponse:
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
            SimpleListResponse object which contains the list of ids, the namespace name,
            pagination information, and usage showing the number of read_units consumed.

        Examples:
            >>> results = index.vector.list_paginated(prefix='99', limit=5,
            ...                                       namespace='my_namespace')
            >>> results.pagination.next
            'eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9'
        """
        args_dict = self._parse_non_empty_args(
            [
                ("prefix", prefix),
                ("limit", limit),
                ("namespace", namespace),
                ("pagination_token", pagination_token),
            ]
        )
        request = ListRequest(**args_dict, **kwargs)
        timeout = kwargs.pop("timeout", None)
        response, _ = self._runner.run(self._stub.List, request, timeout=timeout)

        if response.pagination and response.pagination.next != "":
            pagination = Pagination(next=response.pagination.next)
        else:
            pagination = None

        return SimpleListResponse(
            namespace=response.namespace, vectors=response.vectors, pagination=pagination
        )

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
            try:
                results = self.list_paginated(**kwargs)
            except Exception as e:
                raise e

            if len(results.vectors) > 0:
                yield [v.id for v in results.vectors]

            if results.pagination and results.pagination.next:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

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
        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None
        args_dict = self._parse_non_empty_args([("filter", filter_struct)])
        timeout = kwargs.pop("timeout", None)

        request = DescribeIndexStatsRequest(**args_dict)
        response, _ = self._runner.run(self._stub.DescribeIndexStats, request, timeout=timeout)
        return parse_stats_response(response)
