from __future__ import annotations

import logging
from typing import List, Any, Iterable, cast, Literal, Iterator, TYPE_CHECKING


from pinecone.utils.tqdm import tqdm
from pinecone.utils import require_kwargs
from concurrent.futures import as_completed, Future


from .utils import (
    dict_to_proto_struct,
    parse_fetch_response,
    parse_fetch_by_metadata_response,
    parse_query_response,
    query_response_to_dict,
    parse_stats_response,
    parse_upsert_response,
    parse_update_response,
    parse_delete_response,
    parse_namespace_description,
    parse_list_namespaces_response,
)
from .vector_factory_grpc import VectorFactoryGRPC
from .sparse_values_factory import SparseValuesFactory

from pinecone.core.openapi.db_data.models import (
    IndexDescription as DescribeIndexStatsResponse,
    NamespaceDescription,
    ListNamespacesResponse,
)
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
    QueryVector as GRPCQueryVector,
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
    DescribeNamespaceRequest,
    DeleteNamespaceRequest,
    ListNamespacesRequest,
    CreateNamespaceRequest,
    MetadataSchema,
    MetadataFieldProperties,
)
from pinecone.core.grpc.protos.db_data_2025_10_pb2_grpc import VectorServiceStub
from pinecone import Vector, SparseValues
from pinecone.db_data.query_results_aggregator import QueryNamespacesResults, QueryResultsAggregator
from .base import GRPCIndexBase
from .future import PineconeGrpcFuture

if TYPE_CHECKING:
    from typing import Type
from ..db_data.types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorTuple,
    FilterTypedDict,
    VectorMetadataTypedDict,
)


__all__ = [
    "GRPCIndex",
    "GRPCVector",
    "GRPCQueryVector",
    "GRPCSparseValues",
    "NamespaceDescription",
    "ListNamespacesResponse",
]

_logger = logging.getLogger(__name__)
""" :meta private: """


class GRPCIndex(GRPCIndexBase):
    """A client for interacting with a Pinecone index via GRPC API."""

    @property
    def stub_class(self) -> "Type[VectorServiceStub]":
        """:meta private:"""
        return VectorServiceStub

    def upsert(
        self,
        vectors: list[Vector] | list[GRPCVector] | list[VectorTuple] | list[VectorTypedDict],
        async_req: bool = False,
        namespace: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse | PineconeGrpcFuture:
        """
        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        Examples:

        .. code-block:: python

            >>> index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}),
                              ('id2', [1.0, 2.0, 3.0])
                              ],
                              namespace='ns1', async_req=True)
            >>> index.upsert([{'id': 'id1', 'values': [1.0, 2.0, 3.0], 'metadata': {'key': 'value'}},
                              {'id': 'id2',
                                        'values': [1.0, 2.0, 3.0],
                                        'sparse_values': {'indices': [1, 8], 'values': [0.2, 0.4]},
                              ])
            >>> index.upsert([GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                              GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                              GRPCVector(id='id3',
                                         values=[1.0, 2.0, 3.0],
                                         sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))])

        Args:
            vectors (Union[list[Vector], list[Tuple]]): A list of vectors to upsert.

                     A vector can be represented by a 1) GRPCVector object, a 2) tuple or 3) a dictionary
                     1) if a tuple is used, it must be of the form (id, values, metadata) or (id, values).
                        where id is a string, vector is a list of floats, and metadata is a dict.
                        Examples: ('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])

                    2) if a GRPCVector object is used, a GRPCVector object must be of the form
                        GRPCVector(id, values, metadata), where metadata is an optional argument of type
                        dict[str, Union[str, float, int, bool, list[int], list[float], list[str]]]
                       Examples: GRPCVector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'}),
                                 GRPCVector(id='id2', values=[1.0, 2.0, 3.0]),
                                 GRPCVector(id='id3',
                                            values=[1.0, 2.0, 3.0],
                                            sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]))

                    3) if a dictionary is used, it must be in the form
                       {'id': str, 'values': list[float], 'sparse_values': {'indices': list[int], 'values': list[float]},
                        'metadata': dict}

                    Note: the dimension of each vector must match the dimension of the index.
            async_req (bool): If True, the upsert operation will be performed asynchronously.
                              Cannot be used with batch_size.
                              Defaults to False. See: https://docs.pinecone.io/docs/performance-tuning [optional]
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                                Cannot be used with async_req=True.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns: UpsertResponse, contains the number of vectors upserted
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
            future_result = self.runner.run(self.stub.Upsert.future, request, timeout=timeout)
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
        response, initial_metadata = self.runner.run(
            self.stub.Upsert, request, timeout=timeout, **kwargs
        )
        return parse_upsert_response(response, initial_metadata=initial_metadata)

    def upsert_from_dataframe(
        self,
        df: Any,
        namespace: str | None = None,
        batch_size: int = 500,
        use_async_requests: bool = True,
        show_progress: bool = True,
    ) -> UpsertResponse:
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            use_async_requests: Whether to upsert multiple requests at the same time using asynchronous request mechanism.
                                Set to ``False``
            show_progress: Whether to show a progress bar.
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
            # Type cast: dataframe dicts match VectorTypedDict structure
            res = self.upsert(
                vectors=cast(list[VectorTypedDict], chunk),
                namespace=namespace,
                async_req=use_async_requests,
            )
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
    def _iter_dataframe(df: Any, batch_size: int) -> Iterator[list[dict[str, Any]]]:
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
        """
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.

        Args:
            ids (list[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                               Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                             If not specified, the default namespace is used.
            filter (FilterTypedDict):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                     See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]
            async_req (bool): If True, the delete operation will be performed asynchronously.
                              Defaults to False. [optional]

        Returns: DeleteResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.

        .. admonition:: Note

            For any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:

        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete all must be set to False)

        Examples:

        .. code-block:: python

            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace', async_req=True)
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
            future_result = self.runner.run(self.stub.Delete.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_delete_response
            )
        else:
            response, initial_metadata = self.runner.run(self.stub.Delete, request, timeout=timeout)
            return parse_delete_response(response, initial_metadata=initial_metadata)

    def fetch(
        self,
        ids: list[str] | None,
        namespace: str | None = None,
        async_req: bool | None = False,
        **kwargs,
    ) -> FetchResponse | PineconeGrpcFuture:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:

        .. code-block:: python

            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (list[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        timeout = kwargs.pop("timeout", None)

        args_dict = self._parse_non_empty_args([("namespace", namespace)])

        request = FetchRequest(ids=ids, **args_dict, **kwargs)

        if async_req:
            future_result = self.runner.run(self.stub.Fetch.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, result_transformer=parse_fetch_response, timeout=timeout
            )
        else:
            response, initial_metadata = self.runner.run(self.stub.Fetch, request, timeout=timeout)
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
        """
        Fetch vectors by metadata filter.

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
            filter (dict[str, Union[str, float, int, bool, List, dict]]):
                Metadata filter expression to select vectors.
                See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`
            namespace (str): The namespace to fetch vectors from.
                            If not specified, the default namespace is used. [optional]
            limit (int): Max number of vectors to return. Defaults to 100. [optional]
            pagination_token (str): Pagination token to continue a previous listing operation. [optional]
            async_req (bool): If True, the fetch operation will be performed asynchronously.
                             Defaults to False. [optional]

        Returns:
            FetchByMetadataResponse: Object containing the fetched vectors, namespace, usage, and pagination token.
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
            future_result = self.runner.run(
                self.stub.FetchByMetadata.future, request, timeout=timeout
            )
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, result_transformer=parse_fetch_by_metadata_response, timeout=timeout
            )
        else:
            response, initial_metadata = self.runner.run(
                self.stub.FetchByMetadata, request, timeout=timeout
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
        response, initial_metadata = self.runner.run(self.stub.Query, request, timeout=timeout)
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
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        Examples:

        .. code-block:: python

            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=GRPCSparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (list[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each ``query()`` request can contain only one of the parameters
                                  ``id`` or ``vector``.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each ``query()`` request can contain only one of the parameters
                      ``vector`` or ``id``.. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (dict[str, Union[str, float, int, bool, List, dict]]):
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
            future_result = self.runner.run(self.stub.Query.future, request, timeout=timeout)
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
        if namespaces is None or len(namespaces) == 0:
            raise ValueError("At least one namespace must be specified")
        if len(vector) == 0:
            raise ValueError("Query vector must not be empty")

        overall_topk = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(top_k=overall_topk, metric=metric)

        target_namespaces = set(namespaces)  # dedup namespaces
        futures = [
            self.threadpool_executor.submit(
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
        id: str | None = None,
        async_req: bool = False,
        values: list[float] | None = None,
        set_metadata: VectorMetadataTypedDict | None = None,
        namespace: str | None = None,
        sparse_values: (GRPCSparseValues | SparseVectorTypedDict) | None = None,
        filter: FilterTypedDict | None = None,
        dry_run: bool | None = None,
        **kwargs,
    ) -> UpdateResponse | PineconeGrpcFuture:
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

            >>> # Update vector values
            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> # Update vector metadata
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace', async_req=True)
            >>> # Update vector values and sparse values
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        **Bulk update by metadata filter:**

        .. code-block:: python

            >>> # Update metadata for all vectors matching the filter
            >>> response = index.update(set_metadata={'status': 'active'}, filter={'genre': {'$eq': 'drama'}},
            >>>                        namespace='my_namespace')
            >>> print(f"Updated {response.matched_records} vectors")
            >>> # Preview how many vectors would be updated (dry run)
            >>> response = index.update(set_metadata={'status': 'active'}, filter={'genre': {'$eq': 'drama'}},
            >>>                        namespace='my_namespace', dry_run=True)
            >>> print(f"Would update {response.matched_records} vectors")

        Args:
            id (str): Vector's unique id. Required for single vector updates. Must not be provided when using filter. [optional]
            async_req (bool): If True, the update operation will be performed asynchronously.
                              Defaults to False. [optional]
            values (list[float]): Vector values to set. [optional]
            set_metadata (dict[str, Union[str, float, int, bool, list[int], list[float], list[str]]]]):
                Metadata to merge with existing metadata on the vector(s). Fields specified will overwrite
                existing fields with the same key, while fields not specified will remain unchanged. [optional]
            namespace (str): Namespace name where to update the vector(s). [optional]
            sparse_values: (dict[str, Union[list[float], list[int]]]): Sparse values to update for the vector.
                           Expected to be either a GRPCSparseValues object or a dict of the form:
                           {'indices': list[int], 'values': list[float]} where the lists each have the same length. [optional]
            filter (dict[str, Union[str, float, int, bool, List, dict]]): A metadata filter expression.
                    When provided, updates all vectors in the namespace that match the filter criteria.
                    See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_`.
                    Must not be provided when using id. Either `id` or `filter` must be provided. [optional]
            dry_run (bool): If `True`, return the number of records that match the `filter` without executing
                    the update. Only meaningful when using `filter` (not with `id`). Useful for previewing
                    the impact of a bulk update before applying changes. Defaults to `False`. [optional]

        Returns:
            UpdateResponse or PineconeGrpcFuture: When using filter-based updates, the UpdateResponse includes
            `matched_records` indicating the number of vectors that were updated (or would be updated if
            `dry_run=True`). If `async_req=True`, returns a PineconeGrpcFuture object instead.
        """
        # Validate that exactly one of id or filter is provided
        if id is None and filter is None:
            raise ValueError("Either 'id' or 'filter' must be provided to update vectors.")
        if id is not None and filter is not None:
            raise ValueError(
                "Cannot provide both 'id' and 'filter' in the same update call. Use 'id' for single vector updates or 'filter' for bulk updates."
            )

        if set_metadata is not None:
            set_metadata_struct = dict_to_proto_struct(set_metadata)
        else:
            set_metadata_struct = None

        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        timeout = kwargs.pop("timeout", None)
        sparse_values = SparseValuesFactory.build(sparse_values)
        args_dict = self._parse_non_empty_args(
            [
                ("id", id),
                ("values", values),
                ("set_metadata", set_metadata_struct),
                ("namespace", namespace),
                ("sparse_values", sparse_values),
                ("filter", filter_struct),
                ("dry_run", dry_run),
            ]
        )

        request = UpdateRequest(**args_dict)
        if async_req:
            future_result = self.runner.run(self.stub.Update.future, request, timeout=timeout)
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_update_response
            )
        else:
            response, initial_metadata = self.runner.run(self.stub.Update, request, timeout=timeout)
            return parse_update_response(response, initial_metadata=initial_metadata)

    def list_paginated(
        self,
        prefix: str | None = None,
        limit: int | None = None,
        pagination_token: str | None = None,
        namespace: str | None = None,
        **kwargs,
    ) -> SimpleListResponse:
        """
        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the ``list`` method to avoid having to handle pagination tokens manually.

        Examples:

        .. code-block:: python

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

        Returns: SimpleListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
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
        response, _ = self.runner.run(self.stub.List, request, timeout=timeout)

        if response.pagination and response.pagination.next != "":
            pagination = Pagination(next=response.pagination.next)
        else:
            pagination = None

        return SimpleListResponse(
            namespace=response.namespace, vectors=response.vectors, pagination=pagination
        )

    def list(self, **kwargs) -> Iterator[list[str]]:
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:

        .. code-block:: python

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
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Examples:

        .. code-block:: python

            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See `metadata filtering <https://www.pinecone.io/docs/metadata-filtering/>_` [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None
        args_dict = self._parse_non_empty_args([("filter", filter_struct)])
        timeout = kwargs.pop("timeout", None)

        request = DescribeIndexStatsRequest(**args_dict)
        response, _ = self.runner.run(self.stub.DescribeIndexStats, request, timeout=timeout)
        return parse_stats_response(response)

    @require_kwargs
    def create_namespace(
        self, name: str, schema: dict[str, Any] | None = None, async_req: bool = False, **kwargs
    ) -> NamespaceDescription | PineconeGrpcFuture:
        """
        The create_namespace operation creates a namespace in a serverless index.

        Examples:

        .. code-block:: python

            >>> index.create_namespace(name='my_namespace')

            >>> # Create namespace asynchronously
            >>> future = index.create_namespace(name='my_namespace', async_req=True)
            >>> namespace = future.result()

        Args:
            name (str): The name of the namespace to create.
            schema (Optional[dict[str, Any]]): Optional schema configuration for the namespace as a dictionary. [optional]
            async_req (bool): If True, the create_namespace operation will be performed asynchronously. [optional]

        Returns: NamespaceDescription object which contains information about the created namespace, or a PineconeGrpcFuture object if async_req is True.
        """
        timeout = kwargs.pop("timeout", None)

        # Build MetadataSchema from dict if provided
        metadata_schema = None
        if schema is not None:
            if isinstance(schema, dict):
                # Convert dict to MetadataSchema
                fields = {}
                for key, value in schema.get("fields", {}).items():
                    if isinstance(value, dict):
                        filterable = value.get("filterable", False)
                        fields[key] = MetadataFieldProperties(filterable=filterable)
                    else:
                        # If value is already a MetadataFieldProperties, use it directly
                        fields[key] = value
                metadata_schema = MetadataSchema(fields=fields)
            else:
                # Assume it's already a MetadataSchema
                metadata_schema = schema

        request_kwargs: dict[str, Any] = {"name": name}
        if metadata_schema is not None:
            request_kwargs["schema"] = metadata_schema

        request = CreateNamespaceRequest(**request_kwargs)

        if async_req:
            future_result = self.runner.run(
                self.stub.CreateNamespace.future, request, timeout=timeout
            )
            # For .future calls, runner returns (future, None, None) since .future doesn't support with_call
            future = future_result[0] if isinstance(future_result, tuple) else future_result
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_namespace_description
            )

        response, initial_metadata = self.runner.run(
            self.stub.CreateNamespace, request, timeout=timeout
        )
        return parse_namespace_description(response, initial_metadata=initial_metadata)

    @require_kwargs
    def describe_namespace(self, namespace: str, **kwargs) -> NamespaceDescription:
        """
        The describe_namespace operation returns information about a specific namespace,
        including the total number of vectors in the namespace.

        Examples:

        .. code-block:: python

            >>> index.describe_namespace(namespace='my_namespace')

        Args:
            namespace (str): The namespace to describe.

        Returns: NamespaceDescription object which contains information about the namespace.
        """
        timeout = kwargs.pop("timeout", None)
        request = DescribeNamespaceRequest(namespace=namespace)
        response, initial_metadata = self.runner.run(
            self.stub.DescribeNamespace, request, timeout=timeout
        )
        return parse_namespace_description(response, initial_metadata=initial_metadata)

    @require_kwargs
    def delete_namespace(self, namespace: str, **kwargs) -> dict[str, Any]:
        """
        The delete_namespace operation deletes a namespace from an index.
        This operation is irreversible and will permanently delete all data in the namespace.

        Examples:

        .. code-block:: python

            >>> index.delete_namespace(namespace='my_namespace')

        Args:
            namespace (str): The namespace to delete.

        Returns: Empty dictionary indicating successful deletion.
        """
        timeout = kwargs.pop("timeout", None)
        request = DeleteNamespaceRequest(namespace=namespace)
        response, initial_metadata = self.runner.run(
            self.stub.DeleteNamespace, request, timeout=timeout
        )
        return parse_delete_response(response, initial_metadata=initial_metadata)

    @require_kwargs
    def list_namespaces_paginated(
        self, limit: int | None = None, pagination_token: str | None = None, **kwargs
    ) -> ListNamespacesResponse:
        """
        The list_namespaces_paginated operation returns a list of all namespaces in a serverless index.
        It returns namespaces in a paginated form, with a pagination token to fetch the next page of results.

        Examples:

        .. code-block:: python

            >>> results = index.list_namespaces_paginated(limit=10)
            >>> [ns.name for ns in results.namespaces]
            ['namespace1', 'namespace2', 'namespace3']
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_namespaces_paginated(limit=10, pagination_token=results.pagination.next)

        Args:
            limit (Optional[int]): The maximum number of namespaces to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]

        Returns: ListNamespacesResponse object which contains the list of namespaces and pagination information.
        """
        args_dict = self._parse_non_empty_args(
            [("limit", limit), ("pagination_token", pagination_token)]
        )
        timeout = kwargs.pop("timeout", None)
        request = ListNamespacesRequest(**args_dict, **kwargs)
        response, _ = self.runner.run(self.stub.ListNamespaces, request, timeout=timeout)
        return parse_list_namespaces_response(response)

    @require_kwargs
    def list_namespaces(self, limit: int | None = None, **kwargs):
        """
        The list_namespaces operation accepts all of the same arguments as list_namespaces_paginated, and returns a generator that yields
        each namespace. It automatically handles pagination tokens on your behalf.

        Args:
            limit (Optional[int]): The maximum number of namespaces to fetch in each network call. If unspecified, the server will use a default value. [optional]

        Returns:
            Returns a generator that yields each namespace. It automatically handles pagination tokens on your behalf so you can
            easily iterate over all results. The ``list_namespaces`` method accepts all of the same arguments as list_namespaces_paginated

        Examples:

        .. code-block:: python

            >>> for namespace in index.list_namespaces():
            >>>     print(namespace.name)
            namespace1
            namespace2
            namespace3

        You can convert the generator into a list by wrapping the generator in a call to the built-in ``list`` function:

        .. code-block:: python

            namespaces = list(index.list_namespaces())

        You should be cautious with this approach because it will fetch all namespaces at once, which could be a large number
        of network calls and a lot of memory to hold the results.
        """
        done = False
        while not done:
            try:
                results = self.list_namespaces_paginated(limit=limit, **kwargs)
            except Exception as e:
                raise e

            if results.namespaces and len(results.namespaces) > 0:
                for namespace in results.namespaces:
                    yield namespace

            if results.pagination and results.pagination.next:
                kwargs.update({"pagination_token": results.pagination.next})
            else:
                done = True

    @staticmethod
    def _parse_non_empty_args(args: List[tuple[str, Any]]) -> dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}
