from typing import Optional, Union, List, Dict, Awaitable, Any

from tqdm.asyncio import tqdm
import asyncio
from google.protobuf import json_format

from pinecone.core.openapi.data.models import (
    FetchResponse,
    QueryResponse,
    DescribeIndexStatsResponse,
)
from pinecone.models.list_response import ListResponse as SimpleListResponse
from pinecone.core.grpc.protos.vector_service_pb2 import (
    Vector as GRPCVector,
    UpsertRequest,
    UpsertResponse,
    DeleteRequest,
    QueryRequest,
    FetchRequest,
    UpdateRequest,
    DescribeIndexStatsRequest,
    DeleteResponse,
    UpdateResponse,
    SparseValues as GRPCSparseValues,
)

from pinecone import Vector as NonGRPCVector
from pinecone.core.grpc.protos.vector_service_pb2_grpc import VectorServiceStub
from pinecone.utils import parse_non_empty_args

from pinecone.config import Config
from grpc._channel import Channel

from .base import GRPCIndexBase
from .config import GRPCClientConfig
from .sparse_vector import SparseVectorTypedDict
from .utils import (
    dict_to_proto_struct,
    parse_fetch_response,
    parse_query_response,
    parse_stats_response,
    parse_sparse_values_arg,
)
from .vector_factory_grpc import VectorFactoryGRPC
from ..data.query_results_aggregator import QueryResultsAggregator, QueryNamespacesResults


class GRPCIndexAsyncio(GRPCIndexBase):
    """A client for interacting with a Pinecone index over GRPC with asyncio."""

    def __init__(
        self,
        index_name: str,
        config: Config,
        channel: Optional[Channel] = None,
        grpc_config: Optional[GRPCClientConfig] = None,
        _endpoint_override: Optional[str] = None,
    ):
        super().__init__(
            index_name=index_name,
            config=config,
            channel=channel,
            grpc_config=grpc_config,
            _endpoint_override=_endpoint_override,
            use_asyncio=True,
        )

    @property
    def stub_class(self):
        return VectorServiceStub

    def _get_semaphore(
        self,
        max_concurrent_requests: Optional[int] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> asyncio.Semaphore:
        if semaphore is not None and max_concurrent_requests is not None:
            raise ValueError("Cannot specify both `max_concurrent_requests` and `semaphore`")
        if semaphore is not None:
            return semaphore
        if max_concurrent_requests is None:
            return asyncio.Semaphore(25)
        return asyncio.Semaphore(max_concurrent_requests)

    async def upsert(
        self,
        vectors: Union[List[GRPCVector], List[NonGRPCVector], List[tuple], List[dict]],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        max_concurrent_requests: Optional[int] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs,
    ) -> UpsertResponse:
        timeout = kwargs.pop("timeout", None)
        vectors = list(map(VectorFactoryGRPC.build, vectors))
        semaphore = self._get_semaphore(max_concurrent_requests, semaphore)

        if batch_size is None:
            return await self._upsert_batch(
                vectors=vectors, namespace=namespace, timeout=timeout, semaphore=semaphore, **kwargs
            )

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        vector_batches = [vectors[i : i + batch_size] for i in range(0, len(vectors), batch_size)]
        tasks = [
            self._upsert_batch(
                vectors=batch, semaphore=semaphore, namespace=namespace, timeout=100, **kwargs
            )
            for batch in vector_batches
        ]

        if namespace is not None:
            pbar_desc = f"Upserted vectors in namespace '{namespace}'"
        else:
            pbar_desc = "Upserted vectors in namespace ''"

        upserted_count = 0
        with tqdm(total=len(vectors), disable=not show_progress, desc=pbar_desc) as pbar:
            for task in asyncio.as_completed(tasks):
                res = await task
                pbar.update(res.upserted_count)
                upserted_count += res.upserted_count
        return UpsertResponse(upserted_count=upserted_count)

    async def _upsert_batch(
        self,
        vectors: List[GRPCVector],
        semaphore: asyncio.Semaphore,
        namespace: Optional[str],
        timeout: Optional[int] = None,
        **kwargs,
    ) -> UpsertResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])
        request = UpsertRequest(vectors=vectors, **args_dict)
        return await self.runner.run_asyncio(
            self.stub.Upsert, request, timeout=timeout, semaphore=semaphore, **kwargs
        )

    async def _query(
        self,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if vector is not None and id is not None:
            raise ValueError("Cannot specify both `id` and `vector`")

        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        sparse_vector = parse_sparse_values_arg(sparse_vector)
        args_dict = parse_non_empty_args(
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
        semaphore = self._get_semaphore(None, semaphore)

        response = await self.runner.run_asyncio(
            self.stub.Query, request, timeout=timeout, semaphore=semaphore
        )
        parsed = json_format.MessageToDict(response)
        return parsed

    async def query(
        self,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: Optional[int] = 10,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs,
    ) -> QueryResponse:
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        Examples:
            >>> await index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> await index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> await index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> await index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> await index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> await index.query(vector=[1, 2, 3], sparse_vector=GRPCSparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`.. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
                            Expected to be either a GRPCSparseValues object or a dict of the form:
                             {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
        # We put everything but the response parsing into the private _query method so
        # that we can reuse it when querying over multiple namespaces. Since we need to do
        # some work to aggregate and present the results from multiple namespaces in that
        # case, we don't want to create a bunch of intermediate openapi QueryResponse
        # objects that will just be thrown out in favor of a different presentation of those
        # aggregate results.
        json_response = await self._query(
            vector=vector,
            id=id,
            namespace=namespace,
            top_k=top_k,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
            semaphore=semaphore,
            **kwargs,
        )
        return parse_query_response(json_response, _check_type=False)

    async def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        show_progress: Optional[bool] = True,
        max_concurrent_requests: Optional[int] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        aggregator_lock = asyncio.Lock()
        semaphore = self._get_semaphore(max_concurrent_requests, semaphore)

        if len(namespaces) == 0:
            raise ValueError("At least one namespace must be specified")
        if len(vector) == 0:
            raise ValueError("Query vector must not be empty")

        # The caller may only want the top_k=1 result across all queries,
        # but we need to get at least 2 results from each query in order to
        # aggregate them correctly. So we'll temporarily set topK to 2 for the
        # subqueries, and then we'll take the topK=1 results from the aggregated
        # results.
        overall_topk = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(top_k=overall_topk)
        subquery_topk = overall_topk if overall_topk > 2 else 2

        target_namespaces = set(namespaces)  # dedup namespaces
        query_tasks = [
            self._query(
                vector=vector,
                namespace=ns,
                top_k=subquery_topk,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                semaphore=semaphore,
                **kwargs,
            )
            for ns in target_namespaces
        ]

        with tqdm(
            total=len(query_tasks), disable=not show_progress, desc="Querying namespaces"
        ) as pbar:
            for query_task in asyncio.as_completed(query_tasks):
                response = await query_task
                pbar.update(1)
                async with aggregator_lock:
                    aggregator.add_results(response)

        final_results = aggregator.get_results()
        return final_results

    async def upsert_from_dataframe(
        self,
        df,
        namespace: str = "",
        batch_size: int = 500,
        use_async_requests: bool = True,
        show_progress: bool = True,
    ) -> Awaitable[UpsertResponse]:
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            use_async_requests: Whether to upsert multiple requests at the same time using asynchronous request mechanism.
                                Set to `False`
            show_progress: Whether to show a progress bar.
        """
        # try:
        #     import pandas as pd
        # except ImportError:
        #     raise RuntimeError(
        #         "The `pandas` package is not installed. Please install pandas to use `upsert_from_dataframe()`"
        #     )

        # if not isinstance(df, pd.DataFrame):
        #     raise ValueError(f"Only pandas dataframes are supported. Found: {type(df)}")

        # pbar = tqdm(total=len(df), disable=not show_progress, desc="sending upsert requests")
        # results = []
        # for chunk in self._iter_dataframe(df, batch_size=batch_size):
        #     res = self.upsert(vectors=chunk, namespace=namespace, async_req=use_async_requests)
        #     pbar.update(len(chunk))
        #     results.append(res)

        # if use_async_requests:
        #     cast_results = cast(List[PineconeGrpcFuture], results)
        #     results = [
        #         async_result.result()
        #         for async_result in tqdm(
        #             cast_results, disable=not show_progress, desc="collecting async responses"
        #         )
        #     ]

        # upserted_count = 0
        # for res in results:
        #     if hasattr(res, "upserted_count") and isinstance(res.upserted_count, int):
        #         upserted_count += res.upserted_count

        # return UpsertResponse(upserted_count=upserted_count)
        raise NotImplementedError(
            "upsert_from_dataframe is not yet implemented for GRPCIndexAsyncio"
        )

    # @staticmethod
    # def _iter_dataframe(df, batch_size):
    #     for i in range(0, len(df), batch_size):
    #         batch = df.iloc[i : i + batch_size].to_dict(orient="records")
    #         yield batch

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        **kwargs,
    ) -> Awaitable[DeleteResponse]:
        """
        The Delete operation deletes vectors from the index, from a single namespace.
        No error raised if the vector id does not exist.
        Note: for any delete call, if namespace is not specified, the default namespace is used.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
           (note that for this option delete all must be set to False)

        Examples:
            >>> await index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> await index.delete(delete_all=True, namespace='my_namespace')
            >>> await index.delete(filter={'key': 'value'}, namespace='my_namespace', async_req=True)

        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                               Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                             If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                     See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DeleteResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """
        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None

        args_dict = parse_non_empty_args(
            [
                ("ids", ids),
                ("delete_all", delete_all),
                ("namespace", namespace),
                ("filter", filter_struct),
            ]
        )
        timeout = kwargs.pop("timeout", None)

        request = DeleteRequest(**args_dict, **kwargs)
        return await self.runner.run_asyncio(self.stub.Delete, request, timeout=timeout)

    async def fetch(
        self, ids: Optional[List[str]], namespace: Optional[str] = None, **kwargs
    ) -> Awaitable[FetchResponse]:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        Examples:
            >>> await index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> await index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        timeout = kwargs.pop("timeout", None)

        args_dict = parse_non_empty_args([("namespace", namespace)])

        request = FetchRequest(ids=ids, **args_dict, **kwargs)
        response = await self.runner.run_asyncio(self.stub.Fetch, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_fetch_response(json_response)

    async def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[
            Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]
        ] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Awaitable[UpdateResponse]:
        """
        The Update operation updates vector in a namespace.
        If a value is included, it will overwrite the previous value.
        If a set_metadata is included, the values of the fields specified in it will be added or overwrite the previous value.

        Examples:
            >>> await index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> await index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace', async_req=True)
            >>> await index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> await index.update(id='id1', values=[1, 2, 3], sparse_values=GRPCSparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        Args:
            id (str): Vector's unique id.
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a GRPCSparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.


        Returns: UpdateResponse (contains no data) or a PineconeGrpcFuture object if async_req is True.
        """
        if set_metadata is not None:
            set_metadata_struct = dict_to_proto_struct(set_metadata)
        else:
            set_metadata_struct = None

        timeout = kwargs.pop("timeout", None)
        sparse_values = parse_sparse_values_arg(sparse_values)
        args_dict = parse_non_empty_args(
            [
                ("values", values),
                ("set_metadata", set_metadata_struct),
                ("namespace", namespace),
                ("sparse_values", sparse_values),
            ]
        )

        request = UpdateRequest(id=id, **args_dict)
        return await self.runner.run_asyncio(self.stub.Update, request, timeout=timeout)

    async def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> Awaitable[SimpleListResponse]:
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

        Returns: SimpleListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
        """
        # args_dict = parse_non_empty_args(
        #     [
        #         ("prefix", prefix),
        #         ("limit", limit),
        #         ("namespace", namespace),
        #         ("pagination_token", pagination_token),
        #     ]
        # )
        # request = ListRequest(**args_dict, **kwargs)
        # timeout = kwargs.pop("timeout", None)
        # response = self.runner.run(self.stub.List, request, timeout=timeout)

        # if response.pagination and response.pagination.next != "":
        #     pagination = Pagination(next=response.pagination.next)
        # else:
        #     pagination = None

        # return SimpleListResponse(
        #     namespace=response.namespace, vectors=response.vectors, pagination=pagination
        # )
        raise NotImplementedError("list_paginated is not yet implemented for GRPCIndexAsyncio")

    async def list(self, **kwargs):
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
        # done = False
        # while not done:
        #     try:
        #         results = self.list_paginated(**kwargs)
        #     except Exception as e:
        #         raise e

        #     if len(results.vectors) > 0:
        #         yield [v.id for v in results.vectors]

        #     if results.pagination and results.pagination.next:
        #         kwargs.update({"pagination_token": results.pagination.next})
        #     else:
        #         done = True
        raise NotImplementedError("list is not yet implemented for GRPCIndexAsyncio")

    async def describe_index_stats(
        self, filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None, **kwargs
    ) -> Awaitable[DescribeIndexStatsResponse]:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        Examples:
            >>> await index.describe_index_stats()
            >>> await index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None
        args_dict = parse_non_empty_args([("filter", filter_struct)])
        timeout = kwargs.pop("timeout", None)

        request = DescribeIndexStatsRequest(**args_dict)
        response = await self.runner.run_asyncio(
            self.stub.DescribeIndexStats, request, timeout=timeout
        )
        json_response = json_format.MessageToDict(response)
        return parse_stats_response(json_response)
