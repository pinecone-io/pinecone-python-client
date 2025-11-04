import logging
from typing import Optional, Dict, Union, List, Tuple, Any, Iterable, cast, Literal

from google.protobuf import json_format

from pinecone.utils.tqdm import tqdm
from concurrent.futures import as_completed, Future

from ..utils import (
    dict_to_proto_struct,
    parse_fetch_response,
    parse_fetch_by_metadata_response,
    parse_query_response,
    parse_stats_response,
    parse_upsert_response,
    parse_update_response,
    parse_delete_response,
)
from ..vector_factory_grpc import VectorFactoryGRPC
from ..sparse_values_factory import SparseValuesFactory

from pinecone.core.openapi.db_data.models import (
    FetchResponse,
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
)
from pinecone.db_data.dataclasses import FetchByMetadataResponse
from pinecone.db_control.models.list_response import ListResponse as SimpleListResponse, Pagination
from pinecone.core.grpc.protos.db_data_2025_10_pb2 import (
    Vector as GRPCVector,
    UpsertRequest,
    UpsertResponse,
    DeleteRequest,
    QueryRequest,
    FetchRequest,
    FetchByMetadataRequest,
    UpdateRequest,
    ListRequest,
    DescribeIndexStatsRequest,
    DeleteResponse,
    UpdateResponse,
    SparseValues as GRPCSparseValues,
)
from pinecone import Vector, SparseValues
from pinecone.db_data.query_results_aggregator import QueryNamespacesResults, QueryResultsAggregator
from ..future import PineconeGrpcFuture
from pinecone.db_data.types import (
    SparseVectorTypedDict,
    VectorTypedDict,
    VectorTuple,
    FilterTypedDict,
    VectorMetadataTypedDict,
)

logger = logging.getLogger(__name__)
""" :meta private: """


class VectorResourceGRPC:
    """Resource for vector-centric operations on a Pinecone index via GRPC."""

    def __init__(self, stub, runner, threadpool_executor):
        self.stub = stub
        """ :meta private: """
        self.runner = runner
        """ :meta private: """
        self.threadpool_executor = threadpool_executor
        """ :meta private: """

    @staticmethod
    def _parse_non_empty_args(args: List[Tuple[str, Any]]) -> Dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}

    def upsert(
        self,
        vectors: Union[List[Vector], List[GRPCVector], List[VectorTuple], List[VectorTypedDict]],
        async_req: bool = False,
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Union[UpsertResponse, PineconeGrpcFuture]:
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
            future = self.runner.run(self.stub.Upsert.future, request, timeout=timeout)
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_upsert_response
            )

        if batch_size is None:
            return self._upsert_batch(vectors, namespace, timeout=timeout, **kwargs)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        pbar = tqdm(total=len(vectors), disable=not show_progress, desc="Upserted vectors")
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch_result = self._upsert_batch(
                vectors[i : i + batch_size], namespace, timeout=timeout, **kwargs
            )
            pbar.update(batch_result.upserted_count)
            total_upserted += batch_result.upserted_count

        return UpsertResponse(upserted_count=total_upserted)

    def _upsert_batch(
        self, vectors: List[GRPCVector], namespace: Optional[str], timeout: Optional[int], **kwargs
    ) -> UpsertResponse:
        args_dict = self._parse_non_empty_args([("namespace", namespace)])
        request = UpsertRequest(vectors=vectors, **args_dict)
        return self.runner.run(self.stub.Upsert, request, timeout=timeout, **kwargs)

    def upsert_from_dataframe(
        self,
        df,
        namespace: str = "",
        batch_size: int = 500,
        use_async_requests: bool = True,
        show_progress: bool = True,
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
            res = self.upsert(vectors=chunk, namespace=namespace, async_req=use_async_requests)
            pbar.update(len(chunk))
            results.append(res)

        if use_async_requests:
            cast_results = cast(List[PineconeGrpcFuture], results)
            results = [
                async_result.result()
                for async_result in tqdm(
                    iterable=cast_results,
                    disable=not show_progress,
                    desc="collecting async responses",
                )
            ]

        upserted_count = 0
        for res in results:
            if hasattr(res, "upserted_count") and isinstance(res.upserted_count, int):
                upserted_count += res.upserted_count

        return UpsertResponse(upserted_count=upserted_count)

    @staticmethod
    def _iter_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size].to_dict(orient="records")
            yield batch

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        async_req: bool = False,
        **kwargs,
    ) -> Union[DeleteResponse, PineconeGrpcFuture]:
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
            future = self.runner.run(self.stub.Delete.future, request, timeout=timeout)
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_delete_response
            )
        else:
            return self.runner.run(self.stub.Delete, request, timeout=timeout)

    def fetch(
        self,
        ids: Optional[List[str]],
        namespace: Optional[str] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[FetchResponse, PineconeGrpcFuture]:
        timeout = kwargs.pop("timeout", None)

        args_dict = self._parse_non_empty_args([("namespace", namespace)])

        request = FetchRequest(ids=ids, **args_dict, **kwargs)

        if async_req:
            future = self.runner.run(self.stub.Fetch.future, request, timeout=timeout)
            return PineconeGrpcFuture(
                future, result_transformer=parse_fetch_response, timeout=timeout
            )
        else:
            response = self.runner.run(self.stub.Fetch, request, timeout=timeout)
            return parse_fetch_response(response)

    def fetch_by_metadata(
        self,
        filter: FilterTypedDict,
        namespace: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[FetchByMetadataResponse, PineconeGrpcFuture]:
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
            future = self.runner.run(self.stub.FetchByMetadata.future, request, timeout=timeout)
            return PineconeGrpcFuture(
                future, result_transformer=parse_fetch_by_metadata_response, timeout=timeout
            )
        else:
            response = self.runner.run(self.stub.FetchByMetadata, request, timeout=timeout)
            return parse_fetch_by_metadata_response(response)

    def query(
        self,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, GRPCSparseValues, SparseVectorTypedDict]
        ] = None,
        async_req: Optional[bool] = False,
        **kwargs,
    ) -> Union[QueryResponse, PineconeGrpcFuture]:
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

        if async_req:
            future = self.runner.run(self.stub.Query.future, request, timeout=timeout)
            return PineconeGrpcFuture(
                future, result_transformer=parse_query_response, timeout=timeout
            )
        else:
            response = self.runner.run(self.stub.Query, request, timeout=timeout)
            json_response = json_format.MessageToDict(response)
            return parse_query_response(json_response, _check_type=False)

    def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        metric: Literal["cosine", "euclidean", "dotproduct"],
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
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
                self.query,
                vector=vector,
                namespace=ns,
                top_k=overall_topk,
                filter=filter,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_vector,
                async_req=False,
                **kwargs,
            )
            for ns in target_namespaces
        ]

        only_futures = cast(Iterable[Future], futures)
        for response in as_completed(only_futures):
            aggregator.add_results(response.result())

        final_results = aggregator.get_results()
        return final_results

    def update(
        self,
        id: str,
        async_req: bool = False,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[GRPCSparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Union[UpdateResponse, PineconeGrpcFuture]:
        if set_metadata is not None:
            set_metadata_struct = dict_to_proto_struct(set_metadata)
        else:
            set_metadata_struct = None

        timeout = kwargs.pop("timeout", None)
        sparse_values = SparseValuesFactory.build(sparse_values)
        args_dict = self._parse_non_empty_args(
            [
                ("values", values),
                ("set_metadata", set_metadata_struct),
                ("namespace", namespace),
                ("sparse_values", sparse_values),
            ]
        )

        request = UpdateRequest(id=id, **args_dict)
        if async_req:
            future = self.runner.run(self.stub.Update.future, request, timeout=timeout)
            return PineconeGrpcFuture(
                future, timeout=timeout, result_transformer=parse_update_response
            )
        else:
            return self.runner.run(self.stub.Update, request, timeout=timeout)

    def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> SimpleListResponse:
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
        response = self.runner.run(self.stub.List, request, timeout=timeout)

        if response.pagination and response.pagination.next != "":
            pagination = Pagination(next=response.pagination.next)
        else:
            pagination = None

        return SimpleListResponse(
            namespace=response.namespace, vectors=response.vectors, pagination=pagination
        )

    def list(self, **kwargs):
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
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        if filter is not None:
            filter_struct = dict_to_proto_struct(filter)
        else:
            filter_struct = None
        args_dict = self._parse_non_empty_args([("filter", filter_struct)])
        timeout = kwargs.pop("timeout", None)

        request = DescribeIndexStatsRequest(**args_dict)
        response = self.runner.run(self.stub.DescribeIndexStats, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_stats_response(json_response)

