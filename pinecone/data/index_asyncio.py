from tqdm.autonotebook import tqdm

import logging
import asyncio
import json

from .interfaces import AsyncioIndexInterface
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
    setup_openapi_client,
    parse_non_empty_args,
    build_plugin_setup_client,
    validate_and_convert_errors,
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
from pinecone_plugin_interface import load_and_install as install_plugins

logger = logging.getLogger(__name__)

__all__ = ["_AsyncioIndex"]

_OPENAPI_ENDPOINT_PARAMS = (
    "_return_http_data_only",
    "_preload_content",
    "_request_timeout",
    "_check_input_type",
    "_check_return_type",
    "_host_index",
)


def parse_query_response(response: QueryResponse):
    if hasattr(response, "_data_store"):
        # I'm not sure, but I think this is no longer needed. At some point
        # in the past the query response returned "results" instead of matches
        # and then for some time it returned both keys even though "results"
        # was always empty. I'm leaving this here just in case.
        response._data_store.pop("results", None)
    return response


class _AsyncioIndex(AsyncioIndexInterface, ImportFeatureMixinAsyncio):
    def __init__(
        self,
        api_key: str,
        host: str,
        pool_threads: Optional[int] = 1,
        additional_headers: Optional[Dict[str, str]] = {},
        openapi_config=None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            host=host,
            pool_threads=pool_threads,
            additional_headers=additional_headers,
            openapi_config=openapi_config,
            **kwargs,
        )

        self.config = ConfigBuilder.build(
            api_key=api_key, host=host, additional_headers=additional_headers, **kwargs
        )
        self._openapi_config = ConfigBuilder.build_openapi_config(self.config, openapi_config)
        self._pool_threads = pool_threads

        if kwargs.get("connection_pool_maxsize", None):
            self._openapi_config.connection_pool_maxsize = kwargs.get("connection_pool_maxsize")

        self._vector_api = setup_openapi_client(
            api_client_klass=AsyncioApiClient,
            api_klass=AsyncioVectorOperationsApi,
            config=self.config,
            openapi_config=self._openapi_config,
            pool_threads=pool_threads,
            api_version=API_VERSION,
        )

        self._load_plugins()

    def _load_plugins(self):
        """@private"""
        try:
            # I don't expect this to ever throw, but wrapping this in a
            # try block just in case to make sure a bad plugin doesn't
            # halt client initialization.
            openapi_client_builder = build_plugin_setup_client(
                config=self.config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
            install_plugins(self, openapi_client_builder)
        except Exception as e:
            logger.error(f"Error loading plugins in Index: {e}")

    def __aenter__(self):
        return self

    def __aexit__(self, exc_type, exc_value, traceback):
        self._vector_api.api_client.close()

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

    def _openapi_kwargs(self, kwargs):
        return {k: v for k, v in kwargs.items() if k in OPENAPI_ENDPOINT_PARAMS}
