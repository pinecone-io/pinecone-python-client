from tqdm.autonotebook import tqdm

import logging
import json
from typing import Union, List, Optional, Dict, Any

from pinecone.config import ConfigBuilder

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.db_data.api.vector_operations_api import VectorOperationsApi
from pinecone.core.openapi.db_data import API_VERSION
from pinecone.core.openapi.db_data.models import (
    FetchResponse,
    QueryRequest,
    QueryResponse,
    RpcStatus,
    ScoredVector,
    SingleQueryResults,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertRequest,
    UpsertResponse,
    Vector,
    DeleteRequest,
    UpdateRequest,
    DescribeIndexStatsRequest,
    ListResponse,
    SparseValues,
)
from .interfaces import IndexInterface
from .features.bulk_import import ImportFeatureMixin
from ..utils import (
    setup_openapi_client,
    parse_non_empty_args,
    build_plugin_setup_client,
    validate_and_convert_errors,
)
from .vector_factory import VectorFactory
from .query_results_aggregator import QueryResultsAggregator, QueryNamespacesResults

from multiprocessing.pool import ApplyResult
from concurrent.futures import as_completed

from pinecone_plugin_interface import load_and_install as install_plugins

logger = logging.getLogger(__name__)

__all__ = [
    "Index",
    "_Index",
    "FetchResponse",
    "QueryRequest",
    "QueryResponse",
    "RpcStatus",
    "ScoredVector",
    "SingleQueryResults",
    "DescribeIndexStatsResponse",
    "UpsertRequest",
    "UpsertResponse",
    "UpdateRequest",
    "Vector",
    "DeleteRequest",
    "UpdateRequest",
    "DescribeIndexStatsRequest",
    "SparseValues",
]

_OPENAPI_ENDPOINT_PARAMS = (
    "_return_http_data_only",
    "_preload_content",
    "_request_timeout",
    "_check_input_type",
    "_check_return_type",
    "_host_index",
    "async_req",
    "async_threadpool_executor",
)


def parse_query_response(response: QueryResponse):
    response._data_store.pop("results", None)
    return response


class IndexClientInstantiationError(Exception):
    def __init__(self, index_args, index_kwargs):
        formatted_args = ", ".join(map(repr, index_args))
        formatted_kwargs = ", ".join(f"{key}={repr(value)}" for key, value in index_kwargs.items())
        combined_args = ", ".join([a for a in [formatted_args, formatted_kwargs] if a.strip()])

        self.message = f"""You are attempting to access the Index client directly from the pinecone module. The Index client must be instantiated through the parent Pinecone client instance so that it can inherit shared configurations such as API key.

    INCORRECT USAGE:
        ```
        import pinecone

        pc = pinecone.Pinecone(api_key='your-api-key')
        index = pinecone.Index({combined_args})
        ```

    CORRECT USAGE:
        ```
        from pinecone import Pinecone

        pc = Pinecone(api_key='your-api-key')
        index = pc.Index({combined_args})
        ```
        """
        super().__init__(self.message)


class Index:
    def __init__(self, *args, **kwargs):
        raise IndexClientInstantiationError(args, kwargs)


class _Index(IndexInterface, ImportFeatureMixin):
    """
    A client for interacting with a Pinecone index via REST API.
    For improved performance, use the Pinecone GRPC index client.
    """

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
            api_client_klass=ApiClient,
            api_klass=VectorOperationsApi,
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vector_api.api_client.close()

    @validate_and_convert_errors
    def upsert(
        self,
        vectors: Union[List[Vector], List[tuple], List[dict]],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse:
        _check_type = kwargs.pop("_check_type", True)

        if kwargs.get("async_req", False) and batch_size is not None:
            raise ValueError(
                "async_req is not supported when batch_size is provided."
                "To upsert in parallel, please follow: "
                "https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel"
            )

        if batch_size is None:
            return self._upsert_batch(vectors, namespace, _check_type, **kwargs)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        pbar = tqdm(total=len(vectors), disable=not show_progress, desc="Upserted vectors")
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch_result = self._upsert_batch(
                vectors[i : i + batch_size], namespace, _check_type, **kwargs
            )
            pbar.update(batch_result.upserted_count)
            # we can't use here pbar.n for the case show_progress=False
            total_upserted += batch_result.upserted_count

        return UpsertResponse(upserted_count=total_upserted)

    def _upsert_batch(
        self,
        vectors: Union[List[Vector], List[tuple], List[dict]],
        namespace: Optional[str],
        _check_type: bool,
        **kwargs,
    ) -> UpsertResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])

        def vec_builder(v):
            return VectorFactory.build(v, check_type=_check_type)

        return self._vector_api.upsert_vectors(
            UpsertRequest(
                vectors=list(map(vec_builder, vectors)),
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )

    @staticmethod
    def _iter_dataframe(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size].to_dict(orient="records")
            yield batch

    def upsert_from_dataframe(
        self, df, namespace: Optional[str] = None, batch_size: int = 500, show_progress: bool = True
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
        for res in results:
            upserted_count += res.upserted_count

        return UpsertResponse(upserted_count=upserted_count)

    @validate_and_convert_errors
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args(
            [("ids", ids), ("delete_all", delete_all), ("namespace", namespace), ("filter", filter)]
        )

        return self._vector_api.delete_vectors(
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
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> FetchResponse:
        args_dict = parse_non_empty_args([("namespace", namespace)])
        return self._vector_api.fetch_vectors(ids=ids, **args_dict, **kwargs)

    @validate_and_convert_errors
    def query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, Dict[str, Union[List[float], List[int]]]]
        ] = None,
        **kwargs,
    ) -> Union[QueryResponse, ApplyResult]:
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
            return response
        else:
            return parse_query_response(response)

    def _query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, Dict[str, Union[List[float], List[int]]]]
        ] = None,
        **kwargs,
    ) -> QueryResponse:
        if len(args) > 0:
            raise ValueError(
                "The argument order for `query()` has changed; please use keyword arguments instead of positional arguments. Example: index.query(vector=[0.1, 0.2, 0.3], top_k=10, namespace='my_namespace')"
            )

        if vector is not None and id is not None:
            raise ValueError("Cannot specify both `id` and `vector`")

        _check_type = kwargs.pop("_check_type", False)

        sparse_vector = self._parse_sparse_values_arg(sparse_vector)
        args_dict = parse_non_empty_args(
            [
                ("vector", vector),
                ("id", id),
                ("queries", None),
                ("top_k", top_k),
                ("namespace", namespace),
                ("filter", filter),
                ("include_values", include_values),
                ("include_metadata", include_metadata),
                ("sparse_vector", sparse_vector),
            ]
        )

        response = self._vector_api.query_vectors(
            QueryRequest(
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )
        return response

    @validate_and_convert_errors
    def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[
            Union[SparseValues, Dict[str, Union[List[float], List[int]]]]
        ] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        if namespaces is None or len(namespaces) == 0:
            raise ValueError("At least one namespace must be specified")
        if len(vector) == 0:
            raise ValueError("Query vector must not be empty")

        overall_topk = top_k if top_k is not None else 10
        aggregator = QueryResultsAggregator(top_k=overall_topk)

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

        for result in as_completed(async_futures):
            raw_result = result.result()
            response = json.loads(raw_result.data.decode("utf-8"))
            aggregator.add_results(response)

        final_results = aggregator.get_results()
        return final_results

    @validate_and_convert_errors
    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[
            Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]
        ] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[
            Union[SparseValues, Dict[str, Union[List[float], List[int]]]]
        ] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        _check_type = kwargs.pop("_check_type", False)
        sparse_values = self._parse_sparse_values_arg(sparse_values)
        args_dict = parse_non_empty_args(
            [
                ("values", values),
                ("set_metadata", set_metadata),
                ("namespace", namespace),
                ("sparse_values", sparse_values),
            ]
        )
        return self._vector_api.update_vector(
            UpdateRequest(
                id=id,
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )

    @validate_and_convert_errors
    def describe_index_stats(
        self, filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        _check_type = kwargs.pop("_check_type", False)
        args_dict = parse_non_empty_args([("filter", filter)])

        return self._vector_api.describe_index_stats(
            DescribeIndexStatsRequest(
                **args_dict,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
                _check_type=_check_type,
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )

    @validate_and_convert_errors
    def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> ListResponse:
        args_dict = parse_non_empty_args(
            [
                ("prefix", prefix),
                ("limit", limit),
                ("namespace", namespace),
                ("pagination_token", pagination_token),
            ]
        )
        return self._vector_api.list_vectors(**args_dict, **kwargs)

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

    @staticmethod
    def _parse_sparse_values_arg(
        sparse_values: Optional[Union[SparseValues, Dict[str, Union[List[float], List[int]]]]],
    ) -> Optional[SparseValues]:
        if sparse_values is None:
            return None

        if isinstance(sparse_values, SparseValues):
            return sparse_values

        if (
            not isinstance(sparse_values, dict)
            or "indices" not in sparse_values
            or "values" not in sparse_values
        ):
            raise ValueError(
                "Invalid sparse values argument. Expected a dict of: {'indices': List[int], 'values': List[float]}."
                f"Received: {sparse_values}"
            )

        return SparseValues(indices=sparse_values["indices"], values=sparse_values["values"])
