from tqdm.autonotebook import tqdm
from importlib.util import find_spec
import numbers
import numpy as np

from collections.abc import Iterable, Mapping
from typing import Union, List, Tuple, Optional, Dict, Any

from .core.client.model.sparse_values import SparseValues
from pinecone import Config
from pinecone.core.client import ApiClient
from .core.client.models import (
    FetchResponse,
    ProtobufAny,
    QueryRequest,
    QueryResponse,
    QueryVector,
    RpcStatus,
    ScoredVector,
    SingleQueryResults,
    DescribeIndexStatsResponse,
    UpsertRequest,
    UpsertResponse,
    UpdateRequest,
    Vector,
    DeleteRequest,
    UpdateRequest,
    DescribeIndexStatsRequest,
)
from pinecone.core.client.api.vector_operations_api import VectorOperationsApi
from pinecone.core.utils import fix_tuple_length, get_user_agent, warn_deprecated
import copy

__all__ = [
    "Index",
    "FetchResponse",
    "ProtobufAny",
    "QueryRequest",
    "QueryResponse",
    "QueryVector",
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

from .core.utils.constants import REQUIRED_VECTOR_FIELDS, OPTIONAL_VECTOR_FIELDS
from .core.utils.error_handling import validate_and_convert_errors

_OPENAPI_ENDPOINT_PARAMS = (
    "_return_http_data_only",
    "_preload_content",
    "_request_timeout",
    "_check_input_type",
    "_check_return_type",
    "_host_index",
    "async_req",
)


def parse_query_response(response: QueryResponse, unary_query: bool):
    if unary_query:
        response._data_store.pop("results", None)
    else:
        response._data_store.pop("matches", None)
        response._data_store.pop("namespace", None)
    return response


def upsert_numpy_deprecation_notice(context):
    numpy_deprecataion_notice = "The ability to pass a numpy ndarray as part of a dictionary argument to upsert() will be removed in a future version of the pinecone client. To remove this warning, use the numpy.ndarray.tolist method to convert your ndarray into a python list before calling upsert()."
    message = " ".join([context, numpy_deprecataion_notice])
    warn_deprecated(message, deprecated_in="2.2.1", removal_in="3.0.0")


class Index(ApiClient):
    """A class for interacting with a Pinecone index via REST API.
    
    The ``Index`` class is used to perform data operations (upsert, query, etc) against Pinecone indexes. Usually it will 
    be instantiated using the `pinecone` module after the required configuration values have been initialized.

    ```python
    import pinecone
    pinecone.init(api_key="my-api-key", environment="my-environment")
    index = pinecone.Index("my-index")
    ```
    For improved performance, use the Pinecone GRPCIndex client. For more details, see [Performance tuning](https://docs.pinecone.io/docs/performance-tuning]).

    Args:
        index_name (str): The name of the index to interact with.
    """

    def __init__(self, index_name: str, pool_threads=1):
        openapi_client_config = copy.deepcopy(Config.OPENAPI_CONFIG)
        openapi_client_config.api_key = openapi_client_config.api_key or {}
        openapi_client_config.api_key["ApiKeyAuth"] = openapi_client_config.api_key.get("ApiKeyAuth", Config.API_KEY)
        openapi_client_config.server_variables = openapi_client_config.server_variables or {}
        openapi_client_config.server_variables = {
            **{"environment": Config.ENVIRONMENT, "index_name": index_name, "project_name": Config.PROJECT_NAME},
            **openapi_client_config.server_variables,
        }
        super().__init__(configuration=openapi_client_config, pool_threads=pool_threads)

        self.user_agent = get_user_agent()
        """@private"""
        self._vector_api = VectorOperationsApi(self)

    @validate_and_convert_errors
    def upsert(
        self,
        vectors: Union[List[Vector], List[tuple], List[dict]],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse:
        """Upsert records to the index.
        
        If a new value is upserted for an existing vector id, it will overwrite the previous value. To upsert 
        in parallel follow [sending upserts in parallel](https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel).

        A vector can be represented by a list of either a ``Vector`` object, a tuple, or a dictionary.

        **Tuple**

        If a tuple is used, it must be one of the following forms:
        - `(id, values)`
        - `(id, values, metadata)`
        - `(id, values, metadata, sparse_values)`

        where `id` is a string, `values` is a list of floats, `metadata` is a dict, and `sparse_values`
        is a dict of the form `{'indices': List[int], 'values': List[float]}`.

        Examples:
        ```python
        # id, values
        ('id2', [1.0, 2.0, 3.0])
        # id, values, metadata
        ('id1', [1.0, 2.0, 3.0], {'key': 'value'})
        # id, values, metadata, sparse_values
        ('id1', [1.0, 2.0, 3.0], {'key': 'value'}, {'indices': [1, 2], 'values': [0.2, 0.4]})
        # sending sparse_values without any metadata specified
        ('id1', [1.0, 2.0, 3.0], None, {'indices': [1, 2], 'values': [0.2, 0.4]})
        ```
        **Vector object**

        If a ``Vector`` object is used, it can be instantiated like so: `Vector(id, values, metadata, sparse_values)`.
        Metadata and sparse_values are optional arguments.

        Examples:
        ```python
        Vector(id='id2', values=[1.0, 2.0, 3.0])
        Vector(id='id1', values=[1.0, 2.0, 3.0], metadata={'key': 'value'})
        Vector(id='id3', values=[1.0, 2.0, 3.0], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]))
        ```
        **Dictionary**

        If a dictionary is used, it must be in the form `{'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]}, 'metadata': dict}`.

        Examples:
        ```python
        # upsert a list of tuples
        index.upsert([('id1', [1.0, 2.0, 3.0], {'key': 'value'}), ('id2', [1.0, 2.0, 3.0])])
        # upsert a list of Vector objects
        index.upsert(
            [
                Vector(id="id1", values=[1.0, 2.0, 3.0], metadata={"key": "value"}),
                Vector(
                    id="id2",
                    values=[1.0, 2.0, 3.0],
                    sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
                ),
            ]
        )
        # upsert a list of dictionaries
        index.upsert(
            [
                {"id": "id1", "values": [1.0, 2.0, 3.0], "metadata": {"key": "value"}},
                {
                    "id": "id2",
                    "values": [1.0, 2.0, 3.0],
                    "sparse_values": {"indices": [1, 8], "values": [0.2, 0.4]},
                },
            ]
        )
        ```

        [Pinecone API reference](https://docs.pinecone.io/reference/upsert)

        Args:
            vectors (Union[List[Vector], List[Tuple]]): A list of vectors to upsert. Must be a `tuple`, `dictionary`, or ``Vector`` object.
            namespace (str, optional): The namespace to write to. If not specified, the default namespace of `''` is used.
            batch_size (int, optional): The number of vectors to upsert in each batch. If not specified, all vectors will be upserted in a single batch.
            show_progress (bool, optional): Whether to show a progress bar using tqdm. Applied only if batch_size is provided. Default: True

        Keyword Args:
            Supports OpenAPI client keyword arguments. See `UpsertRequest` for more details.

        Returns: 
            An ``UpsertResponse`` which includes the number of vectors upserted.
        """
        _check_type = kwargs.pop("_check_type", False)

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
            batch_result = self._upsert_batch(vectors[i : i + batch_size], namespace, _check_type, **kwargs)
            pbar.update(batch_result.upserted_count)
            # we can't use here pbar.n for the case show_progress=False
            total_upserted += batch_result.upserted_count

        return UpsertResponse(upserted_count=total_upserted)

    def _upsert_batch(
        self, vectors: List[Vector], namespace: Optional[str], _check_type: bool, **kwargs
    ) -> UpsertResponse:
        args_dict = self._parse_non_empty_args([("namespace", namespace)])

        def _dict_to_vector(item):
            item_keys = set(item.keys())
            if not item_keys.issuperset(REQUIRED_VECTOR_FIELDS):
                raise ValueError(
                    f"Vector dictionary is missing required fields: {list(REQUIRED_VECTOR_FIELDS - item_keys)}"
                )

            excessive_keys = item_keys - (REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)
            if len(excessive_keys) > 0:
                raise ValueError(
                    f"Found excess keys in the vector dictionary: {list(excessive_keys)}. "
                    f"The allowed keys are: {list(REQUIRED_VECTOR_FIELDS | OPTIONAL_VECTOR_FIELDS)}"
                )

            if "sparse_values" in item:
                if not isinstance(item["sparse_values"], Mapping):
                    raise ValueError(
                        f"Column `sparse_values` is expected to be a dictionary, found {type(item['sparse_values'])}"
                    )

                indices = item["sparse_values"].get("indices", None)
                values = item["sparse_values"].get("values", None)

                if isinstance(values, np.ndarray):
                    upsert_numpy_deprecation_notice("Deprecated type passed in sparse_values['values'].")
                    values = values.tolist()
                if isinstance(indices, np.ndarray):
                    upsert_numpy_deprecation_notice("Deprecated type passed in sparse_values['indices'].")
                    indices = indices.tolist()
                try:
                    item["sparse_values"] = SparseValues(indices=indices, values=values)
                except TypeError as e:
                    raise ValueError(
                        "Found unexpected data in column `sparse_values`. "
                        "Expected format is `'sparse_values': {'indices': List[int], 'values': List[float]}`."
                    ) from e

            if "metadata" in item:
                metadata = item.get("metadata")
                if not isinstance(metadata, Mapping):
                    raise TypeError(f"Column `metadata` is expected to be a dictionary, found {type(metadata)}")

            if isinstance(item["values"], np.ndarray):
                upsert_numpy_deprecation_notice("Deprecated type passed in 'values'.")
                item["values"] = item["values"].tolist()

            try:
                return Vector(**item)
            except TypeError as e:
                # if not isinstance(item['values'], Iterable) or not isinstance(item['values'][0], numbers.Real):
                #     raise TypeError(f"Column `values` is expected to be a list of floats")
                if not isinstance(item["values"], Iterable) or not isinstance(item["values"][0], numbers.Real):
                    raise TypeError(f"Column `values` is expected to be a list of floats")
                raise

        def _vector_transform(item: Union[Vector, Tuple]):
            print("ITEM: ", item)
            if isinstance(item, Vector):
                return item
            elif isinstance(item, tuple):
                if len(item) > 3:
                    raise ValueError(
                        f"Found a tuple of length {len(item)} which is not supported. "
                        f"Vectors can be represented as tuples either the form (id, values, metadata) or (id, values). "
                        f"To pass sparse values please use either dicts or a Vector objects as inputs."
                    )
                id, values, metadata = fix_tuple_length(item, 3)
                return Vector(id=id, values=values, metadata=metadata or {}, _check_type=_check_type)
            elif isinstance(item, Mapping):
                return _dict_to_vector(item)
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

        return self._vector_api.upsert(
            UpsertRequest(
                vectors=list(map(_vector_transform, vectors)),
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
        self, df, namespace: str = None, batch_size: int = 500, show_progress: bool = True
    ) -> UpsertResponse:
        """Upserts a pandas dataframe to the index.

        The datafram must have the following columns: `id`, `vector`, `sparse_values`, and `metadata`.

        Example:
        ```python
        import pandas as pd
        import pinecone

        data = {
            'id': ['id1', 'id2'], 
            'vector': [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], 
            'sparse_values': [{'indices': [1, 2], 'values': [0.2, 0.4]}, None], 
            'metadata': [None, {'genre': 'classical'}]
            }

        dataframe = pd.DataFrame(data)
        pinecone.upsert_from_dataframe(df=dataframe)
        ```

        Args:
            df (DataFrame): A pandas dataframe with the following columns: `id`, `vector`, `sparse_values`, and `metadata`.
            namespace (str, optional): The namespace to write to. If not specified, the default namespace of `''` is used.
            batch_size (int, optional): The number of vectors to upsert in each batch. Default: 500
            show_progress (bool, optional): Whether to show a progress bar using tqdm. Applied only if batch_size is provided. Default: True
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
        """Delete vectors from the index within a single namespace.
        
        For any delete call, if namespace is not specified the default namespace of `''` will be used.
        There are no errors raised if a vector id does not exist.

        Deletion can occur in one of the following mutually exclusive ways:
        - Delete by ids from a single namespace
        - Delete all vectors from a single namespace by setting `delete_all` to True
        - Delete all vectors from a single namespace by specifying a metadata filter. For this option
            `delete_all` must be set to False

        Examples:
        ```python
        index.delete(ids=['id1', 'id2'], namespace='my_namespace')
        index.delete(delete_all=True, namespace='my_namespace')
        index.delete(filter={'key': 'value'}, namespace='my_namespace')
        ```

        [Pinecone API reference](https://docs.pinecone.io/reference/delete_post)
        
        Args:
            ids (List[str], optional): The ids of the vectors to delete.
            delete_all (bool, optional): This indicates that all vectors in the index namespace should be deleted.
            namespace (str): The namespace to delete vectors from. If not specified, the default namespace of `''` is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]): If specified, the metadata filter will be used to 
                select the vectors to delete. This is mutually exclusive with specifying ids to delete in the `ids` param or 
                using `delete_all=True`. See [Filtering with metadata](https://www.pinecone.io/docs/metadata-filtering/) for
                more on deleting records with filters.

        Keyword Args:
            Supports OpenAPI client keyword arguments. See ``DeleteRequest`` for more details.

        Returns: 
            An empty dictionary if the delete operation was successful.
        """
        _check_type = kwargs.pop("_check_type", False)
        args_dict = self._parse_non_empty_args(
            [("ids", ids), ("delete_all", delete_all), ("namespace", namespace), ("filter", filter)]
        )

        return self._vector_api.delete(
            DeleteRequest(
                **args_dict,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS and v is not None},
                _check_type=_check_type,
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )

    @validate_and_convert_errors
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> FetchResponse:
        """Fetch vectors from the index.

        Examples:
        ```python
        index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
        index.fetch(ids=['id1', 'id2'])
        ```

        [Pinecone API reference](https://docs.pinecone.io/reference/fetch)
        
        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str, optional): The namespace to fetch vectors from. If not specified, the default namespace of `''` is used.
        Keyword Args:
            Supports OpenAPI client keyword arguments. See ``FetchResponse`` for more details.

        Returns: 
            ``FetchResponse`` object which contains the list of Vector objects, and namespace name.
        """
        args_dict = self._parse_non_empty_args([("namespace", namespace)])
        return self._vector_api.fetch(ids=ids, **args_dict, **kwargs)

    @validate_and_convert_errors
    def query(
        self,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        queries: Optional[Union[List[QueryVector], List[Tuple]]] = None,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Union[str, float, int, bool, List, dict]]] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, Dict[str, Union[List[float], List[int]]]]] = None,
        **kwargs,
    ) -> QueryResponse:
        """Query vectors from the index using a query vector.

        Query is used to find the `top_k` vectors in the index whose vector values are most similar to the vector values of
        the query according to the distance metric you have configured for your index. See [Query data](https://docs.pinecone.io/docs/query-data) 
        for more on querying.

        **Note:** Each query request can only contain one of the following parameters: `vector`, `id`, or `queries`.

        Examples:
        ```python
        index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
        index.query(id='id1', top_k=10, namespace='my_namespace')
        index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
        index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
        index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
                    top_k=10, namespace='my_namespace')
        index.query(vector=[1, 2, 3], sparse_vector=SparseValues([1, 2], [0.2, 0.4]),
                    top_k=10, namespace='my_namespace')
        ```

        [Pinecone API reference](https://docs.pinecone.io/reference/query)
        
        Args:
            vector (List[float], optional): The query vector. This should be the same length as the dimension of the index being queried.
            id (str, optional): The unique ID of the vector to be used as a query vector.
            queries ([QueryVector], optional): DEPRECATED. The query vectors.
            top_k (int, optional): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str, optional): The namespace to query vectors from. If not specified, the default namespace of `''` is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]): The metadata filter to apply. You can use vector metadata 
                to limit your search. See [Filtering with metadata](https://www.pinecone.io/docs/metadata-filtering/) for more on filtering.
            include_values (bool, optional): Indicates whether vector values are included in the response. If omitted, the server will 
                use the default value of False.
            include_metadata (bool, optional): Indicates whether metadata is included in the response as well as the ids. If omitted the server will 
                use the default value of False.
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector. Expected to be 
                either a ``SparseValues`` object or a dict of the form: `{'indices': List[int], 'values': List[float]}`, 
                where the lists each have the same length.

        Keyword Args:
            Supports OpenAPI client keyword arguments. See ``QueryRequest`` for more details.

        Returns: 
            ``QueryResponse`` object which contains the list of the closest vectors as ``ScoredVector`` objects along with namespace name.
        """

        def _query_transform(item):
            if isinstance(item, QueryVector):
                return item
            if isinstance(item, tuple):
                values, filter = fix_tuple_length(item, 2)
                if filter is None:
                    return QueryVector(values=values, _check_type=_check_type)
                else:
                    return QueryVector(values=values, filter=filter, _check_type=_check_type)
            if isinstance(item, Iterable):
                return QueryVector(values=item, _check_type=_check_type)
            raise ValueError(f"Invalid query vector value passed: cannot interpret type {type(item)}")

        _check_type = kwargs.pop("_check_type", False)
        queries = list(map(_query_transform, queries)) if queries is not None else None

        sparse_vector = self._parse_sparse_values_arg(sparse_vector)
        args_dict = self._parse_non_empty_args(
            [
                ("vector", vector),
                ("id", id),
                ("queries", queries),
                ("top_k", top_k),
                ("namespace", namespace),
                ("filter", filter),
                ("include_values", include_values),
                ("include_metadata", include_metadata),
                ("sparse_vector", sparse_vector),
            ]
        )
        response = self._vector_api.query(
            QueryRequest(
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )
        return parse_query_response(response, vector is not None or id)

    @validate_and_convert_errors
    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, Dict[str, Union[List[float], List[int]]]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """ Update a vector in the index within a specific namespace.

        If `values` are included they will overwrite the previous values. If `set_metadata` is included,
        the values of the fields specified will overwrite and merge with existing metadata.

        Examples:
        ```python
        index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
        index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace')
        index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
                     namespace='my_namespace')
        index.update(id='id1', values=[1, 2, 3], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
                     namespace='my_namespace')
        ```

        [Pinecone API reference](https://docs.pinecone.io/reference/update)

        Args:
            id (str): The unique id of the vector you would like to update.
            values (List[float], optional): The vector values you would like to update.
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]], optional):
                The metadata you would like to update.
            namespace (str, optional): The namespace from which to update the vector. If not specified, the default namespace of `''` is used.
            sparse_values: (Dict[str, Union[List[float], List[int]]], optional): The sparse values you would like to store with this vector. 
                Expected to be either a SparseValues object or a dict of the form: `{'indices': List[int], 'values': List[float]}` where 
                the lists each have the same length.

        Keyword Args:
            Supports OpenAPI client keyword arguments. See ``UpdateRequest`` for more details.

        Returns: 
            An empty dictionary if the update was successful.
        """
        _check_type = kwargs.pop("_check_type", False)
        sparse_values = self._parse_sparse_values_arg(sparse_values)
        args_dict = self._parse_non_empty_args(
            [
                ("values", values),
                ("set_metadata", set_metadata),
                ("namespace", namespace),
                ("sparse_values", sparse_values),
            ]
        )
        return self._vector_api.update(
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
        """ Describe statistics about the index's contents.

        Returns details such as total number of vectors, vectors per namespace, and the index's dimension size.

        Examples:
            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        [Pinecone API reference](https://docs.pinecone.io/reference/describe_index_stats_post)

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]], optional): If this parameter is present, 
                the operation only returns statistics for vectors that satisfy the filter.
                See [Filtering with metadata](https://www.pinecone.io/docs/metadata-filtering/) for more on filtering.

        Returns: 
            ``DescribeIndexStatsResponse`` object which contains stats about the index.
        """
        _check_type = kwargs.pop("_check_type", False)
        args_dict = self._parse_non_empty_args([("filter", filter)])

        return self._vector_api.describe_index_stats(
            DescribeIndexStatsRequest(
                **args_dict,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
                _check_type=_check_type,
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS},
        )

    @staticmethod
    def _parse_non_empty_args(args: List[Tuple[str, Any]]) -> Dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}

    @staticmethod
    def _parse_sparse_values_arg(
        sparse_values: Optional[Union[SparseValues, Dict[str, Union[List[float], List[int]]]]]
    ) -> Optional[SparseValues]:
        if sparse_values is None:
            return None

        if isinstance(sparse_values, SparseValues):
            return sparse_values

        if not isinstance(sparse_values, dict) or "indices" not in sparse_values or "values" not in sparse_values:
            raise ValueError(
                "Invalid sparse values argument. Expected a dict of: {'indices': List[int], 'values': List[float]}."
                f"Received: {sparse_values}"
            )

        return SparseValues(indices=sparse_values["indices"], values=sparse_values["values"])
