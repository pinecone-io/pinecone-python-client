#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from collections.abc import Iterable
from typing import Union, List, Tuple, Optional, Dict, Any

from pinecone import Config
from pinecone.core.client import ApiClient
from .core.client.models import FetchResponse, ProtobufAny, QueryRequest, QueryResponse, QueryVector, RpcStatus, \
    ScoredVector, SingleQueryResults, DescribeIndexStatsResponse, UpsertRequest, UpsertResponse, UpdateRequest, \
    Vector, DeleteRequest, UpdateRequest, DescribeIndexStatsRequest
from pinecone.core.client.api.vector_operations_api import VectorOperationsApi
from pinecone.core.utils import fix_tuple_length, get_user_agent
import copy

__all__ = [
    "Index", "FetchResponse", "ProtobufAny", "QueryRequest", "QueryResponse", "QueryVector", "RpcStatus",
    "ScoredVector", "SingleQueryResults", "DescribeIndexStatsResponse", "UpsertRequest", "UpsertResponse",
    "UpdateRequest", "Vector", "DeleteRequest", "UpdateRequest", "DescribeIndexStatsRequest"
]

from .core.utils.error_handling import validate_and_convert_errors

_OPENAPI_ENDPOINT_PARAMS = (
    '_return_http_data_only', '_preload_content', '_request_timeout',
    '_check_input_type', '_check_return_type', '_host_index', 'async_req'
)


def parse_query_response(response: QueryResponse, unary_query: bool):
    if unary_query:
        response._data_store.pop('results', None)
    else:
        response._data_store.pop('matches', None)
        response._data_store.pop('namespace', None)
    return response


class Index(ApiClient):

    def __init__(self, index_name: str, pool_threads=1):
        openapi_client_config = copy.deepcopy(Config.OPENAPI_CONFIG)
        openapi_client_config.api_key = openapi_client_config.api_key or {}
        openapi_client_config.api_key['ApiKeyAuth'] = openapi_client_config.api_key.get('ApiKeyAuth', Config.API_KEY)
        openapi_client_config.server_variables = openapi_client_config.server_variables or {}
        openapi_client_config.server_variables = {
            **{
                'environment': Config.ENVIRONMENT,
                'index_name': index_name,
                'project_name': Config.PROJECT_NAME
            },
            **openapi_client_config.server_variables
        }
        super().__init__(configuration=openapi_client_config, pool_threads=pool_threads)
        self.user_agent = get_user_agent()
        self._vector_api = VectorOperationsApi(self)

    @validate_and_convert_errors
    def upsert(self,
               vectors: Union[List[Vector], List[Tuple]],
               namespace: Optional[str] = None,
               **kwargs) -> UpsertResponse:
        _check_type = kwargs.pop('_check_type', False)
        args_dict = self._parse_args_to_dict([('namespace', namespace)])

        def _vector_transform(item):
            if isinstance(item, Vector):
                return item
            if isinstance(item, tuple):
                id, values, metadata = fix_tuple_length(item, 3)
                return Vector(id=id, values=values, metadata=metadata or {}, _check_type=_check_type)
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

        return self._vector_api.upsert(
            UpsertRequest(
                vectors=list(map(_vector_transform, vectors)),
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS}
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )

    @validate_and_convert_errors
    def delete(self,
               ids: Optional[List[str]] = None,
               delete_all: Optional[bool] = None,
               namespace: Optional[str] = None,
               filter: Optional[Dict[str, Any]] = None,
               **kwargs) -> Dict[str, Any]:
        _check_type = kwargs.pop('_check_type', False)
        args_dict = self._parse_args_to_dict([('ids', ids),
                                              ('delete_all', delete_all),
                                              ('namespace', namespace),
                                              ('filter', filter)])

        return self._vector_api.delete(
            DeleteRequest(
                **args_dict,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS and v is not None},
                _check_type=_check_type
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )

    @validate_and_convert_errors
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> FetchResponse:
        args_dict = self._parse_args_to_dict([('namespace', namespace)])
        return self._vector_api.fetch(ids, **args_dict, **kwargs)

    @validate_and_convert_errors
    def query(self,
              vector: Optional[List[float]] = None,
              id: Optional[str] = None,
              queries: Optional[List[QueryVector]] = None,
              top_k: Optional[int] = None,
              namespace: Optional[str] = None,
              filter: Optional[Dict[str, Any]] = None,
              include_values: Optional[bool] = None,
              include_metadata: Optional[bool] = None,
              **kwargs) -> QueryResponse:

        def _query_transform(item):
            if isinstance(item, QueryVector):
                return item
            if isinstance(item, tuple):
                values, filter = fix_tuple_length(item, 2)
                return QueryVector(values=values, filter=filter, _check_type=_check_type)
            if isinstance(item, Iterable):
                return QueryVector(values=item, _check_type=_check_type)
            raise ValueError(f"Invalid query vector value passed: cannot interpret type {type(item)}")

        _check_type = kwargs.pop('_check_type', False)
        queries = list(map(_query_transform, queries)) if queries is not None else None
        args_dict = self._parse_args_to_dict([('vector', vector),
                                              ('id', id),
                                              ('queries', queries),
                                              ('top_k', top_k),
                                              ('namespace', namespace),
                                              ('filter', filter),
                                              ('include_values', include_values),
                                              ('include_metadata', include_metadata)])

        def _query_transform(item):
            if isinstance(item, QueryVector):
                return item
            if isinstance(item, tuple):
                values, filter = fix_tuple_length(item, 2)
                return QueryVector(values=values, filter=filter, _check_type=_check_type)
            if isinstance(item, Iterable):
                return QueryVector(values=item, _check_type=_check_type)
            raise ValueError(f"Invalid query vector value passed: cannot interpret type {type(item)}")

        response = self._vector_api.query(
            QueryRequest(
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS}
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )
        return parse_query_response(response, vector is not None or id)

    @validate_and_convert_errors
    def update(self,
               id: str,
               values: Optional[List[float]] = None,
               set_metadata: Optional[Dict[str, Any]] = None,
               namespace: Optional[str] = None,
               **kwargs):
        _check_type = kwargs.pop('_check_type', False)
        args_dict = self._parse_args_to_dict([('values', values),
                                              ('set_metadata', set_metadata),
                                              ('namespace', namespace)])
        return self._vector_api.update(UpdateRequest(
                id=id,
                **args_dict,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS}
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS})

    @validate_and_convert_errors
    def describe_index_stats(self, filter: Dict[str, Any] = {}, **kwargs) -> DescribeIndexStatsResponse:
        _check_type = kwargs.pop('_check_type', False)
        return self._vector_api.describe_index_stats(
            DescribeIndexStatsRequest(
                filter=filter,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
                _check_type=_check_type
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )

    @staticmethod
    def _parse_args_to_dict(args: List[Tuple[str, Any]]) -> Dict[str, Any]:
        return {arg_name: val for arg_name, val in args if val is not None}
