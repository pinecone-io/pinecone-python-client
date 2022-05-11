#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from collections.abc import Iterable

from pinecone import Config
from pinecone.core.client import ApiClient, Configuration
from .core.client.models import FetchResponse, ProtobufAny, QueryRequest, QueryResponse, QueryVector, RpcStatus, \
    ScoredVector, SingleQueryResults, DescribeIndexStatsResponse, UpsertRequest, UpsertResponse, UpdateRequest, \
    Vector, DeleteRequest, UpdateRequest
from pinecone.core.client.api.vector_operations_api import VectorOperationsApi
from pinecone.core.utils import fix_tuple_length, get_user_agent
import copy

__all__ = [
    "Index", "FetchResponse", "ProtobufAny", "QueryRequest", "QueryResponse", "QueryVector", "RpcStatus",
    "ScoredVector", "SingleQueryResults", "DescribeIndexStatsResponse", "UpsertRequest", "UpsertResponse",
    "UpdateRequest", "Vector", "DeleteRequest", "UpdateRequest"
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
    def upsert(self, vectors, **kwargs):
        _check_type = kwargs.pop('_check_type', False)

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
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS}
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )

    @validate_and_convert_errors
    def delete(self, *args, **kwargs):
        _check_type = kwargs.pop('_check_type', False)
        return self._vector_api.delete(
            DeleteRequest(
                *args,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS},
                _check_type=_check_type
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )

    @validate_and_convert_errors
    def fetch(self, *args, **kwargs):
        return self._vector_api.fetch(*args, **kwargs)

    @validate_and_convert_errors
    def query(self, vector=[], id='', queries=[], **kwargs):
        _check_type = kwargs.pop('_check_type', False)

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
                queries=list(map(_query_transform, queries)),
                vector=vector,
                id=id,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS}
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS}
        )
        return parse_query_response(response, vector or id)

    @validate_and_convert_errors
    def update(self, id, **kwargs):
        _check_type = kwargs.pop('_check_type', False)
        return self._vector_api.update(UpdateRequest(
                id=id,
                _check_type=_check_type,
                **{k: v for k, v in kwargs.items() if k not in _OPENAPI_ENDPOINT_PARAMS}
            ),
            **{k: v for k, v in kwargs.items() if k in _OPENAPI_ENDPOINT_PARAMS})

    @validate_and_convert_errors
    def describe_index_stats(self, *args, **kwargs):
        return self._vector_api.describe_index_stats(*args, **kwargs)
