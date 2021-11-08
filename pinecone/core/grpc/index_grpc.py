#
# Copyright (c) 2020-2021 Pinecone Systems Inc. All right reserved.
#

from abc import ABC, abstractmethod
from functools import wraps
from typing import NamedTuple, Optional, Dict, Iterable

import grpc
import certifi
from google.protobuf import json_format

from pinecone import logger
from pinecone.config import Config
from pinecone.core.grpc.protos.vector_column_service_pb2_grpc import VectorColumnServiceStub
from pinecone.core.grpc.protos import vector_service_pb2, vector_column_service_pb2
from pinecone.core.utils import _generate_request_id, dict_to_proto_struct, fix_tuple_length, proto_struct_to_dict
from pinecone.core.utils.sentry import sentry_decorator as sentry
from pinecone.core.grpc.protos.vector_service_pb2_grpc import VectorServiceStub
from pinecone.core.grpc.retry import RetryOnRpcErrorClientInterceptor, RetryConfig
from pinecone.core.utils.constants import MAX_MSG_SIZE, REQUEST_ID, CLIENT_VERSION
from pinecone.core.grpc.protos.vector_service_pb2 import Vector, QueryVector, UpsertRequest, DeleteRequest, \
    QueryRequest, FetchRequest, DescribeIndexStatsRequest, UpsertResponse
from pinecone.core.utils.error_handling import validate_and_convert_errors

__all__ = ["GRPCIndex", "QueryVector", "Vector"]


class GRPCClientConfig(NamedTuple):
    """
    GRPC client configuration options.

    :param secure: Whether to use encrypted protocol (SSL). defaults to True.
    :type traceroute: bool, optional
    :param timeout: defaults to 2 seconds. Fail if gateway doesn't receive response within timeout.
    :type timeout: int, optional
    :param conn_timeout: defaults to 1. Timeout to retry connection if gRPC is unavailable. 0 is no retry.
    :type conn_timeout: int, optional
    :param reuse_channel: Whether to reuse the same grpc channel for multiple requests
    :type reuse_channel: bool, optional
    :param retry_config: RetryConfig indicating how requests should be retried
    :type retry_config: RetryConfig, optional
    :param grpc_channel_options: A dict of gRPC channel arguments
    :type grpc_channel_options: Dict[str, str]
    """
    secure: bool = True
    timeout: int = 20
    conn_timeout: int = 1
    reuse_channel: bool = True
    retry_config: Optional[RetryConfig] = None
    grpc_channel_options: Dict[str, str] = None

    @classmethod
    def _from_dict(cls, kwargs: dict):
        cls_kwargs = {kk: vv for kk, vv in kwargs.items() if kk in cls._fields}
        return cls(**cls_kwargs)


class GRPCIndexBase(ABC):
    """
    Base class for grpc-based interaction with Pinecone indexes
    """

    def __init__(self, name: str, channel=None, grpc_config: GRPCClientConfig = None, _endpoint_override: str = None):
        self.name = name

        self.grpc_client_config = grpc_config or GRPCClientConfig()
        self.retry_config = self.grpc_client_config.retry_config or RetryConfig()
        self.fixed_metadata = {
            "api-key": Config.API_KEY,
            "service-name": name,
            "client-version": CLIENT_VERSION
        }
        self._endpoint_override = _endpoint_override
        self._channel = channel or self._gen_channel()
        # self._check_readiness(grpc_config)
        # atexit.register(self.close)
        self.stub = self.stub_class(self._channel)

    @property
    @abstractmethod
    def stub_class(self):
        pass

    def _endpoint(self):
        return self._endpoint_override if self._endpoint_override \
            else f"{self.name}-{Config.PROJECT_NAME}.svc.{Config.ENVIRONMENT}.pinecone.io:443"

    def _gen_channel(self, options=None):
        target = self._endpoint()
        default_options = {
            "grpc.max_send_message_length": MAX_MSG_SIZE,
            "grpc.max_receive_message_length": MAX_MSG_SIZE
        }
        if self.grpc_client_config.secure:
            default_options['grpc.ssl_target_name_override'] = target.split(':')[0]
        user_provided_options = options or {}
        _options = tuple((k, v) for k, v in {**default_options, **user_provided_options}.items())
        logger.debug('creating new channel with endpoint {} options {} and config {}',
                     target, _options, self.grpc_client_config)
        if not self.grpc_client_config.secure:
            channel = grpc.insecure_channel(target, options=_options)
        else:
            root_cas = open(certifi.where(), "rb").read()
            tls = grpc.ssl_channel_credentials(root_certificates=root_cas)
            channel = grpc.secure_channel(target, tls, options=_options)
        interceptor = RetryOnRpcErrorClientInterceptor(self.retry_config)
        return grpc.intercept_channel(channel, interceptor)

    @property
    def channel(self):
        """Creates GRPC channel."""
        if self.grpc_client_config.reuse_channel and self._channel and self.grpc_server_on():
            return self._channel
        self._channel = self._gen_channel()
        return self._channel

    def grpc_server_on(self) -> bool:
        try:
            grpc.channel_ready_future(self._channel).result(timeout=self.grpc_client_config.conn_timeout)
            return True
        except grpc.FutureTimeoutError:
            return False

    def _check_readiness(self, grpc_config: dict):
        """Sets up a connection to an index."""
        # api = ControllerAPI(host=Config.CONTROLLER_HOST, api_key=Config.API_KEY)
        # status = api.get_status(self.name)
        # if not status.get("ready"):
        #     raise ConnectionError

        # if self.name not in api.list_services():
        #     raise RuntimeError("Index '{}' is not found.".format(self.name))
        pass

    @sentry
    def close(self):
        """Closes the connection to the index."""
        try:
            self._channel.close()
        except TypeError:
            pass

    def _wrap_grpc_call(self, func, request, timeout=None, metadata=None, credentials=None, wait_for_ready=None,
                        compression=None):
        @sentry
        @wraps(func)
        def wrapped():
            user_provided_metadata = metadata or {}
            _metadata = tuple((k, v) for k, v in {
                **self.fixed_metadata, **self._request_metadata(), **user_provided_metadata
            }.items())
            return func(request, timeout=timeout, metadata=_metadata, credentials=credentials,
                        wait_for_ready=wait_for_ready, compression=compression)

        return wrapped()

    def _request_metadata(self) -> Dict[str, str]:
        return {REQUEST_ID: _generate_request_id()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class vector(object):
    def __init__(self, id: str, values: list, metadata: dict):
        self.id = id
        self.values = values
        self.metadata = metadata

    def __repr__(self):
        return str({'id': self.id, 'values': self.values, 'metadata': self.metadata})


class FetchResponse(object):
    def __init__(self, vectors: dict, namespace: str):
        self.vectors = vectors
        self.namespace = namespace

    def __repr__(self):
        return str({'vectors': self.vectors, 'namespace': self.namespace})


def parse_fetch_response(response: dict):
    vd = {}
    vectors = response.get('vectors')
    if not vectors:
        return None
    for id, vec in vectors.items():
        v_obj = vector(id=vec['id'], values=vec['values'], metadata=vec.get('metadata', None))
        vd[id] = v_obj
    namespace = response.get('namespace', None)
    f = FetchResponse(vectors=vd, namespace=namespace)
    return f


class ScoredVector(object):
    def __init__(self, id: str, score: float, values: list, metadata: dict):
        self.id = id
        self.score = score
        self.values = values
        self.metadata = metadata

    def __repr__(self):
        return str({'id': self.id, 'score': self.score, 'values': self.values, 'metadtata': self.metadata})


class QueryResult(object):
    def __init__(self, matches: list, namespace: str):
        self.matches = matches
        self.namespace = namespace

    def __repr__(self):
        return str({'matches': self.matches, 'namespace': self.namespace})


class QueryResponse(object):
    def __init__(self, results: list):
        self.results = results

    def __repr__(self):
        return str({'results': self.results})


def parse_query_response(response: dict):
    res = []

    for match in response['results']:
        namespace = match.get('namespace', None)
        m = []
        for item in match['matches']:
            sc = ScoredVector(id=item['id'], score=item['score'], values=item.get('values', []),
                              metadata=item.get('metadata', {}))
            m.append(sc)
        res.append(QueryResult(matches=m, namespace=namespace))
    return QueryResponse(results=res)


class GRPCIndex(GRPCIndexBase):

    @property
    def stub_class(self):
        return VectorServiceStub

    @sentry
    @validate_and_convert_errors
    def upsert(self, vectors, **kwargs):
        def _vector_transform(item):
            if isinstance(item, Vector):
                item.metadata
                return item
            if isinstance(item, tuple):
                id, values, metadata = fix_tuple_length(item, 3)
                return Vector(id=id, values=values, metadata=dict_to_proto_struct(metadata) or {})
            raise ValueError(f"Invalid vector value passed: cannot interpret type {type(item)}")

        request = UpsertRequest(vectors=list(map(_vector_transform, vectors)), **kwargs)
        timeout = kwargs.pop('timeout', None)
        response = self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout)
        return response

    @sentry
    @validate_and_convert_errors
    def delete(self, *args, **kwargs):
        request = DeleteRequest(*args, **kwargs)
        timeout = kwargs.pop('timeout', None)
        response = self._wrap_grpc_call(self.stub.Delete, request, timeout=timeout)
        return response

    @sentry
    @validate_and_convert_errors
    def fetch(self, *args, **kwargs):
        timeout = kwargs.pop('timeout', None)
        request = FetchRequest(*args, **kwargs)
        response = self._wrap_grpc_call(self.stub.Fetch, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_fetch_response(json_response)

    @sentry
    @validate_and_convert_errors
    def query(self, queries, **kwargs):
        timeout = kwargs.pop('timeout', None)

        def _query_transform(item):
            if isinstance(item, QueryVector):
                return item
            if isinstance(item, tuple):
                values, filter = fix_tuple_length(item, 2)
                filter = dict_to_proto_struct(filter)
                return QueryVector(values=values, filter=filter)
            if isinstance(item, Iterable):
                return QueryVector(values=item)
            raise ValueError(f"Invalid query vector value passed: cannot interpret type {type(item)}")

        _QUERY_ARGS = ['namespace', 'top_k', 'filter', 'include_values', 'include_metadata']
        if 'filter' in kwargs:
            kwargs['filter'] = dict_to_proto_struct(kwargs['filter'])
        request = QueryRequest(queries=list(map(_query_transform, queries)),
                               **{k: v for k, v in kwargs.items() if k in _QUERY_ARGS})

        response = self._wrap_grpc_call(self.stub.Query, request, timeout=timeout)
        json_response = json_format.MessageToDict(response)
        return parse_query_response(json_response)

    @sentry
    @validate_and_convert_errors
    def describe_index_stats(self, **kwargs):
        timeout = kwargs.pop('timeout', None)
        request = DescribeIndexStatsRequest()
        response = self._wrap_grpc_call(self.stub.DescribeIndexStats, request, timeout=timeout)
        return response


class CIndex(GRPCIndex):

    @property
    def stub_class(self):
        return VectorColumnServiceStub

    def upsert(self,
               request: 'vector_column_service_pb2.UpsertRequest',
               timeout: int = None,
               metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout, metadata=metadata)

    def delete(self,
               request: 'vector_column_service_pb2.DeleteRequest',
               timeout: int = None,
               metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Delete, request, timeout=timeout, metadata=metadata)

    def fetch(self,
              request: 'vector_column_service_pb2.FetchRequest',
              timeout: int = None,
              metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Fetch, request, timeout=timeout, metadata=metadata)

    def query(self,
              request: 'vector_column_service_pb2.QueryRequest',
              timeout: int = None,
              metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Query, request, timeout=timeout, metadata=metadata)

    def describe_index_stats(self,
                             request: 'vector_column_service_pb2.DescribeIndexStatsRequest',
                             timeout: int = None,
                             metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.DescribeIndexStats, request, timeout=timeout, metadata=metadata)
