import atexit
import random
from abc import ABC, abstractmethod
from functools import wraps
from typing import NamedTuple, Optional, Tuple, Dict

import grpc

from pinecone import logger
from pinecone.constants import Config, CLIENT_VERSION
from pinecone.protos.vector_column_service_pb2_grpc import VectorColumnServiceStub
from pinecone.protos import vector_service_pb2, vector_column_service_pb2
from pinecone.utils import _generate_request_id
from pinecone.utils.sentry import sentry_decorator as sentry
from pinecone.protos.vector_service_pb2_grpc import VectorServiceStub
from pinecone.retry import RetryOnRpcErrorClientInterceptor, RetryConfig
from pinecone.utils.constants import MAX_MSG_SIZE, REQUEST_ID


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


class GRPCIndex(ABC):
    """
    Base class for grpc-based interaction with Pinecone indexes
    """

    def __init__(self, name: str, channel=None, grpc_config: GRPCClientConfig = None, _endpoint_override: str = None):
        self.name = name
        # self.batch_size = batch_size
        # self.disable_progress_bar = disable_progress_bar

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
            tls = grpc.ssl_channel_credentials()
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

    def _wrap_grpc_call(self, func, request, timeout=None, metadata=None, credentials=None, wait_for_ready=None, compression=None):
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


class Index(GRPCIndex):

    @property
    def stub_class(self):
        return VectorServiceStub

    def upsert(self,
               request: 'vector_service_pb2.UpsertRequest',
               timeout: int = None,
               metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Upsert, request, timeout=timeout, metadata=metadata)

    def delete(self,
               request: 'vector_service_pb2.DeleteRequest',
               timeout: int = None,
               metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Delete, request, timeout=timeout, metadata=metadata)

    def fetch(self,
              request: 'vector_service_pb2.FetchRequest',
              timeout: int = None,
              metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Fetch, request, timeout=timeout, metadata=metadata)

    def query(self,
              request: 'vector_service_pb2.QueryRequest',
              timeout: int = None,
              metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Query, request, timeout=timeout, metadata=metadata)

    def list(self,
             request: 'vector_service_pb2.ListRequest',
             timeout: int = None,
             metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.List, request, timeout=timeout, metadata=metadata)

    def list_namespaces(self,
                        request: 'vector_service_pb2.ListNamespacesRequest',
                        timeout: int = None,
                        metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.ListNamespaces, request, timeout=timeout, metadata=metadata)

    def summarize(self,
                  request: 'vector_service_pb2.SummarizeRequest',
                  timeout: int = None,
                  metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Summarize, request, timeout=timeout, metadata=metadata)


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

    def list(self,
             request: 'vector_column_service_pb2.ListRequest',
             timeout: int = None,
             metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.List, request, timeout=timeout, metadata=metadata)

    def list_namespaces(self,
                        request: 'vector_column_service_pb2.ListNamespacesRequest',
                        timeout: int = None,
                        metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.ListNamespaces, request, timeout=timeout, metadata=metadata)

    def summarize(self,
                  request: 'vector_column_service_pb2.SummarizeRequest',
                  timeout: int = None,
                  metadata: Dict[str, str] = None):
        return self._wrap_grpc_call(self.stub.Summarize, request, timeout=timeout, metadata=metadata)


