import atexit
import random
from functools import wraps
from typing import NamedTuple, Optional, Tuple, Dict

import grpc

from pinecone.constants import Config, CLIENT_VERSION
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


class Index:

    def __init__(self, name: str, channel=None, batch_size=100, disable_progress_bar=False, grpc_config: GRPCClientConfig = None, _endpoint_override: str = None):
        self.name = name
        self.batch_size = batch_size
        self.disable_progress_bar = disable_progress_bar

        self.grpc_client_config = grpc_config or GRPCClientConfig()
        self.retry_config = self.grpc_client_config.retry_config or RetryConfig()
        self.fixed_metadata = (("api-key", Config.API_KEY),
                               ("service-name", name),
                               ("client-version", CLIENT_VERSION))
        self._endpoint_override = _endpoint_override
        self._channel = channel or self._gen_channel()
        # self._check_readiness(grpc_config)
        # atexit.register(self.close)
        self.stub = VectorServiceStub(self._channel)

    def upsert(self, *args):
        c = self._wrap_callable(self.stub.Upsert)
        return c(*args)

    def delete(self, *args):
        c = self._wrap_callable(self.stub.Delete)
        return c(*args)

    def fetch(self, *args):
        c = self._wrap_callable(self.stub.Fetch)
        return c(*args)

    def query(self, *args):
        c = self._wrap_callable(self.stub.Query)
        return c(*args)

    def list(self, *args):
        c = self._wrap_callable(self.stub.List)
        return c(*args)

    def list_namespaces(self, *args):
        c = self._wrap_callable(self.stub.ListNamespaces)
        return c(*args)

    def summarize(self, *args):
        c = self._wrap_callable(self.stub.Summarize)
        return c(*args)

    def _wrap_callable(self, func):
        @sentry
        @wraps(func)
        def wrapped(request,
                    timeout=None,
                    metadata=None,
                    credentials=None,
                    wait_for_ready=None,
                    compression=None):
            _metadata = self.fixed_metadata + self._request_metadata() #+ (metadata or ())
            return func(request, timeout=timeout, metadata=_metadata, credentials=credentials,
                        wait_for_ready=wait_for_ready, compression=compression)
        return wrapped

    def _request_metadata(self) -> Tuple[Tuple[str, str]]:
        return (REQUEST_ID, _generate_request_id()),

    def _endpoint(self):
        return self._endpoint_override if self._endpoint_override \
            else f"{self.name}-{Config.PROJECT_NAME}.svc.{Config.ENVIRONMENT}.pinecone.io:443"

    def _gen_channel(self, options=None):
        target = self._endpoint()
        default_options = {
            "grpc.max_send_message_length": MAX_MSG_SIZE,
            "grpc.max_receive_message_length": MAX_MSG_SIZE
        }
        user_provided_options = options or {}
        options = tuple((k, v) for k, v in {**default_options, **user_provided_options}.items())
        if not self.grpc_client_config.secure:
            channel = grpc.insecure_channel(target, options=options)
        else:
            tls = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel(target, tls, options=options)
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
