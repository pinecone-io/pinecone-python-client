import atexit
from typing import NamedTuple, Optional

import grpc

from pinecone.constants import Config, CLIENT_VERSION
from pinecone.utils.sentry import sentry_decorator as sentry
from .protos.vector_service_pb2_grpc import VectorServiceStub
from .retry import RetryOnRpcErrorClientInterceptor, RetryConfig
from .utils.constants import MAX_MSG_SIZE


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
    :type reuse_channel: RetryConfig, optional
    """
    secure: bool = True
    timeout: int = 20
    conn_timeout: int = 1
    reuse_channel: bool = True
    retry_config: Optional[RetryConfig] = None

    @classmethod
    def _from_dict(cls, kwargs: dict):
        cls_kwargs = {kk: vv for kk, vv in kwargs.items() if kk in cls._fields}
        return cls(**cls_kwargs)


class Index(VectorServiceStub):

    def __init__(self, name: str, channel=None, batch_size=100, disable_progress_bar=False, grpc_config: GRPCClientConfig = None):
        super().__init__(channel)
        self.name = name
        self.batch_size = batch_size
        self.disable_progress_bar = disable_progress_bar

        self.grpc_client_config = grpc_config
        self.retry_config = self.grpc_client_config.retry_config or RetryConfig()
        self.metadata = (("api-key", Config.API_KEY),
                         ("service-name", name),
                         ("client-version", CLIENT_VERSION))
        self._channel = channel
        # self._check_readiness(grpc_config)
        atexit.register(self.close)

    @sentry
    def Upsert(self,
               request,
               timeout,
               metadata=None,
               with_call=False):
        """
        TODO: docstring
        """
        _metadata = self.metadata + metadata
        super().Upsert(request, timeout, _metadata, with_call)

    @sentry
    def Delete(self,
               request,
               timeout,
               metadata=None,
               with_call=False):
        """
        TODO: docstring
        """
        _metadata = self.metadata + metadata
        super().Delete(request, timeout, _metadata, with_call)

    @sentry
    def Fetch(self,
              request,
              timeout,
              metadata=None,
              with_call=False):
        """
        TODO: docstring
        """
        _metadata = self.metadata + metadata
        super().Fetch(request, timeout, _metadata, with_call)

    @sentry
    def Query(self,
              request,
              timeout,
              metadata=None,
              with_call=False):
        """
        TODO: docstring
        """
        _metadata = self.metadata + metadata
        super().Query(request, timeout, _metadata, with_call)

    @sentry
    def List(self,
             request,
             timeout,
             metadata=None,
             with_call=False):
        """
        TODO: docstring
        """
        _metadata = self.metadata + metadata
        super().List(request, timeout, _metadata, with_call)

    @sentry
    def ListNamespaces(self,
                       request,
                       timeout,
                       metadata=None,
                       with_call=False):
        """
        TODO: docstring
        """
        _metadata = self.metadata + metadata
        super().ListNamespaces(request, timeout, _metadata, with_call)

    @sentry
    def Summarize(self,
                  request,
                  timeout,
                  metadata=None,
                  with_call=False):
        """
        TODO: docstring
        """
        _metadata = self.metadata + metadata
        super().Summarize(request, timeout, _metadata, with_call)

    def _endpoint(self):
        return f"{self.name}-{Config.PROJECT_NAME}.svc.{Config.ENVIRONMENT}.pinecone.io"

    @property
    def channel(self):
        """Creates GRPC channel."""

        def _gen():
            target = self._endpoint() + ':443'
            options = (
                ("grpc.max_send_message_length", MAX_MSG_SIZE),
                ("grpc.max_receive_message_length", MAX_MSG_SIZE),
            )
            if not self.grpc_client_config.secure:
                channel = grpc.insecure_channel(target, options=options)
            else:
                tls = grpc.ssl_channel_credentials()
                channel = grpc.secure_channel(
                    target, tls, options=(("grpc.ssl_target_name_override", self._endpoint()),) + options
                )
            interceptor = RetryOnRpcErrorClientInterceptor(self.retry_config)
            return grpc.intercept_channel(channel, interceptor)

        if self.grpc_client_config.reuse_channel and self._channel and self.grpc_server_on():
            return self._channel
        self._channel = _gen()
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
