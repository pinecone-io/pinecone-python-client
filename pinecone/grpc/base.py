import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Optional

import grpc
from grpc._channel import _InactiveRpcError, Channel

from .retry import RetryConfig
from .channel_factory import GrpcChannelFactory

from pinecone import Config
from .utils import _generate_request_id
from .config import GRPCClientConfig
from pinecone.utils.constants import REQUEST_ID, CLIENT_VERSION
from pinecone.exceptions.exceptions import PineconeException

_logger = logging.getLogger(__name__)


class GRPCIndexBase(ABC):
    """
    Base class for grpc-based interaction with Pinecone indexes
    """

    _pool = None

    def __init__(
        self,
        index_name: str,
        config: Config,
        channel: Optional[Channel] = None,
        grpc_config: Optional[GRPCClientConfig] = None,
        _endpoint_override: Optional[str] = None,
    ):
        self.config = config
        self.grpc_client_config = grpc_config or GRPCClientConfig()
        self.retry_config = self.grpc_client_config.retry_config or RetryConfig()

        self.fixed_metadata = {
            "api-key": config.api_key,
            "service-name": index_name,
            "client-version": CLIENT_VERSION,
        }
        if self.grpc_client_config.additional_metadata:
            self.fixed_metadata.update(self.grpc_client_config.additional_metadata)

        self._endpoint_override = _endpoint_override

        self.channel_factory = GrpcChannelFactory(
            config=self.config, grpc_client_config=self.grpc_client_config, use_asyncio=False
        )
        self._channel = channel or self._gen_channel()
        self.stub = self.stub_class(self._channel)

    @property
    @abstractmethod
    def stub_class(self):
        pass

    def _endpoint(self):
        grpc_host = self.config.host.replace("https://", "")
        if ":" not in grpc_host:
            grpc_host = f"{grpc_host}:443"
        return self._endpoint_override if self._endpoint_override else grpc_host

    def _gen_channel(self):
        return self.channel_factory.create_channel(self._endpoint())

    @property
    def channel(self):
        """Creates GRPC channel."""
        if self.grpc_client_config.reuse_channel and self._channel and self.grpc_server_on():
            return self._channel
        self._channel = self._gen_channel()
        return self._channel

    def grpc_server_on(self) -> bool:
        try:
            grpc.channel_ready_future(self._channel).result(
                timeout=self.grpc_client_config.conn_timeout
            )
            return True
        except grpc.FutureTimeoutError:
            return False

    def close(self):
        """Closes the connection to the index."""
        try:
            self._channel.close()
        except TypeError:
            pass

    def _wrap_grpc_call(
        self,
        func,
        request,
        timeout=None,
        metadata=None,
        credentials=None,
        wait_for_ready=None,
        compression=None,
    ):
        @wraps(func)
        def wrapped():
            user_provided_metadata = metadata or {}
            _metadata = tuple(
                (k, v)
                for k, v in {
                    **self.fixed_metadata,
                    **self._request_metadata(),
                    **user_provided_metadata,
                }.items()
            )
            try:
                return func(
                    request,
                    timeout=timeout,
                    metadata=_metadata,
                    credentials=credentials,
                    wait_for_ready=wait_for_ready,
                    compression=compression,
                )
            except _InactiveRpcError as e:
                raise PineconeException(e._state.debug_error_string) from e

        return wrapped()

    def _request_metadata(self) -> Dict[str, str]:
        return {REQUEST_ID: _generate_request_id()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
