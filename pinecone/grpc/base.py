from abc import ABC, abstractmethod
from typing import Optional

import grpc
from grpc._channel import Channel

from .channel_factory import GrpcChannelFactory

from pinecone import Config
from .config import GRPCClientConfig
from .grpc_runner import GrpcRunner
from .utils import normalize_endpoint


class GRPCIndexBase(ABC):
    """
    Base class for grpc-based interaction with Pinecone indexes
    """

    def __init__(
        self,
        index_name: str,
        config: Config,
        channel: Optional[Channel] = None,
        grpc_config: Optional[GRPCClientConfig] = None,
        _endpoint_override: Optional[str] = None,
        use_asyncio: Optional[bool] = False,
    ):
        self.config = config
        self.grpc_client_config = grpc_config or GRPCClientConfig()
        self._endpoint_override = _endpoint_override

        self.runner = GrpcRunner(
            index_name=index_name, config=config, grpc_config=self.grpc_client_config
        )
        self.channel_factory = GrpcChannelFactory(
            config=self.config, grpc_client_config=self.grpc_client_config, use_asyncio=use_asyncio
        )
        self._channel = channel or self._gen_channel()
        self.stub = self.stub_class(self._channel)

    @property
    @abstractmethod
    def stub_class(self):
        pass

    def _endpoint(self):
        grpc_host = normalize_endpoint(self.config.host)
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.close()
        return True
