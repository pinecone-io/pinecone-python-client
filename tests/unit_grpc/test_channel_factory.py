import grpc
import re
import pytest
from unittest.mock import patch, MagicMock, ANY

from pinecone import Config
from pinecone.grpc.channel_factory import GrpcChannelFactory, GRPCClientConfig
from pinecone.utils.constants import MAX_MSG_SIZE


@pytest.fixture
def config():
    return Config(ssl_ca_certs=None, proxy_url=None)


@pytest.fixture
def grpc_client_config():
    return GRPCClientConfig(secure=True)


class TestGrpcChannelFactory:
    def test_create_secure_channel_with_default_settings(self, config, grpc_client_config):
        factory = GrpcChannelFactory(
            config=config, grpc_client_config=grpc_client_config, use_asyncio=False
        )
        endpoint = "test.endpoint:443"

        with (
            patch("grpc.secure_channel") as mock_secure_channel,
            patch("certifi.where", return_value="/path/to/certifi/cacert.pem"),
            patch("builtins.open", new_callable=MagicMock) as mock_open,
        ):
            # Mock the file object to return bytes when read() is called
            mock_file = MagicMock()
            mock_file.read.return_value = b"mocked_cert_data"
            mock_open.return_value = mock_file
            channel = factory.create_channel(endpoint)

            mock_secure_channel.assert_called_once()
            assert mock_secure_channel.call_args[0][0] == endpoint
            assert isinstance(mock_secure_channel.call_args[1]["options"], tuple)

            options = dict(mock_secure_channel.call_args[1]["options"])
            assert options["grpc.ssl_target_name_override"] == "test.endpoint"
            assert options["grpc.max_send_message_length"] == MAX_MSG_SIZE
            assert options["grpc.per_rpc_retry_buffer_size"] == MAX_MSG_SIZE
            assert options["grpc.max_receive_message_length"] == MAX_MSG_SIZE
            assert "grpc.service_config" in options
            assert options["grpc.enable_retries"] is True
            assert (
                re.search(
                    r"python-client\[grpc\]-\d+\.\d+\.\d+", options["grpc.primary_user_agent"]
                )
                is not None
            )

            assert isinstance(channel, MagicMock)

    def test_create_secure_channel_with_proxy(self):
        grpc_client_config = GRPCClientConfig(secure=True)
        config = Config(proxy_url="http://test.proxy:8080")
        factory = GrpcChannelFactory(
            config=config, grpc_client_config=grpc_client_config, use_asyncio=False
        )
        endpoint = "test.endpoint:443"

        with patch("grpc.secure_channel") as mock_secure_channel:
            channel = factory.create_channel(endpoint)

            mock_secure_channel.assert_called_once()
            assert "grpc.http_proxy" in dict(mock_secure_channel.call_args[1]["options"])
            assert (
                "http://test.proxy:8080"
                == dict(mock_secure_channel.call_args[1]["options"])["grpc.http_proxy"]
            )
            assert isinstance(channel, MagicMock)

    def test_create_insecure_channel(self, config):
        grpc_client_config = GRPCClientConfig(secure=False)
        factory = GrpcChannelFactory(
            config=config, grpc_client_config=grpc_client_config, use_asyncio=False
        )
        endpoint = "test.endpoint:50051"

        with patch("grpc.insecure_channel") as mock_insecure_channel:
            channel = factory.create_channel(endpoint)

            mock_insecure_channel.assert_called_once_with(endpoint, options=ANY)
            assert isinstance(channel, MagicMock)


class TestGrpcChannelFactoryAsyncio:
    def test_create_secure_channel_with_default_settings(self, config, grpc_client_config):
        factory = GrpcChannelFactory(
            config=config, grpc_client_config=grpc_client_config, use_asyncio=True
        )
        endpoint = "test.endpoint:443"

        with (
            patch("grpc.aio.secure_channel") as mock_secure_aio_channel,
            patch("certifi.where", return_value="/path/to/certifi/cacert.pem"),
            patch("builtins.open", new_callable=MagicMock) as mock_open,
        ):
            # Mock the file object to return bytes when read() is called
            mock_file = MagicMock()
            mock_file.read.return_value = b"mocked_cert_data"
            mock_open.return_value = mock_file
            channel = factory.create_channel(endpoint)

            mock_secure_aio_channel.assert_called_once()
            assert mock_secure_aio_channel.call_args[0][0] == endpoint
            assert isinstance(mock_secure_aio_channel.call_args[1]["options"], tuple)

            options = dict(mock_secure_aio_channel.call_args[1]["options"])
            assert options["grpc.ssl_target_name_override"] == "test.endpoint"
            assert options["grpc.max_send_message_length"] == MAX_MSG_SIZE
            assert options["grpc.per_rpc_retry_buffer_size"] == MAX_MSG_SIZE
            assert options["grpc.max_receive_message_length"] == MAX_MSG_SIZE
            assert "grpc.service_config" in options
            assert options["grpc.enable_retries"] is True
            assert (
                re.search(
                    r"python-client\[grpc\]-\d+\.\d+\.\d+", options["grpc.primary_user_agent"]
                )
                is not None
            )

            security_credentials = mock_secure_aio_channel.call_args[1]["credentials"]
            assert security_credentials is not None
            assert isinstance(security_credentials, grpc.ChannelCredentials)

            assert isinstance(channel, MagicMock)

    def test_create_insecure_channel_asyncio(self, config):
        grpc_client_config = GRPCClientConfig(secure=False)
        factory = GrpcChannelFactory(
            config=config, grpc_client_config=grpc_client_config, use_asyncio=True
        )
        endpoint = "test.endpoint:50051"

        with patch("grpc.aio.insecure_channel") as mock_aio_insecure_channel:
            channel = factory.create_channel(endpoint)

            mock_aio_insecure_channel.assert_called_once_with(endpoint, options=ANY)
            assert isinstance(channel, MagicMock)
