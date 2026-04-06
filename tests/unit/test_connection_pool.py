"""Tests for configurable connection pool limits."""

from __future__ import annotations

from unittest.mock import patch

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import (
    AsyncHTTPClient,
    HTTPClient,
    _default_pool_size,
)


def _get_sync_max_connections(client: HTTPClient) -> int:
    """Extract max_connections from the sync client's transport chain."""
    retry_transport = client._client._transport
    http_transport = retry_transport._transport  # type: ignore[union-attr]
    return http_transport._pool._max_connections  # type: ignore[union-attr]


def _get_async_max_connections(client: AsyncHTTPClient) -> int:
    """Extract max_connections from the async client's transport chain."""
    async_client = client._ensure_client()
    retry_transport = async_client._transport
    http_transport = retry_transport._transport  # type: ignore[union-attr]
    return http_transport._pool._max_connections  # type: ignore[union-attr]


class TestDefaultPoolSize:
    def test_default_pool_size_is_5x_cpu(self) -> None:
        with patch("pinecone._internal.http_client.os.cpu_count", return_value=4):
            assert _default_pool_size() == 20

    def test_default_pool_size_floor_20(self) -> None:
        with patch("pinecone._internal.http_client.os.cpu_count", return_value=1):
            assert _default_pool_size() == 20

    def test_default_pool_size_none_cpu_count(self) -> None:
        with patch("pinecone._internal.http_client.os.cpu_count", return_value=None):
            assert _default_pool_size() == 20

    def test_default_pool_size_high_cpu(self) -> None:
        with patch("pinecone._internal.http_client.os.cpu_count", return_value=16):
            assert _default_pool_size() == 80


class TestSyncClientPool:
    def test_sync_client_uses_default_pool(self) -> None:
        config = PineconeConfig(api_key="test-key")
        client = HTTPClient(config, api_version="2025-10")
        assert _get_sync_max_connections(client) == _default_pool_size()
        client.close()

    def test_sync_client_uses_custom_pool(self) -> None:
        config = PineconeConfig(api_key="test-key", connection_pool_maxsize=50)
        client = HTTPClient(config, api_version="2025-10")
        assert _get_sync_max_connections(client) == 50
        client.close()


class TestAsyncClientPool:
    def test_async_client_uses_default_pool(self) -> None:
        config = PineconeConfig(api_key="test-key")
        client = AsyncHTTPClient(config, api_version="2025-10")
        assert _get_async_max_connections(client) == _default_pool_size()

    def test_async_client_uses_custom_pool(self) -> None:
        config = PineconeConfig(api_key="test-key", connection_pool_maxsize=50)
        client = AsyncHTTPClient(config, api_version="2025-10")
        assert _get_async_max_connections(client) == 50


class TestPineconeConfigPool:
    def test_pinecone_config_pool_size_default(self) -> None:
        config = PineconeConfig(api_key="test-key")
        assert config.connection_pool_maxsize == 0

    def test_pinecone_config_repr_includes_pool(self) -> None:
        config = PineconeConfig(api_key="test-key")
        assert "connection_pool_maxsize" in repr(config)
