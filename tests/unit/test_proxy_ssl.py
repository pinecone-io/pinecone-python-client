"""Tests for proxy and SSL configuration wiring in HTTPClient and AsyncHTTPClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone._internal.http_client import AsyncHTTPClient, HTTPClient


API_VERSION = "2025-10"


def _make_config(**overrides: object) -> PineconeConfig:
    defaults = {"api_key": "test-key", "host": "https://api.example.com"}
    defaults.update(overrides)
    return PineconeConfig(**defaults)  # type: ignore[arg-type]


class TestSyncProxyUrl:
    @patch("pinecone._internal.http_client.httpx.Client")
    def test_sync_proxy_url_passed(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(proxy_url="http://proxy:8080")
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_client_cls.call_args
        assert kwargs["proxy"] == "http://proxy:8080"

    @patch("pinecone._internal.http_client.httpx.Client")
    def test_sync_no_proxy_by_default(self, mock_client_cls: MagicMock) -> None:
        config = _make_config()
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_client_cls.call_args
        assert kwargs["proxy"] is None


class TestSyncSSL:
    @patch("pinecone._internal.http_client.httpx.Client")
    def test_sync_ssl_ca_certs_passed(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(ssl_ca_certs="/path/to/cert.pem")
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_client_cls.call_args
        assert kwargs["verify"] == "/path/to/cert.pem"

    @patch("pinecone._internal.http_client.httpx.Client")
    def test_sync_ssl_verify_false(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(ssl_verify=False)
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_client_cls.call_args
        assert kwargs["verify"] is False

    @patch("pinecone._internal.http_client.httpx.Client")
    def test_sync_ssl_verify_true_by_default(self, mock_client_cls: MagicMock) -> None:
        config = _make_config()
        HTTPClient(config, API_VERSION)
        _, kwargs = mock_client_cls.call_args
        assert kwargs["verify"] is True


class TestAsyncProxyUrl:
    @patch("pinecone._internal.http_client.httpx.AsyncClient")
    def test_async_proxy_url_passed(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(proxy_url="http://proxy:8080")
        client = AsyncHTTPClient(config, API_VERSION)
        client._ensure_client()
        _, kwargs = mock_client_cls.call_args
        assert kwargs["proxy"] == "http://proxy:8080"


class TestAsyncSSL:
    @patch("pinecone._internal.http_client.httpx.AsyncClient")
    def test_async_ssl_ca_certs_passed(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(ssl_ca_certs="/path/to/cert.pem")
        client = AsyncHTTPClient(config, API_VERSION)
        client._ensure_client()
        _, kwargs = mock_client_cls.call_args
        assert kwargs["verify"] == "/path/to/cert.pem"

    @patch("pinecone._internal.http_client.httpx.AsyncClient")
    def test_async_ssl_verify_false(self, mock_client_cls: MagicMock) -> None:
        config = _make_config(ssl_verify=False)
        client = AsyncHTTPClient(config, API_VERSION)
        client._ensure_client()
        _, kwargs = mock_client_cls.call_args
        assert kwargs["verify"] is False
