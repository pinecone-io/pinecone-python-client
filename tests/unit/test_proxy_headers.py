"""Tests for proxy_headers configuration."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from pinecone import AsyncPinecone, Pinecone


class TestProxyHeadersPassedToHttpxClient:
    """Verify proxy headers are forwarded to the httpx Client as an httpx.Proxy."""

    def test_proxy_headers_passed_to_httpx_client(self) -> None:
        headers = {"Proxy-Authorization": "Basic dXNlcjpwYXNz"}
        with patch("pinecone._internal.http_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client_cls
            Pinecone(
                api_key="test-key",
                proxy_url="http://proxy.example.com:8080",
                proxy_headers=headers,
            )
            call_kwargs = mock_client_cls.call_args[1]
            proxy = call_kwargs["proxy"]
            assert isinstance(proxy, httpx.Proxy)
            assert proxy.url == httpx.URL("http://proxy.example.com:8080")
            assert proxy.headers["proxy-authorization"] == "Basic dXNlcjpwYXNz"


class TestProxyHeadersDefaultEmpty:
    """When no proxy_headers are given, a bare proxy URL string is used."""

    def test_proxy_headers_default_empty(self) -> None:
        with patch("pinecone._internal.http_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client_cls
            Pinecone(
                api_key="test-key",
                proxy_url="http://proxy.example.com:8080",
            )
            call_kwargs = mock_client_cls.call_args[1]
            proxy = call_kwargs["proxy"]
            assert proxy == "http://proxy.example.com:8080"

    def test_no_proxy_url_means_no_proxy(self) -> None:
        with patch("pinecone._internal.http_client.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value = mock_client_cls
            Pinecone(api_key="test-key")
            call_kwargs = mock_client_cls.call_args[1]
            assert call_kwargs["proxy"] is None


class TestAsyncProxyHeadersRaises:
    """AsyncPinecone rejects proxy_headers with NotImplementedError."""

    def test_async_proxy_headers_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="proxy_headers is not yet supported"):
            AsyncPinecone(
                api_key="test-key",
                proxy_headers={"Proxy-Authorization": "Basic dXNlcjpwYXNz"},
            )

    def test_async_empty_proxy_headers_ok(self) -> None:
        """Empty dict should not raise."""
        pc = AsyncPinecone(api_key="test-key", proxy_headers={})
        assert pc.config.api_key == "test-key"

    def test_async_none_proxy_headers_ok(self) -> None:
        """None should not raise."""
        pc = AsyncPinecone(api_key="test-key", proxy_headers=None)
        assert pc.config.api_key == "test-key"
