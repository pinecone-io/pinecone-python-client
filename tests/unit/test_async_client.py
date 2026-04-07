"""Unit tests for AsyncPinecone client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import ValidationError


def test_async_pinecone_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="No API key"):
        AsyncPinecone()


def test_async_pinecone_accepts_api_key() -> None:
    pc = AsyncPinecone(api_key="test-key")
    assert pc.config.api_key == "test-key"


def test_async_pinecone_deprecated_kwargs() -> None:
    with pytest.raises(ValidationError, match="no longer supported"):
        AsyncPinecone(api_key="test-key", openapi_config={})


def test_async_pinecone_default_host() -> None:
    pc = AsyncPinecone(api_key="test-key")
    assert pc.config.host == "https://api.pinecone.io"


def test_async_pinecone_custom_host() -> None:
    pc = AsyncPinecone(api_key="test-key", host="https://custom.pinecone.io")
    assert pc.config.host == "https://custom.pinecone.io"


def test_async_pinecone_indexes_property() -> None:
    pc = AsyncPinecone(api_key="test-key")
    from pinecone.async_client.indexes import AsyncIndexes

    indexes = pc.indexes
    assert isinstance(indexes, AsyncIndexes)
    # Verify lazy caching — same instance returned
    assert pc.indexes is indexes


async def test_async_pinecone_context_manager() -> None:
    async with AsyncPinecone(api_key="test-key") as pc:
        assert pc.config.api_key == "test-key"


async def test_async_pinecone_close() -> None:
    pc = AsyncPinecone(api_key="test-key")
    await pc.close()


class TestEnvVarFallback:
    """Test environment variable fallbacks for AsyncPinecone."""

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = AsyncPinecone()
        assert pc.config.api_key == "env-api-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = AsyncPinecone(api_key="explicit-key")
        assert pc.config.api_key == "explicit-key"

    def test_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "https://custom-host.example.com")
        pc = AsyncPinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_host_env_gets_https_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "custom-host.example.com")
        pc = AsyncPinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_additional_headers_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = AsyncPinecone(api_key="test-key")
        assert pc.config.additional_headers == {"X-Env": "from-env"}

    def test_explicit_headers_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = AsyncPinecone(api_key="test-key", additional_headers={"X-Custom": "explicit"})
        assert pc.config.additional_headers == {"X-Custom": "explicit"}

    def test_source_tag_normalization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-key")
        pc = AsyncPinecone(source_tag="My App@V2! test:tag")
        assert pc.config.source_tag == "my_appv2_test:tag"

    def test_proxy_url_param(self) -> None:
        pc = AsyncPinecone(api_key="test-key", proxy_url="http://proxy:8080")
        assert pc.config.proxy_url == "http://proxy:8080"

    def test_ssl_ca_certs_param(self) -> None:
        pc = AsyncPinecone(api_key="test-key", ssl_ca_certs="/path/to/certs.pem")
        assert pc.config.ssl_ca_certs == "/path/to/certs.pem"

    def test_ssl_verify_param(self) -> None:
        pc = AsyncPinecone(api_key="test-key", ssl_verify=False)
        assert pc.config.ssl_verify is False


class TestAsyncIndexFactory:
    """Test AsyncPinecone.index() propagates config to AsyncIndex."""

    @patch("pinecone._internal.http_client.AsyncHTTPClient")
    @patch("pinecone.async_client.async_index.AsyncIndex")
    def test_index_by_host_propagates_config(
        self, mock_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = AsyncPinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        mock_cls.return_value = MagicMock()

        pc.index(host="foo.svc.pinecone.io")

        mock_cls.assert_called_once_with(
            host="foo.svc.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )

    @patch("pinecone._internal.http_client.AsyncHTTPClient")
    @patch("pinecone.async_client.async_index.AsyncIndex")
    def test_index_by_name_cached_propagates_config(
        self, mock_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = AsyncPinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        pc._host_cache["my-index"] = "cached.host.pinecone.io"
        mock_cls.return_value = MagicMock()

        pc.index(name="my-index")

        mock_cls.assert_called_once_with(
            host="cached.host.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )

    @patch("pinecone.async_client.async_index.AsyncIndex")
    def test_index_defaults_propagated(self, mock_cls: MagicMock) -> None:
        pc = AsyncPinecone(api_key="test-key")
        mock_cls.return_value = MagicMock()

        pc.index(host="foo.svc.pinecone.io")

        mock_cls.assert_called_once_with(
            host="foo.svc.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="",
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )
