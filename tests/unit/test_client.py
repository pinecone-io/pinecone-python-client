"""Unit tests for the Pinecone client constructor and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone import Pinecone
from pinecone._internal.constants import DEFAULT_BASE_URL
from pinecone.errors.exceptions import ValidationError


class TestBasicConstruction:
    """Test basic client construction with explicit arguments."""

    def test_create_with_api_key(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc.config.api_key == "test-key"

    def test_default_host(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc.config.host == DEFAULT_BASE_URL

    def test_custom_host(self) -> None:
        pc = Pinecone(api_key="test-key", host="https://custom.example.com")
        assert pc.config.host == "https://custom.example.com"

    def test_custom_timeout(self) -> None:
        pc = Pinecone(api_key="test-key", timeout=60.0)
        assert pc.config.timeout == 60.0

    def test_additional_headers(self) -> None:
        headers = {"X-Custom": "value"}
        pc = Pinecone(api_key="test-key", additional_headers=headers)
        assert pc.config.additional_headers == {"X-Custom": "value"}

    def test_proxy_url(self) -> None:
        pc = Pinecone(api_key="test-key", proxy_url="http://proxy:8080")
        assert pc.config.proxy_url == "http://proxy:8080"

    def test_ssl_ca_certs(self) -> None:
        pc = Pinecone(api_key="test-key", ssl_ca_certs="/path/to/certs.pem")
        assert pc.config.ssl_ca_certs == "/path/to/certs.pem"

    def test_ssl_verify_default_true(self) -> None:
        pc = Pinecone(api_key="test-key")
        assert pc.config.ssl_verify is True

    def test_ssl_verify_false(self) -> None:
        pc = Pinecone(api_key="test-key", ssl_verify=False)
        assert pc.config.ssl_verify is False


class TestEnvVarFallback:
    """Test environment variable fallbacks."""

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = Pinecone()
        assert pc.config.api_key == "env-api-key"

    def test_explicit_api_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
        pc = Pinecone(api_key="explicit-key")
        assert pc.config.api_key == "explicit-key"

    def test_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "https://custom-host.example.com")
        pc = Pinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_explicit_host_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "https://env-host.example.com")
        pc = Pinecone(api_key="test-key", host="https://explicit-host.example.com")
        assert pc.config.host == "https://explicit-host.example.com"

    def test_host_env_gets_https_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_CONTROLLER_HOST", "custom-host.example.com")
        pc = Pinecone(api_key="test-key")
        assert pc.config.host == "https://custom-host.example.com"

    def test_additional_headers_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = Pinecone(api_key="test-key")
        assert pc.config.additional_headers == {"X-Env": "from-env"}

    def test_explicit_headers_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_ADDITIONAL_HEADERS", '{"X-Env": "from-env"}')
        pc = Pinecone(api_key="test-key", additional_headers={"X-Custom": "explicit"})
        assert pc.config.additional_headers == {"X-Custom": "explicit"}


class TestValidation:
    """Test validation errors."""

    def test_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            Pinecone()

    def test_no_api_key_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError) as exc_info:
            Pinecone()
        assert "PINECONE_API_KEY" in str(exc_info.value)


class TestHostNormalization:
    """Test host normalization behavior."""

    def test_bare_hostname_gets_https(self) -> None:
        pc = Pinecone(api_key="test-key", host="custom.example.com")
        assert pc.config.host == "https://custom.example.com"

    def test_https_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", host="https://custom.example.com")
        assert pc.config.host == "https://custom.example.com"

    def test_http_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", host="http://localhost:8080")
        assert pc.config.host == "http://localhost:8080"

    def test_idempotent(self) -> None:
        pc = Pinecone(api_key="test-key", host="https://already-prefixed.com")
        assert pc.config.host == "https://already-prefixed.com"


class TestSourceTagNormalization:
    """Test source tag normalization."""

    def test_uppercase_lowered(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="MyApp")
        assert pc.config.source_tag == "myapp"

    def test_special_chars_stripped(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="my-app@v2!")
        assert pc.config.source_tag == "myappv2"

    def test_spaces_to_underscores(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="my app name")
        assert pc.config.source_tag == "my_app_name"

    def test_colons_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="app:v2")
        assert pc.config.source_tag == "app:v2"

    def test_underscores_preserved(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="my_app")
        assert pc.config.source_tag == "my_app"

    def test_combined_normalization(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag="My App@V2! test:tag")
        assert pc.config.source_tag == "my_appv2_test:tag"

    def test_none_source_tag(self) -> None:
        pc = Pinecone(api_key="test-key", source_tag=None)
        assert pc.config.source_tag == ""


class TestDeprecatedKwargs:
    """Test that deprecated kwargs raise errors."""

    def test_openapi_config_raises(self) -> None:
        with pytest.raises(ValidationError, match="no longer supported"):
            Pinecone(api_key="test-key", openapi_config="something")

    def test_pool_threads_raises(self) -> None:
        with pytest.raises(ValidationError, match="no longer supported"):
            Pinecone(api_key="test-key", pool_threads=4)

    def test_index_api_raises(self) -> None:
        with pytest.raises(ValidationError, match="no longer supported"):
            Pinecone(api_key="test-key", index_api="something")

    def test_multiple_deprecated_raises(self) -> None:
        with pytest.raises(ValidationError, match="no longer supported"):
            Pinecone(api_key="test-key", openapi_config="x", pool_threads=4)

    def test_error_mentions_migration(self) -> None:
        with pytest.raises(ValidationError, match="migration guide"):
            Pinecone(api_key="test-key", openapi_config="x")


class TestContextManager:
    """Test context manager protocol."""

    def test_context_manager(self) -> None:
        with Pinecone(api_key="test-key") as pc:
            assert pc.config.api_key == "test-key"

    def test_close(self) -> None:
        pc = Pinecone(api_key="test-key")
        pc.close()


class TestIndexFactory:
    """Test Pinecone.index() factory method."""

    @patch("pinecone.index.Index")
    def test_host_based_skips_describe(self, mock_index_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key", additional_headers={"X-Test": "val"}, timeout=45.0)
        mock_idx = MagicMock()
        mock_index_cls.return_value = mock_idx

        result = pc.index(host="foo.svc.pinecone.io")

        assert result is mock_idx
        mock_index_cls.assert_called_once_with(
            host="foo.svc.pinecone.io",
            api_key="test-key",
            additional_headers={"X-Test": "val"},
            timeout=45.0,
            proxy_url="",
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone.index.Index")
    def test_name_cached_returns_index_without_describe(self, mock_index_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key", timeout=20.0)
        pc._host_cache["my-index"] = "cached.host.pinecone.io"
        mock_idx = MagicMock()
        mock_index_cls.return_value = mock_idx

        result = pc.index(name="my-index")

        assert result is mock_idx
        mock_index_cls.assert_called_once_with(
            host="cached.host.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=20.0,
            proxy_url="",
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone.index.Index")
    def test_name_describe_resolves_host_and_caches(self, mock_index_cls: MagicMock) -> None:
        pc = Pinecone(api_key="test-key")
        mock_idx = MagicMock()
        mock_index_cls.return_value = mock_idx

        mock_desc = MagicMock()
        mock_desc.host = "resolved.host.pinecone.io"

        mock_indexes = MagicMock()
        mock_indexes.describe.return_value = mock_desc
        pc._indexes = mock_indexes

        result = pc.index(name="my-index")

        assert result is mock_idx
        mock_indexes.describe.assert_called_once_with("my-index")
        assert pc._host_cache["my-index"] == "resolved.host.pinecone.io"
        mock_index_cls.assert_called_once_with(
            host="resolved.host.pinecone.io",
            api_key="test-key",
            additional_headers={},
            timeout=30.0,
            proxy_url="",
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    def test_no_name_or_host_raises_validation_error(self) -> None:
        pc = Pinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host"):
            pc.index()

    @patch("pinecone.index.Index")
    def test_host_based_passes_config_fields(self, mock_index_cls: MagicMock) -> None:
        headers = {"X-Custom": "header-val"}
        pc = Pinecone(api_key="my-api-key", additional_headers=headers, timeout=99.0)
        mock_index_cls.return_value = MagicMock()

        pc.index(host="host.svc.pinecone.io")

        mock_index_cls.assert_called_once_with(
            host="host.svc.pinecone.io",
            api_key="my-api-key",
            additional_headers={"X-Custom": "header-val"},
            timeout=99.0,
            proxy_url="",
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone.index.Index")
    def test_name_cached_passes_config_fields(self, mock_index_cls: MagicMock) -> None:
        headers = {"X-Custom": "header-val"}
        pc = Pinecone(api_key="my-api-key", additional_headers=headers, timeout=99.0)
        pc._host_cache["idx"] = "cached.host"
        mock_index_cls.return_value = MagicMock()

        pc.index(name="idx")

        mock_index_cls.assert_called_once_with(
            host="cached.host",
            api_key="my-api-key",
            additional_headers={"X-Custom": "header-val"},
            timeout=99.0,
            proxy_url="",
            ssl_ca_certs=None,
            ssl_verify=True,
            source_tag="",
            connection_pool_maxsize=0,
        )

    @patch("pinecone._internal.http_client.HTTPClient")
    @patch("pinecone.index.Index")
    def test_index_propagates_proxy_ssl_source_tag(
        self, mock_index_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = Pinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        mock_index_cls.return_value = MagicMock()

        pc.index(host="foo.svc.pinecone.io")

        mock_index_cls.assert_called_once_with(
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

    @patch("pinecone._internal.http_client.HTTPClient")
    @patch("pinecone.index.Index")
    def test_index_by_name_propagates_proxy_ssl_source_tag(
        self, mock_index_cls: MagicMock, _mock_http: MagicMock
    ) -> None:
        pc = Pinecone(
            api_key="test-key",
            proxy_url="http://proxy:8080",
            ssl_ca_certs="/path/to/certs.pem",
            ssl_verify=False,
            source_tag="my_app",
            connection_pool_maxsize=10,
        )
        pc._host_cache["my-index"] = "cached.host.pinecone.io"
        mock_index_cls.return_value = MagicMock()

        pc.index(name="my-index")

        mock_index_cls.assert_called_once_with(
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
