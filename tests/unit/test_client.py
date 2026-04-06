"""Unit tests for the Pinecone client constructor and configuration."""

from __future__ import annotations

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
