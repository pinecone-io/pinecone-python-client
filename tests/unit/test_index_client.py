"""Unit tests for Index data plane client and Pinecone.index() factory."""

from __future__ import annotations

import logging
import warnings

import httpx
import pytest
import respx

from pinecone import Index, Pinecone
from pinecone.errors.exceptions import ValidationError
from tests.factories import make_index_response

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
CONTROL_PLANE_URL = "https://api.pinecone.io"


# ---------------------------------------------------------------------------
# Direct construction
# ---------------------------------------------------------------------------


class TestDirectConstruction:
    """Test creating Index directly with host URL."""

    def test_create_with_host(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key")
        assert idx.host == INDEX_HOST_HTTPS

    def test_host_with_https_preserved(self) -> None:
        idx = Index(host=INDEX_HOST_HTTPS, api_key="test-key")
        assert idx.host == INDEX_HOST_HTTPS

    def test_host_with_http_preserved(self) -> None:
        idx = Index(host="http://localhost:8080", api_key="test-key")
        assert idx.host == "http://localhost:8080"

    def test_repr(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key")
        assert "Index(" in repr(idx)
        assert INDEX_HOST_HTTPS in repr(idx)


class TestApiKeyResolution:
    """Test API key resolution from arg and env var."""

    def test_explicit_api_key(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="explicit-key")
        assert idx._config.api_key == "explicit-key"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-key")
        idx = Index(host=INDEX_HOST)
        assert idx._config.api_key == "env-key"

    def test_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            Index(host=INDEX_HOST)

    def test_api_key_checked_before_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing API key is checked before host validation (unified-ord-0001)."""
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            Index(host="", api_key=None)


class TestHostValidation:
    """Test host validation and normalization."""

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValidationError, match="host must be a non-empty string"):
            Index(host="", api_key="test-key")

    def test_whitespace_host_raises(self) -> None:
        with pytest.raises(ValidationError, match="host must be a non-empty string"):
            Index(host="   ", api_key="test-key")

    def test_host_without_dot_raises(self) -> None:
        """Host must contain a dot or 'localhost' (unified-index-0043)."""
        with pytest.raises(ValidationError, match="does not appear to be a valid URL"):
            Index(host="just-a-word", api_key="test-key")

    def test_localhost_accepted(self) -> None:
        idx = Index(host="http://localhost:8080", api_key="test-key")
        assert idx.host == "http://localhost:8080"

    def test_localhost_no_port_accepted(self) -> None:
        idx = Index(host="http://localhost", api_key="test-key")
        assert idx.host == "http://localhost"

    def test_bare_hostname_gets_https(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key")
        assert idx.host.startswith("https://")


class TestContextManager:
    """Test context manager protocol for resource cleanup."""

    def test_context_manager_returns_self(self) -> None:
        with Index(host=INDEX_HOST, api_key="test-key") as idx:
            assert isinstance(idx, Index)

    def test_close(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key")
        idx.close()  # Should not raise


# ---------------------------------------------------------------------------
# Pinecone.index() factory
# ---------------------------------------------------------------------------


class TestFactoryWithHost:
    """Test Pinecone.index(host=...) — direct construction, no describe call."""

    @respx.mock
    def test_creates_index_with_host(self) -> None:
        pc = Pinecone(api_key="test-key")
        idx = pc.index(host=INDEX_HOST)
        assert isinstance(idx, Index)
        assert idx.host == INDEX_HOST_HTTPS

    @respx.mock
    def test_no_describe_call_when_host_provided(self) -> None:
        """When host is given, no HTTP call should be made."""
        route = respx.get(f"{CONTROL_PLANE_URL}/indexes/anything")
        pc = Pinecone(api_key="test-key")
        pc.index(host=INDEX_HOST)
        assert not route.called

    @respx.mock
    def test_forwards_config(self) -> None:
        pc = Pinecone(
            api_key="forwarded-key",
            timeout=60.0,
            additional_headers={"X-Custom": "val"},
        )
        idx = pc.index(host=INDEX_HOST)
        assert idx._config.api_key == "forwarded-key"
        assert idx._config.timeout == 60.0
        assert idx._config.additional_headers == {"X-Custom": "val"}


class TestFactoryWithName:
    """Test Pinecone.index(name=...) — triggers describe to resolve host."""

    @respx.mock
    def test_resolves_host_via_describe(self) -> None:
        respx.get(f"{CONTROL_PLANE_URL}/indexes/my-index").mock(
            return_value=httpx.Response(200, json=make_index_response(name="my-index")),
        )
        pc = Pinecone(api_key="test-key")
        idx = pc.index(name="my-index")
        assert isinstance(idx, Index)
        assert INDEX_HOST in idx.host

    @respx.mock
    def test_host_cached_after_describe(self) -> None:
        """Second call with same name should use cache, not call describe again."""
        route = respx.get(f"{CONTROL_PLANE_URL}/indexes/my-index").mock(
            return_value=httpx.Response(200, json=make_index_response(name="my-index")),
        )
        pc = Pinecone(api_key="test-key")

        idx1 = pc.index(name="my-index")
        idx2 = pc.index(name="my-index")

        assert route.call_count == 1
        assert idx1.host == idx2.host

    @respx.mock
    def test_different_names_each_call_describe(self) -> None:
        respx.get(f"{CONTROL_PLANE_URL}/indexes/index-a").mock(
            return_value=httpx.Response(
                200,
                json=make_index_response(name="index-a", host="index-a-host.svc.pinecone.io"),
            ),
        )
        respx.get(f"{CONTROL_PLANE_URL}/indexes/index-b").mock(
            return_value=httpx.Response(
                200,
                json=make_index_response(name="index-b", host="index-b-host.svc.pinecone.io"),
            ),
        )
        pc = Pinecone(api_key="test-key")

        idx_a = pc.index(name="index-a")
        idx_b = pc.index(name="index-b")

        assert "index-a-host" in idx_a.host
        assert "index-b-host" in idx_b.host


class TestFactoryValidation:
    """Test Pinecone.index() validation."""

    def test_no_args_raises(self) -> None:
        """unified-index-0041: neither name nor host raises ValidationError."""
        pc = Pinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host must be provided"):
            pc.index()

    def test_empty_strings_raises(self) -> None:
        pc = Pinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host must be provided"):
            pc.index(name="", host="")


# ---------------------------------------------------------------------------
# pool_threads= backcompat shim (BCG-072)
# ---------------------------------------------------------------------------


class TestPoolThreadsBackcompat:
    def test_pool_threads_kwarg_accepted_silently(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert idx is not None

    def test_pool_threads_kwarg_emits_debug_log(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="pinecone.index"):
            Index(host=INDEX_HOST, api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert any(
            "pool_threads" in r.message and "connection_pool_maxsize" in r.message
            for r in caplog.records
        )

    def test_pool_threads_kwarg_does_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            Index(host=INDEX_HOST, api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert len(record) == 0

    def test_unknown_kwarg_still_rejected(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            Index(host=INDEX_HOST, api_key="test-key", bogus=True)  # type: ignore[call-arg]
