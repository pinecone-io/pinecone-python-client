"""Unit tests for AsyncIndex data plane client and AsyncPinecone.index() factory."""

from __future__ import annotations

import httpx
import pytest
import respx

from pinecone import AsyncIndex
from pinecone.async_client.pinecone import AsyncPinecone
from pinecone.errors.exceptions import ValidationError
from tests.factories import make_index_response

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
CONTROL_PLANE_URL = "https://api.pinecone.io"


# ---------------------------------------------------------------------------
# Direct construction
# ---------------------------------------------------------------------------


class TestDirectConstruction:
    """Test creating AsyncIndex directly with host URL."""

    def test_create_with_host(self) -> None:
        idx = AsyncIndex(host=INDEX_HOST, api_key="test-key")
        assert idx.host == INDEX_HOST_HTTPS

    def test_host_with_https_preserved(self) -> None:
        idx = AsyncIndex(host=INDEX_HOST_HTTPS, api_key="test-key")
        assert idx.host == INDEX_HOST_HTTPS

    def test_host_with_http_preserved(self) -> None:
        idx = AsyncIndex(host="http://localhost:8080", api_key="test-key")
        assert idx.host == "http://localhost:8080"

    def test_repr(self) -> None:
        idx = AsyncIndex(host=INDEX_HOST, api_key="test-key")
        assert "AsyncIndex(" in repr(idx)
        assert INDEX_HOST_HTTPS in repr(idx)


class TestApiKeyResolution:
    """Test API key resolution from arg and env var."""

    def test_explicit_api_key(self) -> None:
        idx = AsyncIndex(host=INDEX_HOST, api_key="explicit-key")
        assert idx._config.api_key == "explicit-key"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PINECONE_API_KEY", "env-key")
        idx = AsyncIndex(host=INDEX_HOST)
        assert idx._config.api_key == "env-key"

    def test_no_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            AsyncIndex(host=INDEX_HOST)

    def test_api_key_checked_before_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing API key is checked before host validation (unified-ord-0001)."""
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="No API key provided"):
            AsyncIndex(host="", api_key=None)


class TestHostValidation:
    """Test host validation and normalization."""

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValidationError, match="host must be a non-empty string"):
            AsyncIndex(host="", api_key="test-key")

    def test_whitespace_host_raises(self) -> None:
        with pytest.raises(ValidationError, match="host must be a non-empty string"):
            AsyncIndex(host="   ", api_key="test-key")

    def test_host_without_dot_raises(self) -> None:
        with pytest.raises(ValidationError, match="does not appear to be a valid URL"):
            AsyncIndex(host="just-a-word", api_key="test-key")

    def test_localhost_accepted(self) -> None:
        idx = AsyncIndex(host="http://localhost:8080", api_key="test-key")
        assert idx.host == "http://localhost:8080"

    def test_bare_hostname_gets_https(self) -> None:
        idx = AsyncIndex(host=INDEX_HOST, api_key="test-key")
        assert idx.host.startswith("https://")


class TestAsyncContextManager:
    """Test async context manager protocol for resource cleanup."""

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self) -> None:
        async with AsyncIndex(host=INDEX_HOST, api_key="test-key") as idx:
            assert isinstance(idx, AsyncIndex)

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        idx = AsyncIndex(host=INDEX_HOST, api_key="test-key")
        await idx.close()  # Should not raise


# ---------------------------------------------------------------------------
# AsyncPinecone.index() factory — synchronous (not a coroutine)
# ---------------------------------------------------------------------------


class TestFactoryWithHost:
    """Test AsyncPinecone.index(host=...) — direct construction, no describe call."""

    def test_creates_index_with_host(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        idx = pc.index(host=INDEX_HOST)
        assert isinstance(idx, AsyncIndex)
        assert idx.host == INDEX_HOST_HTTPS

    def test_no_describe_call_needed(self) -> None:
        """When host is given, no HTTP call is needed — method is synchronous."""
        pc = AsyncPinecone(api_key="test-key")
        idx = pc.index(host=INDEX_HOST)
        assert isinstance(idx, AsyncIndex)

    def test_forwards_config(self) -> None:
        pc = AsyncPinecone(
            api_key="forwarded-key",
            timeout=60.0,
            additional_headers={"X-Custom": "val"},
        )
        idx = pc.index(host=INDEX_HOST)
        assert idx._config.api_key == "forwarded-key"
        assert idx._config.timeout == 60.0
        assert idx._config.additional_headers == {"X-Custom": "val"}


class TestFactoryWithName:
    """Test AsyncPinecone.index(name=...) — uses cached host."""

    def test_uncached_name_raises(self) -> None:
        """When name is given but host is not cached, raise a helpful error."""
        pc = AsyncPinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="not cached"):
            pc.index(name="my-index")

    def test_cached_name_returns_index(self) -> None:
        """When name host is pre-populated in cache, return AsyncIndex directly."""
        pc = AsyncPinecone(api_key="test-key")
        pc._host_cache["my-index"] = INDEX_HOST
        idx = pc.index(name="my-index")
        assert isinstance(idx, AsyncIndex)
        assert idx.host == INDEX_HOST_HTTPS

    @pytest.mark.asyncio
    @respx.mock
    async def test_describe_then_index_by_name(self) -> None:
        """After calling describe(), index(name=...) should use the cached host."""
        respx.get(f"{CONTROL_PLANE_URL}/indexes/my-index").mock(
            return_value=httpx.Response(200, json=make_index_response(name="my-index")),
        )
        pc = AsyncPinecone(api_key="test-key")
        desc = await pc.indexes.describe("my-index")
        # Populate cache manually (describe doesn't auto-populate the client cache)
        pc._host_cache["my-index"] = desc.host

        idx = pc.index(name="my-index")
        assert isinstance(idx, AsyncIndex)
        assert INDEX_HOST in idx.host

    def test_different_cached_names(self) -> None:
        """Different cached names return different indexes."""
        pc = AsyncPinecone(api_key="test-key")
        pc._host_cache["index-a"] = "index-a-host.svc.pinecone.io"
        pc._host_cache["index-b"] = "index-b-host.svc.pinecone.io"

        idx_a = pc.index(name="index-a")
        idx_b = pc.index(name="index-b")

        assert "index-a-host" in idx_a.host
        assert "index-b-host" in idx_b.host


class TestFactoryValidation:
    """Test AsyncPinecone.index() validation."""

    def test_no_args_raises(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host must be provided"):
            pc.index()

    def test_empty_strings_raises(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host must be provided"):
            pc.index(name="", host="")

    def test_is_not_coroutine(self) -> None:
        """index() is a synchronous method, not a coroutine (unified-index-0030)."""
        import inspect

        pc = AsyncPinecone(api_key="test-key")
        assert not inspect.iscoroutinefunction(pc.index)
