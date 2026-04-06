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
# AsyncPinecone.index() factory
# ---------------------------------------------------------------------------


class TestFactoryWithHost:
    """Test AsyncPinecone.index(host=...) — direct construction, no describe call."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_creates_index_with_host(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        idx = await pc.index(host=INDEX_HOST)
        assert isinstance(idx, AsyncIndex)
        assert idx.host == INDEX_HOST_HTTPS

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_describe_call_when_host_provided(self) -> None:
        """When host is given, no HTTP call should be made."""
        route = respx.get(f"{CONTROL_PLANE_URL}/indexes/anything")
        pc = AsyncPinecone(api_key="test-key")
        await pc.index(host=INDEX_HOST)
        assert not route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_forwards_config(self) -> None:
        pc = AsyncPinecone(
            api_key="forwarded-key",
            timeout=60.0,
            additional_headers={"X-Custom": "val"},
        )
        idx = await pc.index(host=INDEX_HOST)
        assert idx._config.api_key == "forwarded-key"
        assert idx._config.timeout == 60.0
        assert idx._config.additional_headers == {"X-Custom": "val"}


class TestFactoryWithName:
    """Test AsyncPinecone.index(name=...) — triggers describe to resolve host."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_resolves_host_via_describe(self) -> None:
        respx.get(f"{CONTROL_PLANE_URL}/indexes/my-index").mock(
            return_value=httpx.Response(200, json=make_index_response(name="my-index")),
        )
        pc = AsyncPinecone(api_key="test-key")
        idx = await pc.index(name="my-index")
        assert isinstance(idx, AsyncIndex)
        assert INDEX_HOST in idx.host

    @pytest.mark.asyncio
    @respx.mock
    async def test_host_cached_after_describe(self) -> None:
        """Second call with same name should use cache, not call describe again."""
        route = respx.get(f"{CONTROL_PLANE_URL}/indexes/my-index").mock(
            return_value=httpx.Response(200, json=make_index_response(name="my-index")),
        )
        pc = AsyncPinecone(api_key="test-key")

        idx1 = await pc.index(name="my-index")
        idx2 = await pc.index(name="my-index")

        assert route.call_count == 1
        assert idx1.host == idx2.host

    @pytest.mark.asyncio
    @respx.mock
    async def test_different_names_each_call_describe(self) -> None:
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
        pc = AsyncPinecone(api_key="test-key")

        idx_a = await pc.index(name="index-a")
        idx_b = await pc.index(name="index-b")

        assert "index-a-host" in idx_a.host
        assert "index-b-host" in idx_b.host


class TestFactoryValidation:
    """Test AsyncPinecone.index() validation."""

    @pytest.mark.asyncio
    async def test_no_args_raises(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host must be provided"):
            await pc.index()

    @pytest.mark.asyncio
    async def test_empty_strings_raises(self) -> None:
        pc = AsyncPinecone(api_key="test-key")
        with pytest.raises(ValidationError, match="Either name or host must be provided"):
            await pc.index(name="", host="")
