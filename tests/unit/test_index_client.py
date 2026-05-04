"""Unit tests for Index data plane client and Pinecone.index() factory."""

from __future__ import annotations

import sys
import warnings
from multiprocessing.pool import ApplyResult

import httpx
import pytest
import respx

from pinecone import Index, Pinecone
from pinecone.errors.exceptions import PineconeValueError, ValidationError
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

    def test_pool_threads_installs_legacy_async_pool(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert hasattr(idx, "_legacy_async_pool")

    def test_pool_threads_kwarg_does_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            Index(host=INDEX_HOST, api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert len(record) == 0

    def test_unknown_kwarg_still_rejected(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            Index(host=INDEX_HOST, api_key="test-key", bogus=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# async_req=True opt-in via pool_threads= (BC-0109)
# ---------------------------------------------------------------------------

UPSERT_URL = f"{INDEX_HOST_HTTPS}/vectors/upsert"


class TestAsyncReqOptIn:
    def test_index_without_pool_threads_does_not_import_legacy_async(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delitem(sys.modules, "pinecone._legacy.async_req", raising=False)
        monkeypatch.delitem(sys.modules, "multiprocessing.pool", raising=False)
        Index(host=INDEX_HOST, api_key="test-key")
        assert "pinecone._legacy.async_req" not in sys.modules
        assert "multiprocessing.pool" not in sys.modules

    def test_index_with_pool_threads_imports_legacy_module_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delitem(sys.modules, "multiprocessing.pool", raising=False)
        Index(host=INDEX_HOST, api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert "pinecone._legacy.async_req" in sys.modules
        # multiprocessing.pool is NOT yet imported — pool is lazy, constructed on first submit
        assert "multiprocessing.pool" not in sys.modules

    def test_async_req_true_without_pool_threads_raises_typeerror(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key")
        with pytest.raises(TypeError, match="async_req"):
            idx.upsert(vectors=[("a", [0.1])], async_req=True)  # type: ignore[call-arg]

    @respx.mock
    def test_async_req_true_with_pool_threads_returns_apply_result(self) -> None:
        respx.post(UPSERT_URL).mock(return_value=httpx.Response(200, json={"upsertedCount": 1}))
        idx = Index(host=INDEX_HOST, api_key="test-key", pool_threads=2)  # type: ignore[call-arg]
        result = idx.upsert(vectors=[("a", [0.1, 0.2])], async_req=True)  # type: ignore[call-arg]
        assert isinstance(result, ApplyResult)
        upsert_response = result.get(timeout=5)
        assert upsert_response.upserted_count == 1

    def test_async_req_true_with_batch_size_raises_legacy_text(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key", pool_threads=2)  # type: ignore[call-arg]
        with pytest.raises(PineconeValueError) as excinfo:
            idx.upsert(vectors=[("a", [0.1])], batch_size=10, async_req=True)  # type: ignore[call-arg]
        assert str(excinfo.value) == "async_req is not supported when batch_size is provided."

    @respx.mock
    def test_close_shuts_down_legacy_pool_idempotently(self) -> None:
        respx.post(UPSERT_URL).mock(return_value=httpx.Response(200, json={"upsertedCount": 1}))
        idx = Index(host=INDEX_HOST, api_key="test-key", pool_threads=2)  # type: ignore[call-arg]
        # Trigger pool construction by submitting a real call
        result = idx.upsert(vectors=[("a", [0.1, 0.2])], async_req=True)  # type: ignore[call-arg]
        result.get(timeout=5)
        # Close — should not raise
        idx.close()
        # Second close — should also not raise
        idx.close()

    def test_close_works_when_legacy_pool_never_installed(self) -> None:
        idx = Index(host=INDEX_HOST, api_key="test-key")
        idx.close()  # must not raise; _legacy_async_pool attribute was never set

    def test_pool_threads_emits_no_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            Index(host=INDEX_HOST, api_key="test-key", pool_threads=4)  # type: ignore[call-arg]
        assert [w for w in record if issubclass(w.category, DeprecationWarning)] == []
