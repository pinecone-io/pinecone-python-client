"""Unit tests verifying per-call timeout parameter parity across Index, AsyncIndex, GrpcIndex.

Resolution: Option 1 — per-call `timeout` added to REST Index/AsyncIndex methods to match
GrpcIndex. All three clients now accept `timeout: float | None = None` on every data-plane
method, enabling callers to override the constructor-level default on a per-request basis.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone import AsyncIndex, Index
from pinecone.grpc import GrpcIndex

INDEX_HOST = "test-index-abc1234.svc.us-east1-gcp.pinecone.io"
INDEX_HOST_HTTPS = f"https://{INDEX_HOST}"
_DEFAULT_TIMEOUT = 30.0  # matches PineconeConfig.timeout default


def _request_timeout(route: respx.Route) -> float | None:
    """Return the read-timeout (seconds) recorded on the last httpx request for a route.

    httpx stores timeout as a dict in request.extensions["timeout"] with keys
    connect/read/write/pool. We check 'read' as the most meaningful for data-plane calls.
    """
    ext: dict[str, float] | None = route.calls.last.request.extensions.get("timeout")
    if ext is None:
        return None
    return ext.get("read")


_DATA_PLANE_METHODS = [
    "upsert",
    "query",
    "fetch",
    "delete",
    "update",
    "describe_index_stats",
    "upsert_records",
    "search",
    "search_records",
    "list",
]

# Methods present on Index/AsyncIndex but not GrpcIndex — checked separately below
_HTTP_ONLY_DATA_PLANE_METHODS = [
    "query_namespaces",
    "fetch_by_metadata",
    "delete_namespace",
]

_QUERY_RESPONSE = {
    "matches": [{"id": "v1", "score": 0.9}],
    "namespace": "",
    "usage": {"readUnits": 1},
}


def _make_index() -> Index:
    return Index(host=INDEX_HOST, api_key="test-key")


def _make_async_index() -> AsyncIndex:
    return AsyncIndex(host=INDEX_HOST, api_key="test-key")


def _make_grpc_index() -> GrpcIndex:
    mock_channel = MagicMock()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}
    mock_channel.upsert.return_value = {"upserted_count": 0}
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {"pinecone._grpc": mock_module}):
        return GrpcIndex(host=INDEX_HOST, api_key="test-key")


# ---------------------------------------------------------------------------
# Sync Index — per-call timeout forwarded to httpx
# ---------------------------------------------------------------------------


class TestIndexQueryAcceptsPerCallTimeout:
    """Index.query passes per-call timeout through to the httpx client."""

    @respx.mock
    def test_index_query_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(200, json=_QUERY_RESPONSE)
        )
        idx = _make_index()
        idx.query(top_k=1, vector=[0.1, 0.2, 0.3], timeout=5.0)

        assert route.called
        assert _request_timeout(route) == 5.0

    @respx.mock
    def test_index_query_default_timeout_when_none(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(200, json=_QUERY_RESPONSE)
        )
        idx = _make_index()
        idx.query(top_k=1, vector=[0.1, 0.2, 0.3])  # no timeout arg — uses default
        assert route.called
        assert _request_timeout(route) == _DEFAULT_TIMEOUT

    @respx.mock
    def test_index_upsert_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/upsert").mock(
            return_value=httpx.Response(200, json={"upsertedCount": 1})
        )
        idx = _make_index()
        idx.upsert(vectors=[("v1", [0.1, 0.2])], timeout=10.0)
        assert route.called
        assert _request_timeout(route) == 10.0

    @respx.mock
    def test_index_fetch_accepts_per_call_timeout(self) -> None:
        route = respx.get(f"{INDEX_HOST_HTTPS}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}, "namespace": ""})
        )
        idx = _make_index()
        idx.fetch(ids=["v1"], timeout=3.0)
        assert route.called
        assert _request_timeout(route) == 3.0

    @respx.mock
    def test_index_delete_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/delete").mock(
            return_value=httpx.Response(200, json={})
        )
        idx = _make_index()
        idx.delete(ids=["v1"], timeout=2.0)
        assert route.called
        assert _request_timeout(route) == 2.0

    @respx.mock
    def test_index_update_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/update").mock(
            return_value=httpx.Response(200, json={})
        )
        idx = _make_index()
        idx.update(id="v1", values=[0.1, 0.2], timeout=4.0)
        assert route.called
        assert _request_timeout(route) == 4.0

    @respx.mock
    def test_index_describe_index_stats_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/describe_index_stats").mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": {},
                    "dimension": 3,
                    "totalVectorCount": 0,
                    "indexFullness": 0.0,
                },
            )
        )
        idx = _make_index()
        idx.describe_index_stats(timeout=6.0)
        assert route.called
        assert _request_timeout(route) == 6.0

    @respx.mock
    def test_index_list_paginated_accepts_per_call_timeout(self) -> None:
        route = respx.get(f"{INDEX_HOST_HTTPS}/vectors/list").mock(
            return_value=httpx.Response(
                200, json={"vectors": [], "namespace": "", "usage": {"readUnits": 1}}
            )
        )
        idx = _make_index()
        idx.list_paginated(timeout=7.0)
        assert route.called
        assert _request_timeout(route) == 7.0

    @respx.mock
    def test_index_query_namespaces_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(200, json=_QUERY_RESPONSE)
        )
        idx = _make_index()
        idx.query_namespaces(
            vector=[0.1, 0.2, 0.3],
            namespaces=["ns1"],
            metric="cosine",
            timeout=5.0,
        )
        assert route.called
        assert _request_timeout(route) == 5.0

    @respx.mock
    def test_index_fetch_by_metadata_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/fetch_by_metadata").mock(
            return_value=httpx.Response(
                200, json={"vectors": {}, "namespace": "", "usage": {"readUnits": 1}}
            )
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}}, timeout=3.0)
        assert route.called
        assert _request_timeout(route) == 3.0

    @respx.mock
    def test_index_delete_namespace_accepts_per_call_timeout(self) -> None:
        route = respx.delete(f"{INDEX_HOST_HTTPS}/namespaces/old-ns").mock(
            return_value=httpx.Response(200, json={})
        )
        idx = _make_index()
        idx.delete_namespace(name="old-ns", timeout=2.0)
        assert route.called
        assert _request_timeout(route) == 2.0

    @respx.mock
    def test_index_query_namespaces_default_timeout_when_none(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(200, json=_QUERY_RESPONSE)
        )
        idx = _make_index()
        idx.query_namespaces(
            vector=[0.1, 0.2, 0.3],
            namespaces=["ns1"],
            metric="cosine",
        )
        assert route.called
        assert _request_timeout(route) == _DEFAULT_TIMEOUT

    @respx.mock
    def test_index_fetch_by_metadata_default_timeout_when_none(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/fetch_by_metadata").mock(
            return_value=httpx.Response(
                200, json={"vectors": {}, "namespace": "", "usage": {"readUnits": 1}}
            )
        )
        idx = _make_index()
        idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})
        assert route.called
        assert _request_timeout(route) == _DEFAULT_TIMEOUT

    @respx.mock
    def test_index_delete_namespace_default_timeout_when_none(self) -> None:
        route = respx.delete(f"{INDEX_HOST_HTTPS}/namespaces/old-ns").mock(
            return_value=httpx.Response(200, json={})
        )
        idx = _make_index()
        idx.delete_namespace(name="old-ns")
        assert route.called
        assert _request_timeout(route) == _DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Async Index — per-call timeout forwarded
# ---------------------------------------------------------------------------


class TestAsyncIndexQueryAcceptsPerCallTimeout:
    """AsyncIndex.query passes per-call timeout through to the async httpx client."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_query_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(200, json=_QUERY_RESPONSE)
        )
        idx = _make_async_index()
        await idx.query(top_k=1, vector=[0.1, 0.2, 0.3], timeout=5.0)
        assert route.called
        assert _request_timeout(route) == 5.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_upsert_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/upsert").mock(
            return_value=httpx.Response(200, json={"upsertedCount": 1})
        )
        idx = _make_async_index()
        await idx.upsert(vectors=[("v1", [0.1, 0.2])], timeout=10.0)
        assert route.called
        assert _request_timeout(route) == 10.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_fetch_accepts_per_call_timeout(self) -> None:
        route = respx.get(f"{INDEX_HOST_HTTPS}/vectors/fetch").mock(
            return_value=httpx.Response(200, json={"vectors": {}, "namespace": ""})
        )
        idx = _make_async_index()
        await idx.fetch(ids=["v1"], timeout=3.0)
        assert route.called
        assert _request_timeout(route) == 3.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_describe_index_stats_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/describe_index_stats").mock(
            return_value=httpx.Response(
                200,
                json={
                    "namespaces": {},
                    "dimension": 3,
                    "totalVectorCount": 0,
                    "indexFullness": 0.0,
                },
            )
        )
        idx = _make_async_index()
        await idx.describe_index_stats(timeout=6.0)
        assert route.called
        assert _request_timeout(route) == 6.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_query_namespaces_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(200, json=_QUERY_RESPONSE)
        )
        idx = _make_async_index()
        await idx.query_namespaces(
            vector=[0.1, 0.2, 0.3],
            namespaces=["ns1"],
            metric="cosine",
            timeout=5.0,
        )
        assert route.called
        assert _request_timeout(route) == 5.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_fetch_by_metadata_accepts_per_call_timeout(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/fetch_by_metadata").mock(
            return_value=httpx.Response(
                200, json={"vectors": {}, "namespace": "", "usage": {"readUnits": 1}}
            )
        )
        idx = _make_async_index()
        await idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}}, timeout=3.0)
        assert route.called
        assert _request_timeout(route) == 3.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_delete_namespace_accepts_per_call_timeout(self) -> None:
        route = respx.delete(f"{INDEX_HOST_HTTPS}/namespaces/old-ns").mock(
            return_value=httpx.Response(200, json={})
        )
        idx = _make_async_index()
        await idx.delete_namespace(name="old-ns", timeout=2.0)
        assert route.called
        assert _request_timeout(route) == 2.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_query_namespaces_default_timeout_when_none(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/query").mock(
            return_value=httpx.Response(200, json=_QUERY_RESPONSE)
        )
        idx = _make_async_index()
        await idx.query_namespaces(
            vector=[0.1, 0.2, 0.3],
            namespaces=["ns1"],
            metric="cosine",
        )
        assert route.called
        assert _request_timeout(route) == _DEFAULT_TIMEOUT

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_fetch_by_metadata_default_timeout_when_none(self) -> None:
        route = respx.post(f"{INDEX_HOST_HTTPS}/vectors/fetch_by_metadata").mock(
            return_value=httpx.Response(
                200, json={"vectors": {}, "namespace": "", "usage": {"readUnits": 1}}
            )
        )
        idx = _make_async_index()
        await idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})
        assert route.called
        assert _request_timeout(route) == _DEFAULT_TIMEOUT

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_index_delete_namespace_default_timeout_when_none(self) -> None:
        route = respx.delete(f"{INDEX_HOST_HTTPS}/namespaces/old-ns").mock(
            return_value=httpx.Response(200, json={})
        )
        idx = _make_async_index()
        await idx.delete_namespace(name="old-ns")
        assert route.called
        assert _request_timeout(route) == _DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# GrpcIndex — existing timeout behavior unchanged
# ---------------------------------------------------------------------------


class TestGrpcIndexQueryAcceptsPerCallTimeout:
    """GrpcIndex.query still accepts and forwards timeout (unchanged behavior)."""

    def test_grpc_index_query_accepts_per_call_timeout(self) -> None:
        mock_channel = MagicMock()
        mock_channel.query.return_value = {"matches": [], "namespace": ""}
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {"pinecone._grpc": mock_module}):
            idx = GrpcIndex(host=INDEX_HOST, api_key="test-key")
            idx.query(top_k=1, vector=[0.1, 0.2, 0.3], timeout=5.0)
        mock_channel.query.assert_called_once()
        call_kwargs = mock_channel.query.call_args.kwargs
        assert call_kwargs.get("timeout_s") == 5.0

    def test_grpc_index_query_none_timeout(self) -> None:
        mock_channel = MagicMock()
        mock_channel.query.return_value = {"matches": [], "namespace": ""}
        mock_module = MagicMock()
        mock_module.GrpcChannel.return_value = mock_channel
        with patch.dict("sys.modules", {"pinecone._grpc": mock_module}):
            idx = GrpcIndex(host=INDEX_HOST, api_key="test-key")
            idx.query(top_k=1, vector=[0.1, 0.2, 0.3])
        call_kwargs = mock_channel.query.call_args.kwargs
        assert call_kwargs.get("timeout_s") is None


# ---------------------------------------------------------------------------
# Parity check — all data-plane methods on all three clients have `timeout`
# ---------------------------------------------------------------------------


class TestAllDataPlaneMethodsAcceptTimeoutParity:
    """Every data-plane method on Index, AsyncIndex, and GrpcIndex accepts a timeout param."""

    def _check_methods(self, cls: type, methods: list[str]) -> None:
        for name in methods:
            method = getattr(cls, name, None)
            assert method is not None, f"{cls.__name__} is missing method {name!r}"
            sig = inspect.signature(method)
            assert "timeout" in sig.parameters, (
                f"{cls.__name__}.{name}() is missing a `timeout` parameter"
            )
            param = sig.parameters["timeout"]
            assert param.default is None, (
                f"{cls.__name__}.{name}(timeout) default should be None, got {param.default!r}"
            )

    def test_index_methods_have_timeout(self) -> None:
        self._check_methods(Index, _DATA_PLANE_METHODS)

    def test_async_index_methods_have_timeout(self) -> None:
        self._check_methods(AsyncIndex, _DATA_PLANE_METHODS)

    def test_grpc_index_methods_have_timeout(self) -> None:
        self._check_methods(GrpcIndex, _DATA_PLANE_METHODS)

    def test_index_http_only_methods_have_timeout(self) -> None:
        self._check_methods(Index, _HTTP_ONLY_DATA_PLANE_METHODS)

    def test_async_index_http_only_methods_have_timeout(self) -> None:
        self._check_methods(AsyncIndex, _HTTP_ONLY_DATA_PLANE_METHODS)
