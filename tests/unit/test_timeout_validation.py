"""Non-positive ``timeout`` and ``total_timeout`` raise ``PineconeValueError`` up front.

Both parameters are SDK-only (the server never sees them), so the client is the
only place that can reject a value with no meaning. Every case asserts that no
request was sent, and the sync and async surfaces are checked against the same
table so they cannot disagree.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from pinecone import AsyncPinecone, Pinecone
from pinecone.async_client.async_index import AsyncIndex
from pinecone.errors.exceptions import PineconeValueError
from pinecone.grpc import GrpcIndex
from pinecone.index import Index
from pinecone.models.documents.score_by import TextQuery

HOST = "documents-index-abc123.svc.us-east-1-aws.pinecone.io"
BASE_URL = f"https://{HOST}"
NS = "ns"
DOCS = [{"_id": "doc-1", "title": "Rome"}]
BAD_VALUES = [0, 0.0, -1, -5, -0.5, float("nan")]

_CALLS: dict[str, Callable[..., Any]] = {
    "upsert": lambda d, **kw: d.upsert(namespace=NS, documents=DOCS, **kw),
    "batch_upsert": lambda d, **kw: d.batch_upsert(
        namespace=NS, documents=DOCS, show_progress=False, **kw
    ),
    "search": lambda d, **kw: d.search(
        namespace=NS, score_by=[TextQuery(query="rome", fields=["title"])], top_k=3, **kw
    ),
    "fetch": lambda d, **kw: d.fetch(namespace=NS, ids=["doc-1"], **kw),
    "delete": lambda d, **kw: d.delete(namespace=NS, ids=["doc-1"], **kw),
    "update": lambda d, **kw: d.update(namespace=NS, documents=DOCS, **kw),
    "list": lambda d, **kw: d.list(namespace=NS, **kw),
}


@pytest.fixture
def index() -> Iterator[Index]:
    client = Index(host=HOST, api_key="test-key")
    yield client
    client.close()


@pytest.fixture
async def async_index() -> AsyncIterator[AsyncIndex]:
    client = AsyncIndex(host=HOST, api_key="test-key")
    yield client
    await client.close()


@pytest.mark.parametrize("method", sorted(_CALLS))
@pytest.mark.parametrize("value", BAD_VALUES)
class TestPerRequestTimeout:
    def test_sync_rejects_before_sending(self, index: Index, method: str, value: float) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(PineconeValueError, match="timeout"):
                _CALLS[method](index.documents, timeout=value)
        assert len(route.calls) == 0

    async def test_async_rejects_before_sending(
        self, async_index: AsyncIndex, method: str, value: float
    ) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(PineconeValueError, match="timeout"):
                await _CALLS[method](async_index.documents, timeout=value)
        assert len(route.calls) == 0


@pytest.mark.parametrize("value", [0, -1, -0.5, float("nan")])
class TestTotalTimeout:
    def test_sync_rejects_before_sending(self, index: Index, value: float) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(202, json={}))
            with pytest.raises(PineconeValueError, match="total_timeout"):
                index.documents.batch_upsert(
                    namespace=NS, documents=DOCS, total_timeout=value, show_progress=False
                )
        assert len(route.calls) == 0

    async def test_async_rejects_before_sending(
        self, async_index: AsyncIndex, value: float
    ) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(202, json={}))
            with pytest.raises(PineconeValueError, match="total_timeout"):
                await async_index.documents.batch_upsert(
                    namespace=NS, documents=DOCS, total_timeout=value, show_progress=False
                )
        assert len(route.calls) == 0


class TestValidValuesUnchanged:
    @respx.mock
    def test_sync_none_and_half_second_send(self, index: Index) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(202, json={"upserted_count": 1})
        )
        for total in (None, 0.5):
            result = index.documents.batch_upsert(
                namespace=NS, documents=DOCS, total_timeout=total, timeout=5, show_progress=False
            )
            assert not result.timed_out
        assert len(route.calls) == 2

    @respx.mock
    async def test_async_none_and_half_second_send(self, async_index: AsyncIndex) -> None:
        route = respx.post(f"{BASE_URL}/namespaces/{NS}/documents/upsert").mock(
            return_value=httpx.Response(202, json={"upserted_count": 1})
        )
        for total in (None, 0.5):
            result = await async_index.documents.batch_upsert(
                namespace=NS, documents=DOCS, total_timeout=total, timeout=5, show_progress=False
            )
            assert not result.timed_out
        assert len(route.calls) == 2


@pytest.mark.parametrize("value", [0, -5, -0.1, float("nan")])
class TestClientTimeout:
    def test_pinecone_rejects(self, value: float) -> None:
        with pytest.raises(PineconeValueError, match="timeout"):
            Pinecone(api_key="test-key", timeout=value)

    def test_async_pinecone_rejects(self, value: float) -> None:
        with pytest.raises(PineconeValueError, match="timeout"):
            AsyncPinecone(api_key="test-key", timeout=value)


def test_client_default_timeout_still_accepted() -> None:
    assert Pinecone(api_key="test-key").config.timeout == 30.0
    assert Pinecone(api_key="test-key", timeout=0.5).config.timeout == 0.5


_VECTORS = [("v1", [0.1, 0.2]), ("v2", [0.3, 0.4])]


def _vector_frame() -> Any:
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"id": ["v1", "v2"], "values": [[0.1, 0.2], [0.3, 0.4]]})


def _grpc_index(channel: MagicMock) -> GrpcIndex:
    module = MagicMock()
    module.GrpcChannel.return_value = channel
    with patch.dict("sys.modules", {"pinecone._grpc": module}):
        return GrpcIndex(host=HOST, api_key="test-key")


@pytest.mark.parametrize("value", [0, -1, float("nan")])
class TestVectorTotalTimeout:
    """The vector batch paths share the bulk engines with ``batch_upsert``, so
    they used to return an all-failed result for these values too."""

    def test_sync_upsert(self, index: Index, value: float) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(PineconeValueError, match="total_timeout"):
                index.upsert(
                    vectors=_VECTORS, batch_size=1, total_timeout=value, show_progress=False
                )
        assert len(route.calls) == 0

    def test_sync_upsert_from_dataframe(self, index: Index, value: float) -> None:
        df = _vector_frame()
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(PineconeValueError, match="total_timeout"):
                index.upsert_from_dataframe(
                    df, batch_size=1, total_timeout=value, show_progress=False
                )
        assert len(route.calls) == 0

    async def test_async_upsert(self, async_index: AsyncIndex, value: float) -> None:
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(PineconeValueError, match="total_timeout"):
                await async_index.upsert(
                    vectors=_VECTORS, batch_size=1, total_timeout=value, show_progress=False
                )
        assert len(route.calls) == 0

    async def test_async_upsert_from_dataframe(self, async_index: AsyncIndex, value: float) -> None:
        df = _vector_frame()
        with respx.mock(assert_all_called=False) as router:
            route = router.route().mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(PineconeValueError, match="total_timeout"):
                await async_index.upsert_from_dataframe(
                    df, batch_size=1, total_timeout=value, show_progress=False
                )
        assert len(route.calls) == 0

    def test_grpc_upsert(self, value: float) -> None:
        channel = MagicMock()
        grpc_index = _grpc_index(channel)
        with pytest.raises(PineconeValueError, match="total_timeout"):
            grpc_index.upsert(
                vectors=_VECTORS, batch_size=1, total_timeout=value, show_progress=False
            )
        channel.upsert.assert_not_called()

    def test_grpc_upsert_from_dataframe(self, value: float) -> None:
        df = _vector_frame()
        channel = MagicMock()
        grpc_index = _grpc_index(channel)
        with pytest.raises(PineconeValueError, match="total_timeout"):
            grpc_index.upsert_from_dataframe(
                df, batch_size=1, total_timeout=value, show_progress=False
            )
        channel.upsert.assert_not_called()


@pytest.mark.parametrize("value", [0, -5, float("nan")])
def test_grpc_index_rejects_timeout_before_opening_channel(value: float) -> None:
    module = MagicMock()
    with (
        patch.dict("sys.modules", {"pinecone._grpc": module}),
        patch("pinecone.grpc.ThreadPoolExecutor") as executor,
        pytest.raises(PineconeValueError, match="timeout"),
    ):
        GrpcIndex(host=HOST, api_key="test-key", timeout=value)
    module.GrpcChannel.assert_not_called()
    executor.assert_not_called()
