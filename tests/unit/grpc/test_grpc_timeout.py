"""Unit tests for per-call timeout on GrpcIndex data-plane methods.

Verifies that:
1. Each method accepts a `timeout` kwarg and forwards it as `timeout_s` to the channel.
2. When `timeout=None`, `timeout_s=None` is passed (channel default applies).
3. DEADLINE_EXCEEDED in the exception message raises PineconeTimeoutError.
4. PineconeTimeoutError is catchable as TimeoutError (built-in).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import PineconeTimeoutError
from pinecone.grpc import GrpcIndex

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host="test-index-abc123.svc.pinecone.io",
            api_key="test-api-key",
        )


@pytest.fixture
def mock_channel() -> MagicMock:
    ch = MagicMock()
    ch.query.return_value = {"matches": [], "namespace": ""}
    ch.fetch.return_value = {"vectors": {}, "namespace": ""}
    ch.upsert.return_value = {"upserted_count": 1}
    ch.delete.return_value = {}
    ch.update.return_value = {}
    ch.list.return_value = {"vectors": [], "namespace": ""}
    return ch


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


class TestTimeoutForwarding:
    """Each method forwards timeout kwarg as timeout_s to the underlying channel."""

    def test_query_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.query(top_k=5, vector=[0.1, 0.2], timeout=2.5)
        _, kwargs = mock_channel.query.call_args
        assert kwargs.get("timeout_s") == 2.5

    def test_query_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.query(top_k=5, vector=[0.1, 0.2])
        _, kwargs = mock_channel.query.call_args
        assert kwargs.get("timeout_s") is None

    def test_fetch_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.fetch(ids=["v1"], timeout=3.0)
        _, kwargs = mock_channel.fetch.call_args
        assert kwargs.get("timeout_s") == 3.0

    def test_fetch_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.fetch(ids=["v1"])
        _, kwargs = mock_channel.fetch.call_args
        assert kwargs.get("timeout_s") is None

    def test_upsert_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}], timeout=5.0)
        _args, kwargs = mock_channel.upsert.call_args
        assert kwargs.get("timeout_s") == 5.0

    def test_upsert_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}])
        _args, kwargs = mock_channel.upsert.call_args
        assert kwargs.get("timeout_s") is None

    def test_delete_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.delete(ids=["v1"], timeout=1.0)
        _, kwargs = mock_channel.delete.call_args
        assert kwargs.get("timeout_s") == 1.0

    def test_delete_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.delete(ids=["v1"])
        _, kwargs = mock_channel.delete.call_args
        assert kwargs.get("timeout_s") is None

    def test_update_forwards_timeout(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        grpc_index.update(id="v1", values=[0.1, 0.2], timeout=4.0)
        _, kwargs = mock_channel.update.call_args
        assert kwargs.get("timeout_s") == 4.0

    def test_update_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.update(id="v1", values=[0.1, 0.2])
        _, kwargs = mock_channel.update.call_args
        assert kwargs.get("timeout_s") is None

    def test_list_paginated_forwards_timeout(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.list_paginated(timeout=2.0)
        _, kwargs = mock_channel.list.call_args
        assert kwargs.get("timeout_s") == 2.0

    def test_list_paginated_none_timeout_passes_none(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        grpc_index.list_paginated()
        _, kwargs = mock_channel.list.call_args
        assert kwargs.get("timeout_s") is None


class TestDeadlineExceededRaisesPineconeTimeoutError:
    """PineconeTimeoutError from Rust propagates unchanged through GrpcIndex methods.

    The Rust transport raises PineconeTimeoutError directly when a gRPC
    DEADLINE_EXCEEDED status is received. The Python layer no longer
    does any exception mapping — it just calls self._channel.method(...)
    and lets the typed exception propagate.
    """

    def _deadline_channel(self) -> MagicMock:
        ch = MagicMock()
        exc = PineconeTimeoutError("deadline exceeded after 20s")
        ch.query.side_effect = exc
        ch.fetch.side_effect = exc
        ch.upsert.side_effect = exc
        ch.delete.side_effect = exc
        ch.update.side_effect = exc
        ch.list.side_effect = exc
        return ch

    def test_query_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.query(top_k=5, vector=[0.1, 0.2], timeout=0.001)

    def test_fetch_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.fetch(ids=["v1"], timeout=0.001)

    def test_upsert_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.upsert(vectors=[{"id": "v1", "values": [0.1, 0.2]}], timeout=0.001)

    def test_delete_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.delete(ids=["v1"], timeout=0.001)

    def test_update_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.update(id="v1", values=[0.1, 0.2], timeout=0.001)

    def test_list_paginated_deadline_raises_timeout_error(self) -> None:
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(PineconeTimeoutError):
            idx.list_paginated(timeout=0.001)

    def test_pinecone_timeout_error_is_catchable_as_builtin_timeout_error(self) -> None:
        """PineconeTimeoutError inherits from TimeoutError for broad exception handlers."""
        idx = _make_grpc_index(self._deadline_channel())
        with pytest.raises(TimeoutError):
            idx.query(top_k=5, vector=[0.1, 0.2], timeout=0.001)
