"""Unit tests for GrpcIndex.upsert() batch_size and show_progress parameters (BCG-090)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import PineconeValueError
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


def _make_vectors(n: int) -> list[tuple[str, list[float]]]:
    return [(f"v{i}", [float(i)]) for i in range(n)]


@pytest.fixture
def mock_channel() -> MagicMock:
    ch = MagicMock()
    ch.upsert.return_value = {"upserted_count": 1}
    return ch


@pytest.fixture
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


class TestGrpcUpsertBatching:
    def test_upsert_no_batch_size_calls_channel_once(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """With batch_size=None, all 100 vectors go in a single channel call."""
        mock_channel.upsert.return_value = {"upserted_count": 100}
        vectors = _make_vectors(100)
        result = grpc_index.upsert(vectors=vectors, batch_size=None)
        assert mock_channel.upsert.call_count == 1
        assert result.upserted_count == 100

    def test_upsert_with_batch_size_calls_channel_per_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """250 vectors at batch_size=100 should produce exactly 3 channel upsert calls."""
        mock_channel.upsert.return_value = {"upserted_count": 100}
        vectors = _make_vectors(250)
        grpc_index.upsert(vectors=vectors, batch_size=100, show_progress=False)
        assert mock_channel.upsert.call_count == 3

    def test_upsert_with_batch_size_aggregates_response(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """upserted_count should be the sum of all per-batch counts."""
        mock_channel.upsert.side_effect = [
            {"upserted_count": 100},
            {"upserted_count": 100},
            {"upserted_count": 50},
        ]
        vectors = _make_vectors(250)
        result = grpc_index.upsert(vectors=vectors, batch_size=100, show_progress=False)
        assert result.upserted_count == 250

    def test_upsert_invalid_batch_size_raises(self, grpc_index: GrpcIndex) -> None:
        """batch_size of 0, -1, or a float should raise PineconeValueError."""
        vectors = _make_vectors(5)
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert(vectors=vectors, batch_size=0)
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert(vectors=vectors, batch_size=-1)
        with pytest.raises(PineconeValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert(vectors=vectors, batch_size=1.5)  # type: ignore[arg-type]

    def test_upsert_show_progress_false_does_not_import_tqdm(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """When show_progress=False, tqdm must not be imported."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        with patch.dict("sys.modules", {"tqdm": None, "tqdm.auto": None}):
            result = grpc_index.upsert(vectors=vectors, batch_size=5, show_progress=False)
        assert result.upserted_count == 10

    def test_upsert_namespace_forwarded_per_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The namespace argument must appear in every per-batch channel call."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        grpc_index.upsert(vectors=vectors, namespace="my-ns", batch_size=5, show_progress=False)
        assert mock_channel.upsert.call_count == 2
        for call in mock_channel.upsert.call_args_list:
            assert call[0][1] == "my-ns"

    def test_upsert_timeout_forwarded_per_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """timeout=5.0 should be forwarded to each per-batch channel call."""
        mock_channel.upsert.return_value = {"upserted_count": 5}
        vectors = _make_vectors(10)
        grpc_index.upsert(vectors=vectors, batch_size=5, timeout=5.0, show_progress=False)
        assert mock_channel.upsert.call_count == 2
        for call in mock_channel.upsert.call_args_list:
            assert call[1].get("timeout_s") == 5.0

    def test_upsert_empty_vectors_with_batch_size(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """Empty vector list with batch_size set should make zero channel calls."""
        result = grpc_index.upsert(vectors=[], batch_size=100, show_progress=False)
        assert mock_channel.upsert.call_count == 0
        assert result.upserted_count == 0
