"""Unit tests for GrpcIndex.upsert_from_dataframe()."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from pinecone.grpc import GrpcIndex
from pinecone.models.vectors.responses import UpsertResponse

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    """Helper to create a GrpcIndex with a mocked GrpcChannel."""
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
            host="test-index-abc123.svc.pinecone.io",
            api_key="test-api-key",
        )


def _upsert_response(upserted_count: int) -> UpsertResponse:
    return UpsertResponse(upserted_count=upserted_count)


@pytest.fixture()
def mock_channel() -> MagicMock:
    ch = MagicMock()
    ch.upsert.return_value = {"upserted_count": 500}
    return ch


@pytest.fixture()
def grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    return _make_grpc_index(mock_channel)


class TestGrpcDataframeUpsert:
    """Tests for GrpcIndex.upsert_from_dataframe()."""

    def test_dataframe_1200_rows_creates_3_batches(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """1200 rows at batch_size=500 should produce 3 async calls: 500+500+200."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(1200)],
                "values": [[float(i)] for i in range(1200)],
            }
        )
        mock_channel.upsert.side_effect = [
            {"upserted_count": 500},
            {"upserted_count": 500},
            {"upserted_count": 200},
        ]

        result = grpc_index.upsert_from_dataframe(df, show_progress=False)

        assert mock_channel.upsert.call_count == 3
        batch_sizes = [len(call[0][0]) for call in mock_channel.upsert.call_args_list]
        assert batch_sizes == [500, 500, 200]
        assert result.upserted_count == 1200

    def test_default_batch_size_is_500(self) -> None:
        """The default batch_size parameter should be 500."""
        sig = inspect.signature(GrpcIndex.upsert_from_dataframe)
        assert sig.parameters["batch_size"].default == 500

    def test_custom_batch_size(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        """A custom batch_size should control how many rows per async call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(10)],
                "values": [[float(i)] for i in range(10)],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 3}

        grpc_index.upsert_from_dataframe(df, batch_size=3, show_progress=False)

        assert mock_channel.upsert.call_count == 4
        batch_sizes = [len(call[0][0]) for call in mock_channel.upsert.call_args_list]
        assert batch_sizes == [3, 3, 3, 1]

    def test_results_aggregated(self, grpc_index: GrpcIndex, mock_channel: MagicMock) -> None:
        """upserted_count should be summed across all batches."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": [f"v{i}" for i in range(100)],
                "values": [[float(i)] for i in range(100)],
            }
        )
        mock_channel.upsert.side_effect = [
            {"upserted_count": 50},
            {"upserted_count": 50},
        ]

        result = grpc_index.upsert_from_dataframe(df, batch_size=50, show_progress=False)

        assert result.upserted_count == 100

    def test_namespace_passed_to_each_batch(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """The namespace argument should be forwarded to every upsert call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1", "v2"],
                "values": [[0.1, 0.2], [0.3, 0.4]],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 2}

        grpc_index.upsert_from_dataframe(df, namespace="my-ns", show_progress=False)

        for call in mock_channel.upsert.call_args_list:
            assert call[0][1] == "my-ns"

    def test_single_batch_when_rows_fit(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """DataFrame with fewer rows than batch_size should produce one call."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {
                "id": ["v1", "v2", "v3"],
                "values": [[0.1], [0.2], [0.3]],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 3}

        result = grpc_index.upsert_from_dataframe(df, show_progress=False)

        assert mock_channel.upsert.call_count == 1
        assert result.upserted_count == 3

    def test_invalid_df_raises(self, grpc_index: GrpcIndex) -> None:
        """Non-DataFrame input should raise PineconeValueError."""
        pytest.importorskip("pandas")
        with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
            grpc_index.upsert_from_dataframe([1, 2, 3])  # type: ignore[arg-type]

    def test_batch_size_zero_raises(self, grpc_index: GrpcIndex) -> None:
        """batch_size=0 should raise PineconeValueError."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert_from_dataframe(df, batch_size=0)

    def test_batch_size_negative_raises(self, grpc_index: GrpcIndex) -> None:
        """Negative batch_size should raise PineconeValueError."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"id": ["v1"], "values": [[0.1]]})
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            grpc_index.upsert_from_dataframe(df, batch_size=-1)

    def test_metadata_and_sparse_values_forwarded(
        self, grpc_index: GrpcIndex, mock_channel: MagicMock
    ) -> None:
        """metadata and sparse_values columns should be forwarded to upsert."""
        pd = pytest.importorskip("pandas")
        sparse = {"indices": [0, 2], "values": [0.5, 0.8]}
        df = pd.DataFrame(
            {
                "id": ["v1"],
                "values": [[0.1, 0.2]],
                "sparse_values": [sparse],
                "metadata": [{"genre": "rock"}],
            }
        )
        mock_channel.upsert.return_value = {"upserted_count": 1}

        grpc_index.upsert_from_dataframe(df, show_progress=False)

        vec = mock_channel.upsert.call_args[0][0][0]
        assert vec["sparse_values"] == sparse
        assert vec["metadata"] == {"genre": "rock"}
