"""Unit tests for GrpcIndex.query() sparse-only validation parity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.errors.exceptions import ValidationError
from pinecone.grpc import GrpcIndex

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index() -> tuple[GrpcIndex, MagicMock]:
    mock_channel = MagicMock()
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        idx = GrpcIndex(
            host="test-index-abc123.svc.pinecone.io",
            api_key="test-api-key",
        )
    return idx, mock_channel


def test_grpc_query_sparse_only_accepted() -> None:
    """Sparse-only query (no vector, no id) must be accepted and forwarded to channel."""
    idx, mock_channel = _make_grpc_index()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}

    idx.query(top_k=5, sparse_vector={"indices": [1], "values": [0.5]})

    mock_channel.query.assert_called_once()
    call_kwargs = mock_channel.query.call_args[1]
    assert call_kwargs["sparse_vector"] == {"indices": [1], "values": [0.5]}
    assert call_kwargs["vector"] is None
    assert call_kwargs["id"] is None


def test_grpc_query_vector_id_sparse_none_rejected() -> None:
    """Calling query() with no vector, no id, and no sparse_vector raises ValidationError."""
    idx, _ = _make_grpc_index()
    with pytest.raises(
        ValidationError,
        match="At least one of vector, id, or sparse_vector must be provided",
    ):
        idx.query(top_k=5)
