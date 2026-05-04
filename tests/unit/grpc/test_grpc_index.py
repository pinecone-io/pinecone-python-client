"""Unit tests for the GRPCIndex capitalisation alias (BCG-141)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pinecone.grpc
from pinecone.grpc import GRPCIndex, GrpcIndex

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index_via_alias(mock_channel: MagicMock) -> GRPCIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GRPCIndex(
            host="https://x-abc.svc.pinecone.io",
            api_key="k",
        )


def test_legacy_GRPCIndex_alias_imports() -> None:  # noqa: N802
    from pinecone.grpc import GRPCIndex as _GRPCIndex  # noqa: F401


def test_legacy_GRPCIndex_alias_is_canonical() -> None:  # noqa: N802
    assert pinecone.grpc.GRPCIndex is pinecone.grpc.GrpcIndex


def test_legacy_GRPCIndex_alias_constructs() -> None:  # noqa: N802
    mock_channel = MagicMock()
    mock_channel.query.return_value = {"matches": [], "namespace": ""}
    idx = _make_grpc_index_via_alias(mock_channel)
    assert isinstance(idx, GrpcIndex)
