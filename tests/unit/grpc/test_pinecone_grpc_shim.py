"""Unit tests for the PineconeGRPC backwards-compatibility shim (BCG-140)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone._client import Pinecone
from pinecone.errors.exceptions import PineconeValueError
from pinecone.grpc import GrpcIndex, PineconeGRPC

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_pc_grpc() -> tuple[PineconeGRPC, MagicMock]:
    """Create a PineconeGRPC instance with a mocked Rust gRPC extension."""
    mock_channel = MagicMock()
    mock_grpc_module = MagicMock()
    mock_grpc_module.GrpcChannel.return_value = mock_channel
    return PineconeGRPC(api_key="test-key"), mock_channel


def test_class_importable_from_grpc_module() -> None:
    from pinecone.grpc import PineconeGRPC as _PineconeGRPC  # noqa: F401


def test_is_pinecone_subclass() -> None:
    assert issubclass(PineconeGRPC, Pinecone)


def test_constructor_inherits_signature() -> None:
    pc = PineconeGRPC(api_key="test-key")
    # Touch inherited namespace properties without calling them.
    assert hasattr(pc, "indexes")
    assert hasattr(pc, "collections")
    assert hasattr(pc, "inference")


def test_index_factory_requires_name_or_host() -> None:
    pc = PineconeGRPC(api_key="test-key")
    with pytest.raises(PineconeValueError, match="Either name or host must be specified"):
        pc.Index()


def test_index_factory_returns_grpc_index_when_host_provided() -> None:
    mock_channel = MagicMock()
    mock_grpc_module = MagicMock()
    mock_grpc_module.GrpcChannel.return_value = mock_channel

    pc = PineconeGRPC(api_key="test-key")
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_grpc_module}):
        idx = pc.Index(host="https://x-abc123.svc.pinecone.io")

    assert isinstance(idx, GrpcIndex)


def test_index_factory_strips_legacy_pool_threads() -> None:
    mock_channel = MagicMock()
    mock_grpc_module = MagicMock()
    mock_grpc_module.GrpcChannel.return_value = mock_channel

    pc = PineconeGRPC(api_key="test-key")
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_grpc_module}):
        idx = pc.Index(host="https://x-abc123.svc.pinecone.io", pool_threads=4)

    assert isinstance(idx, GrpcIndex)


def test_index_factory_rejects_truly_unknown_kwargs() -> None:
    pc = PineconeGRPC(api_key="test-key")
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        pc.Index(host="https://x-abc123.svc.pinecone.io", bogus=True)
