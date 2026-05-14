"""Unit tests for the GRPCIndex capitalisation alias (BCG-141) and GrpcIndex.describe_namespace."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pinecone.grpc
from pinecone.grpc import GRPCIndex, GrpcIndex
from pinecone.models.namespaces.models import NamespaceDescription

_MOCK_GRPC_MODULE_PATH = "pinecone._grpc"


def _make_grpc_index_via_alias(mock_channel: MagicMock) -> GRPCIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GRPCIndex(
            host="https://x-abc.svc.pinecone.io",
            api_key="k",
        )


def _make_grpc_index(mock_channel: MagicMock) -> GrpcIndex:
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {_MOCK_GRPC_MODULE_PATH: mock_module}):
        return GrpcIndex(
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


class TestGrpcDescribeNamespace:
    def test_grpc_describe_namespace_routes_to_channel(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "movies",
            "record_count": 42,
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.describe_namespace(name="movies")

        mock_channel.describe_namespace.assert_called_once_with("movies", timeout_s=None)
        assert isinstance(result, NamespaceDescription)
        assert result.name == "movies"
        assert result.record_count == 42
        assert result.schema is None
        assert result.indexed_fields is None

    def test_grpc_describe_namespace_accepts_legacy_namespace_kwarg(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "movies",
            "record_count": 7,
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.describe_namespace(namespace="movies")  # type: ignore[call-arg]

        mock_channel.describe_namespace.assert_called_once_with("movies", timeout_s=None)
        assert result.name == "movies"
        assert result.record_count == 7

    def test_grpc_describe_namespace_both_kwargs_raise(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match=r"name=.*namespace="):
            idx.describe_namespace(name="movies", namespace="movies")  # type: ignore[call-arg]

    def test_grpc_describe_namespace_empty_name_raises(self) -> None:
        from pinecone.errors.exceptions import ValidationError

        mock_channel = MagicMock()
        idx = _make_grpc_index(mock_channel)

        with pytest.raises(ValidationError, match="non-empty"):
            idx.describe_namespace(name="")

        with pytest.raises(ValidationError, match="non-empty"):
            idx.describe_namespace(name="   ")

    def test_grpc_describe_namespace_with_schema_and_indexed_fields(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "docs",
            "record_count": 100,
            "schema": {"fields": {"genre": {"filterable": True}}},
            "indexed_fields": ["genre", "year"],
        }
        idx = _make_grpc_index(mock_channel)

        result = idx.describe_namespace(name="docs")

        assert result.name == "docs"
        assert result.record_count == 100
        assert result.schema is not None
        assert result.schema.fields["genre"].filterable is True
        assert result.indexed_fields is not None
        assert result.indexed_fields.fields == ["genre", "year"]

    def test_grpc_describe_namespace_passes_timeout(self) -> None:
        mock_channel = MagicMock()
        mock_channel.describe_namespace.return_value = {
            "name": "ns1",
            "record_count": 0,
        }
        idx = _make_grpc_index(mock_channel)

        idx.describe_namespace(name="ns1", timeout=5.0)

        mock_channel.describe_namespace.assert_called_once_with("ns1", timeout_s=5.0)
