"""Benchmark GrpcIndex query/fetch dispatch (Python→Rust boundary)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pinecone.grpc import GrpcIndex

QUERY_RESPONSE = {
    "matches": [
        {"id": f"v{i}", "score": 0.9 - i * 0.01, "values": [], "metadata": None} for i in range(100)
    ],
    "namespace": "ns",
    "usage": {"read_units": 1},
}
FETCH_RESPONSE = {
    "vectors": {f"v{i}": {"values": [0.1] * 128, "metadata": None} for i in range(10)},
    "namespace": "ns",
}


@pytest.fixture(scope="module")
def grpc_index() -> GrpcIndex:
    mock_channel = MagicMock()
    mock_channel.query.return_value = QUERY_RESPONSE
    mock_channel.fetch.return_value = FETCH_RESPONSE
    mock_module = MagicMock()
    mock_module.GrpcChannel.return_value = mock_channel
    with patch.dict("sys.modules", {"pinecone._grpc": mock_module}):
        return GrpcIndex(host="test.svc.pinecone.io", api_key="test-key")


def test_query_dispatch_throughput(benchmark: pytest.FixtureRequest, grpc_index: GrpcIndex) -> None:
    benchmark(grpc_index.query, top_k=100, vector=[0.1] * 128)


def test_fetch_dispatch_throughput(benchmark: pytest.FixtureRequest, grpc_index: GrpcIndex) -> None:
    benchmark(grpc_index.fetch, ids=[f"v{i}" for i in range(10)])
