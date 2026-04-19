"""Tests that invalid timeout_s values raise PineconeValueError (not PanicException).

These tests exercise the Rust-backed GrpcChannel directly to verify that
secs_to_duration guards against negative, NaN, and infinite values before
entering allow_threads, where a panic would be unrecoverable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pinecone._grpc", reason="Rust extension not available; run maturin develop first")

from pinecone._grpc import GrpcChannel  # type: ignore[import-not-found]  # noqa: E402
from pinecone.errors.exceptions import PineconeValueError  # noqa: E402

_ENDPOINT = "https://test-index-abc123.svc.us-east-1-aws.pinecone.io:443"
_API_KEY = "test-api-key"
_API_VERSION = "2025-10"
_VERSION = "0.1.0"


def _make_channel() -> GrpcChannel:
    return GrpcChannel(
        _ENDPOINT,
        _API_KEY,
        _API_VERSION,
        _VERSION,
        secure=False,
    )


class TestUpsertTimeoutValidation:
    def test_negative_timeout_raises_value_error(self) -> None:
        ch = _make_channel()
        with pytest.raises(PineconeValueError):
            ch.upsert(vectors=[{"id": "v1", "values": [0.1]}], timeout_s=-1.0)

    def test_nan_timeout_raises_value_error(self) -> None:
        ch = _make_channel()
        with pytest.raises(PineconeValueError):
            ch.upsert(vectors=[{"id": "v1", "values": [0.1]}], timeout_s=float("nan"))

    def test_inf_timeout_raises_value_error(self) -> None:
        ch = _make_channel()
        with pytest.raises(PineconeValueError):
            ch.upsert(vectors=[{"id": "v1", "values": [0.1]}], timeout_s=float("inf"))
