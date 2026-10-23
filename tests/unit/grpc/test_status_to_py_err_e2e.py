"""``status_to_py_err`` (rust/src/transport.rs) driven through a real gRPC channel (#47).

The in-process ``VectorService`` from ``tests/unit/conformance/_grpc_harness.py``
aborts the rpc with a chosen status code, message and trailers; the SDK's real
tonic channel receives that status and the real ``status_to_py_err`` turns it
into the Python exception asserted here. Nothing between the server and the
assertion is mocked, so a changed match arm in the Rust mapping fails these
tests.
"""

from __future__ import annotations

from collections.abc import Iterator

import grpc
import pytest

from pinecone import RetryConfig
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    ServiceError,
    UnauthorizedError,
)
from pinecone.grpc import GrpcIndex
from tests.unit.conformance._grpc_harness import VectorServiceHarness


@pytest.fixture(scope="module")
def harness() -> Iterator[VectorServiceHarness]:
    h = VectorServiceHarness()
    h.start()
    yield h
    h.stop()


@pytest.fixture(scope="module")
def index(harness: VectorServiceHarness) -> Iterator[GrpcIndex]:
    idx = GrpcIndex(host=harness.host, api_key="status-mapping-api-key", secure=False)
    yield idx
    idx.close()


@pytest.fixture(scope="module")
def no_retry_index(harness: VectorServiceHarness) -> Iterator[GrpcIndex]:
    # UNAVAILABLE, RESOURCE_EXHAUSTED and ABORTED are retried by the transport;
    # with no retries the first status is the one mapped, so each test is one rpc.
    idx = GrpcIndex(
        host=harness.host,
        api_key="status-mapping-api-key",
        secure=False,
        retry_config=RetryConfig(max_retries=0),
    )
    yield idx
    idx.close()


@pytest.fixture
def server(harness: VectorServiceHarness) -> VectorServiceHarness:
    harness.reset()
    return harness


def test_not_found_maps_to_not_found_error(server: VectorServiceHarness, index: GrpcIndex) -> None:
    server.abort(
        "Fetch",
        grpc.StatusCode.NOT_FOUND,
        "Namespace missing-ns not found",
        trailing_metadata=(("x-request-id", "req-404"),),
    )

    with pytest.raises(NotFoundError) as info:
        index.fetch(ids=["vec-1"], namespace="missing-ns")

    exc = info.value
    assert type(exc) is NotFoundError
    assert exc.status_code == 404
    assert exc.error_code == "NOT_FOUND"
    assert exc.message == "Namespace missing-ns not found"
    assert exc.body == {"error": {"code": "NOT_FOUND", "message": "Namespace missing-ns not found"}}
    assert exc.request_id == "req-404"
    assert len(server.calls) == 1


def test_not_found_without_request_id_trailer(
    server: VectorServiceHarness, index: GrpcIndex
) -> None:
    server.abort("Fetch", grpc.StatusCode.NOT_FOUND, "Vector not found")

    with pytest.raises(NotFoundError) as info:
        index.fetch(ids=["vec-1"], namespace="ns")

    assert info.value.request_id is None
    assert info.value.status_code == 404


# The #47 table. ApiError rows carry status_code, error_code, body and request_id;
# the rest get only the message, so a moved row fails on type, fields or str().
API_ERROR_CASES: list[tuple[grpc.StatusCode, type[ApiError], int]] = [
    (grpc.StatusCode.NOT_FOUND, NotFoundError, 404),
    (grpc.StatusCode.ALREADY_EXISTS, ConflictError, 409),
    (grpc.StatusCode.UNAUTHENTICATED, UnauthorizedError, 401),
    (grpc.StatusCode.PERMISSION_DENIED, ForbiddenError, 403),
    (grpc.StatusCode.INTERNAL, ServiceError, 500),
    (grpc.StatusCode.UNKNOWN, ServiceError, 500),
    (grpc.StatusCode.INVALID_ARGUMENT, ApiError, 400),
    (grpc.StatusCode.RESOURCE_EXHAUSTED, ApiError, 429),
]

NON_API_ERROR_CASES: list[tuple[grpc.StatusCode, str, type[PineconeError]]] = [
    (grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline exceeded", PineconeTimeoutError),
    (grpc.StatusCode.CANCELLED, "Timeout expired", PineconeTimeoutError),
    (grpc.StatusCode.UNAVAILABLE, "upstream connect error", PineconeConnectionError),
    (grpc.StatusCode.CANCELLED, "Cancelled by peer", PineconeError),
    (grpc.StatusCode.FAILED_PRECONDITION, "Index not ready", PineconeError),
    (grpc.StatusCode.ABORTED, "Concurrent write conflict", PineconeError),
    (grpc.StatusCode.OUT_OF_RANGE, "Offset out of range", PineconeError),
    (grpc.StatusCode.UNIMPLEMENTED, "Method not implemented", PineconeError),
    (grpc.StatusCode.DATA_LOSS, "Data loss", PineconeError),
]


@pytest.mark.parametrize(
    ("code", "exc_class", "status_code"),
    API_ERROR_CASES,
    ids=[case[0].name for case in API_ERROR_CASES],
)
def test_api_error_codes_map_to_typed_exception(
    server: VectorServiceHarness,
    no_retry_index: GrpcIndex,
    code: grpc.StatusCode,
    exc_class: type[ApiError],
    status_code: int,
) -> None:
    message = f"server said {code.name.lower()}"
    server.abort(
        "Fetch", code, message, trailing_metadata=(("x-pinecone-request-id", "req-fallback"),)
    )

    with pytest.raises(PineconeError) as info:
        no_retry_index.fetch(ids=["vec-1"], namespace="ns")

    exc = info.value
    assert type(exc) is exc_class
    assert isinstance(exc, ApiError)
    assert exc.status_code == status_code
    assert exc.error_code == code.name
    assert exc.message == message
    assert exc.body == {"error": {"code": code.name, "message": message}}
    assert exc.request_id == "req-fallback"
    assert str(exc) == f"[{status_code} {code.name}] {message} (request_id: req-fallback)"
    assert len(server.calls) == 1


def test_x_request_id_wins_over_fallback_trailer(
    server: VectorServiceHarness, no_retry_index: GrpcIndex
) -> None:
    server.abort(
        "Fetch",
        grpc.StatusCode.INVALID_ARGUMENT,
        "bad vector",
        trailing_metadata=(
            ("x-pinecone-request-id", "req-fallback"),
            ("x-request-id", "req-primary"),
        ),
    )

    with pytest.raises(ApiError) as info:
        no_retry_index.fetch(ids=["vec-1"], namespace="ns")

    assert info.value.request_id == "req-primary"


@pytest.mark.parametrize(
    ("code", "message", "exc_class"),
    NON_API_ERROR_CASES,
    ids=[f"{case[0].name}-{case[1]}" for case in NON_API_ERROR_CASES],
)
def test_non_api_error_codes_map_to_plain_exception(
    server: VectorServiceHarness,
    no_retry_index: GrpcIndex,
    code: grpc.StatusCode,
    message: str,
    exc_class: type[PineconeError],
) -> None:
    server.abort("Fetch", code, message, trailing_metadata=(("x-request-id", "req-ignored"),))

    with pytest.raises(PineconeError) as info:
        no_retry_index.fetch(ids=["vec-1"], namespace="ns")

    exc = info.value
    assert type(exc) is exc_class
    assert not isinstance(exc, ApiError)
    assert exc.message == message
    assert str(exc) == message
    assert getattr(exc, "status_code", None) is None
    assert getattr(exc, "request_id", None) is None
    assert len(server.calls) == 1
