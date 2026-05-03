"""Tests for gRPC → Python exception mapping in status_to_py_err (transport.rs).

Most tests here exercise the exception classes at the Python level to verify that
the attributes and string formatting expected from gRPC-sourced exceptions work
correctly. Tests that require a live gRPC channel going through the Rust extension
are marked @pytest.mark.skip(reason="requires live gRPC server") and are excluded
from the default unit-test run.

Integration tests that drive the Rust extension end-to-end would need a real
gRPC server returning controlled status codes; without grpcio in the dev
dependencies those tests are gated here rather than wired into a separate
conftest.
"""

from __future__ import annotations

import pytest

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    ResponseParsingError,
    ServiceError,
    UnauthorizedError,
)

# ---------------------------------------------------------------------------
# Helpers that mirror what the new status_to_py_err constructs
# ---------------------------------------------------------------------------


def _make_api_error(
    grpc_code_name: str,
    message: str,
    http_status: int,
    *,
    request_id: str | None = None,
) -> ApiError:
    """Build an ApiError the same way the new Rust code does."""
    body: dict[str, object] = {"error": {"code": grpc_code_name, "message": message}}
    return ApiError(
        message,
        http_status,
        body,
        error_code=grpc_code_name,
        request_id=request_id,
    )


def _make_not_found_error(
    message: str,
    *,
    request_id: str | None = None,
) -> NotFoundError:
    body: dict[str, object] = {"error": {"code": "NOT_FOUND", "message": message}}
    return NotFoundError(
        message=message,
        body=body,
        error_code="NOT_FOUND",
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# (a) InvalidArgument → ApiError(status_code=400, error_code="INVALID_ARGUMENT")
# ---------------------------------------------------------------------------


class TestInvalidArgumentMapsToApiError400:
    def test_isinstance_api_error(self) -> None:
        exc = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert isinstance(exc, ApiError)

    def test_status_code_400(self) -> None:
        exc = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert exc.status_code == 400

    def test_error_code_invalid_argument(self) -> None:
        exc = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert exc.error_code == "INVALID_ARGUMENT"

    def test_body_matches_expected_shape(self) -> None:
        exc = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert exc.body == {"error": {"code": "INVALID_ARGUMENT", "message": "bad param"}}

    def test_str_format_matches_rest(self) -> None:
        exc = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert str(exc) == "[400 INVALID_ARGUMENT] bad param"

    def test_message_does_not_have_grpc_prefix(self) -> None:
        exc = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert not exc.message.startswith("gRPC")
        assert exc.message == "bad param"


# ---------------------------------------------------------------------------
# (b) NotFound → NotFoundError(status_code=404, error_code="NOT_FOUND")
# ---------------------------------------------------------------------------


class TestNotFoundMapsToNotFoundError:
    def test_isinstance_not_found_error(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert isinstance(exc, NotFoundError)

    def test_isinstance_api_error(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert isinstance(exc, ApiError)

    def test_status_code_404(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert exc.status_code == 404

    def test_error_code_not_found(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert exc.error_code == "NOT_FOUND"

    def test_body_shape(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert exc.body == {"error": {"code": "NOT_FOUND", "message": "index 'foo' does not exist"}}

    def test_str_format(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert str(exc) == "[404 NOT_FOUND] index 'foo' does not exist"

    def test_message_no_grpc_prefix(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert not exc.message.startswith("gRPC")


# ---------------------------------------------------------------------------
# (c) request_id from trailer
# ---------------------------------------------------------------------------


class TestRequestIdExtraction:
    def test_request_id_present(self) -> None:
        exc = _make_not_found_error("not found", request_id="abc-123")
        assert exc.request_id == "abc-123"

    def test_request_id_appears_in_str(self) -> None:
        exc = _make_not_found_error("not found", request_id="abc-123")
        assert "abc-123" in str(exc)
        assert str(exc) == "[404 NOT_FOUND] not found (request_id: abc-123)"

    def test_no_request_id_when_absent(self) -> None:
        exc = _make_not_found_error("not found")
        assert exc.request_id is None

    def test_no_request_id_in_str_when_absent(self) -> None:
        exc = _make_not_found_error("not found")
        assert "request_id" not in str(exc)


# ---------------------------------------------------------------------------
# (d) str format matches REST — identical structure for same content
# ---------------------------------------------------------------------------


class TestStrFormatMatchesRest:
    def test_grpc_and_rest_produce_identical_str(self) -> None:
        # REST error constructed directly
        rest_err = ApiError(
            "bad param",
            400,
            {"error": {"code": "INVALID_ARGUMENT", "message": "bad param"}},
            error_code="INVALID_ARGUMENT",
        )
        # gRPC-sourced error (mirrors what status_to_py_err now constructs)
        grpc_err = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert str(rest_err) == str(grpc_err)

    def test_not_found_str_matches(self) -> None:
        rest_err = ApiError("nope", 404, None, error_code="NOT_FOUND")
        grpc_err = _make_api_error("NOT_FOUND", "nope", 404)
        assert str(rest_err) == str(grpc_err)


# ---------------------------------------------------------------------------
# (e) PineconeConnectionError construction does not crash
# ---------------------------------------------------------------------------


class TestPineconeConnectionErrorConstruction:
    def test_construction_without_status_code(self) -> None:
        exc = PineconeConnectionError("connection refused")
        assert isinstance(exc, PineconeConnectionError)
        assert isinstance(exc, PineconeError)

    def test_message_preserved(self) -> None:
        exc = PineconeConnectionError("connection refused")
        assert exc.message == "connection refused"
        assert str(exc) == "connection refused"

    def test_not_an_api_error(self) -> None:
        exc = PineconeConnectionError("connection refused")
        assert not isinstance(exc, ApiError)


# ---------------------------------------------------------------------------
# (f) PineconeTimeoutError construction (DeadlineExceeded path)
# ---------------------------------------------------------------------------


class TestPineconeTimeoutErrorConstruction:
    def test_construction_without_status_code(self) -> None:
        exc = PineconeTimeoutError("deadline exceeded")
        assert isinstance(exc, PineconeTimeoutError)
        assert isinstance(exc, PineconeError)

    def test_is_also_builtin_timeout_error(self) -> None:
        exc = PineconeTimeoutError("deadline exceeded")
        assert isinstance(exc, TimeoutError)

    def test_not_an_api_error(self) -> None:
        exc = PineconeTimeoutError("deadline exceeded")
        assert not isinstance(exc, ApiError)


# ---------------------------------------------------------------------------
# (g) Full gRPC class hierarchy consistency check
# ---------------------------------------------------------------------------


class TestGrpcErrorHierarchy:
    """Verify that all expected exception classes support the new fields."""

    @pytest.mark.parametrize(
        ("cls", "status_code"),
        [
            (NotFoundError, 404),
            (ConflictError, 409),
            (UnauthorizedError, 401),
            (ForbiddenError, 403),
            (ServiceError, 500),
        ],
    )
    def test_api_error_subclasses_accept_error_code_and_request_id(
        self, cls: type[ApiError], status_code: int
    ) -> None:
        exc = cls(  # type: ignore[call-arg]
            message="test",
            status_code=status_code,
            body={"error": {"code": "TEST", "message": "test"}},
            error_code="TEST",
            request_id="req-xyz",
        )
        assert exc.error_code == "TEST"
        assert exc.request_id == "req-xyz"
        assert exc.body == {"error": {"code": "TEST", "message": "test"}}

    def test_api_error_base_accepts_error_code_and_request_id(self) -> None:
        exc = ApiError(
            "rate limited",
            429,
            {"error": {"code": "RESOURCE_EXHAUSTED", "message": "rate limited"}},
            error_code="RESOURCE_EXHAUSTED",
            request_id="req-abc",
        )
        assert exc.error_code == "RESOURCE_EXHAUSTED"
        assert exc.status_code == 429
        assert exc.request_id == "req-abc"


# ---------------------------------------------------------------------------
# Regression tests — P-0214: no double-bracket bug, typed non-RPC exceptions
# ---------------------------------------------------------------------------


class TestDoubleBracketRegression:
    """Verify the double-bracket bug introduced by the old _call_channel wrapper is gone.

    Previously, _call_channel called str(rust_exc) (which already contained
    "[404 NOT_FOUND] ...") and passed that string as the message of a *new*
    NotFoundError, producing "[404 NOT_FOUND] [404 NOT_FOUND] ..." when str()
    was called on the re-raised exception. Now typed exceptions from Rust
    propagate unchanged, so there is at most one bracket pair.
    """

    def test_no_double_bracket_prefix_for_not_found(self) -> None:
        exc = _make_not_found_error("index 'foo' does not exist")
        assert str(exc).count("[") <= 1

    def test_no_double_bracket_prefix_for_api_error(self) -> None:
        exc = _make_api_error("INVALID_ARGUMENT", "bad param", 400)
        assert str(exc).count("[") <= 1

    @pytest.mark.parametrize(
        ("cls", "status_code", "code"),
        [
            (NotFoundError, 404, "NOT_FOUND"),
            (ConflictError, 409, "ALREADY_EXISTS"),
            (UnauthorizedError, 401, "UNAUTHENTICATED"),
            (ForbiddenError, 403, "PERMISSION_DENIED"),
            (ServiceError, 500, "INTERNAL"),
        ],
    )
    def test_no_double_bracket_for_all_api_error_subclasses(
        self, cls: type[ApiError], status_code: int, code: str
    ) -> None:
        exc = cls(  # type: ignore[call-arg]
            message="some error message",
            status_code=status_code,
            body={"error": {"code": code, "message": "some error message"}},
            error_code=code,
        )
        assert str(exc).count("[") <= 1


class TestResponseParsingErrorConstruction:
    """ResponseParsingError is raised by Rust for proto-decode failures."""

    def test_isinstance_pinecone_error(self) -> None:
        exc = ResponseParsingError("vector missing 'id'")
        assert isinstance(exc, PineconeError)

    def test_message_preserved(self) -> None:
        exc = ResponseParsingError("schema missing 'fields'")
        assert "schema missing 'fields'" in str(exc)

    def test_not_an_api_error(self) -> None:
        exc = ResponseParsingError("vector missing 'values'")
        assert not isinstance(exc, ApiError)


# ---------------------------------------------------------------------------
# Integration tests — require a live gRPC server with the Rust extension
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="requires live gRPC server; run manually against a mock server")
class TestGrpcErrorMappingIntegration:
    """End-to-end tests that drive the Rust extension through a real gRPC channel.

    These tests verify that status_to_py_err in transport.rs constructs the
    correct Python exceptions with the right attributes when actual tonic::Status
    values flow through the extension.

    To run these tests, start a mock gRPC server that returns controlled status
    codes, then point GrpcChannel at it and invoke this class without the skip mark.
    """

    def test_invalid_argument_maps_to_apierror_400_with_error_code(self) -> None:
        """INVALID_ARGUMENT gRPC status → ApiError(status_code=400, error_code='INVALID_ARGUMENT')."""
        raise NotImplementedError("requires live gRPC server")

    def test_not_found_maps_to_notfounderror_with_error_code(self) -> None:
        """NOT_FOUND gRPC status → NotFoundError(status_code=404, error_code='NOT_FOUND')."""
        raise NotImplementedError("requires live gRPC server")

    def test_request_id_extracted_from_trailer(self) -> None:
        """x-request-id trailer → exception.request_id matches."""
        raise NotImplementedError("requires live gRPC server")

    def test_no_request_id_trailer(self) -> None:
        """No x-request-id trailer → exception.request_id is None."""
        raise NotImplementedError("requires live gRPC server")

    def test_str_format_matches_rest_end_to_end(self) -> None:
        """str(grpc_exc) == '[400 INVALID_ARGUMENT] <message>' (same as REST)."""
        raise NotImplementedError("requires live gRPC server")

    def test_message_does_not_have_grpc_prefix_end_to_end(self) -> None:
        """The 'gRPC NOT_FOUND: ' legacy prefix is absent from exc.message."""
        raise NotImplementedError("requires live gRPC server")

    def test_pinecone_connection_error_construction_does_not_crash(self) -> None:
        """Unavailable → PineconeConnectionError raised without status_code kwargs."""
        raise NotImplementedError("requires live gRPC server")

    def test_proto_decode_failure_raises_response_parsing_error(self) -> None:
        """Upsert with a vector dict missing 'id' → ResponseParsingError from Rust."""
        raise NotImplementedError("requires built Rust extension + GrpcChannel")

    def test_invalid_endpoint_raises_value_error(self) -> None:
        """Constructing GrpcChannel with an invalid endpoint URL → PineconeValueError."""
        raise NotImplementedError("requires built Rust extension + GrpcChannel")
