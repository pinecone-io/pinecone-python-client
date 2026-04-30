"""Tests for exception __str__ and __repr__ methods."""

from __future__ import annotations

import pytest

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PineconeError,
    PineconeTypeError,
    PineconeValueError,
    ResponseParsingError,
    ServiceError,
    UnauthorizedError,
)


class TestApiErrorStr:
    def test_str_includes_status_code(self) -> None:
        err = ApiError("Something went wrong", status_code=400)
        assert str(err) == "[400] Something went wrong"

    def test_str_message_no_prefix(self) -> None:
        err = ApiError("Something went wrong", status_code=400)
        assert err.message == "Something went wrong"

    def test_str_500_error(self) -> None:
        err = ApiError("Internal error", status_code=500)
        assert str(err) == "[500] Internal error"


class TestSubclassStr:
    def test_not_found_str(self) -> None:
        err = NotFoundError()
        assert str(err) == "[404] Resource not found"

    def test_service_error_str(self) -> None:
        err = ServiceError()
        assert str(err) == "[500] Internal server error"

    def test_unauthorized_str(self) -> None:
        err = UnauthorizedError()
        assert str(err) == "[401] Invalid or missing API key"

    def test_not_found_custom_message_str(self) -> None:
        err = NotFoundError(message="Index 'foo' not found")
        assert str(err) == "[404] Index 'foo' not found"

    def test_message_attribute_unchanged(self) -> None:
        err = NotFoundError()
        assert err.message == "Resource not found"

    def test_subclass_accepts_error_code_and_request_id(self) -> None:
        for cls, default_status in [
            (NotFoundError, 404),
            (ConflictError, 409),
            (UnauthorizedError, 401),
            (ForbiddenError, 403),
            (ServiceError, 500),
        ]:
            err = cls(error_code="X", request_id="Y")  # type: ignore[call-arg]
            assert err.error_code == "X"
            assert err.request_id == "Y"
            assert err.status_code == default_status
            assert "X" in str(err)
            assert "Y" in str(err)


class TestApiErrorStrWithMetadata:
    def test_str_with_error_code(self) -> None:
        err = ApiError("msg", status_code=400, error_code="INVALID_ARGUMENT")
        assert str(err) == "[400 INVALID_ARGUMENT] msg"

    def test_str_with_request_id(self) -> None:
        err = ApiError("msg", status_code=500, request_id="abc-123")
        assert str(err) == "[500] msg (request_id: abc-123)"

    def test_str_with_both(self) -> None:
        err = ApiError("msg", status_code=400, error_code="INVALID_ARGUMENT", request_id="abc-123")
        assert str(err) == "[400 INVALID_ARGUMENT] msg (request_id: abc-123)"

    def test_str_empty_error_code_treated_as_absent(self) -> None:
        err = ApiError("msg", status_code=400, error_code="", request_id="")
        assert str(err) == "[400] msg"

    def test_str_does_not_raise_on_weird_attributes(self) -> None:
        err = ApiError("msg", status_code=400)
        err.error_code = 123  # type: ignore[assignment]
        result = str(err)
        assert isinstance(result, str)


class TestApiErrorRepr:
    def test_basic_repr(self) -> None:
        err = ApiError("Something went wrong", status_code=400)
        result = repr(err)
        assert result == "ApiError(status_code=400, message='Something went wrong')"

    def test_repr_with_body(self) -> None:
        err = ApiError(
            "Bad request",
            status_code=400,
            body={"message": "invalid dimension"},
        )
        result = repr(err)
        assert "status_code=400" in result
        assert "message='Bad request'" in result
        assert "body={'message': 'invalid dimension'}" in result

    def test_repr_without_body(self) -> None:
        err = ApiError("Server error", status_code=500)
        result = repr(err)
        assert "body" not in result

    def test_repr_truncates_long_message(self) -> None:
        long_msg = "x" * 200
        err = ApiError(long_msg, status_code=500)
        result = repr(err)
        # Message should be truncated to 97 chars + "..."
        assert "..." in result
        assert len(long_msg) > 100  # confirm original is long
        # The repr message portion should be at most ~100 chars
        assert "x" * 97 + "..." in result

    def test_repr_exactly_100_chars_not_truncated(self) -> None:
        msg = "a" * 100
        err = ApiError(msg, status_code=400)
        result = repr(err)
        assert "..." not in result

    def test_repr_omits_headers(self) -> None:
        err = ApiError(
            "error",
            status_code=400,
            headers={"X-Request-Id": "abc123"},
        )
        result = repr(err)
        assert "headers" not in result
        assert "abc123" not in result


class TestApiErrorReprWithMetadata:
    def test_repr_includes_error_code(self) -> None:
        err = ApiError("msg", status_code=400, error_code="INVALID_ARGUMENT")
        result = repr(err)
        assert "error_code='INVALID_ARGUMENT'" in result

    def test_repr_includes_request_id(self) -> None:
        err = ApiError("msg", status_code=400, request_id="abc-123")
        result = repr(err)
        assert "request_id='abc-123'" in result

    def test_repr_omits_error_code_when_none(self) -> None:
        err = ApiError("msg", status_code=400)
        result = repr(err)
        assert "error_code" not in result

    def test_repr_omits_request_id_when_none(self) -> None:
        err = ApiError("msg", status_code=400)
        result = repr(err)
        assert "request_id" not in result

    def test_repr_does_not_raise_on_weird_body(self) -> None:
        class BadRepr:
            def __repr__(self) -> str:
                raise RuntimeError("repr failed")

        err = ApiError("msg", status_code=400)
        err.body = BadRepr()  # type: ignore[assignment]
        result = repr(err)
        assert isinstance(result, str)
        assert "unrenderable" in result


class TestSubclassRepr:
    def test_not_found_error_shows_class_name(self) -> None:
        err = NotFoundError()
        result = repr(err)
        assert result.startswith("NotFoundError(")
        assert "status_code=404" in result

    def test_conflict_error_shows_class_name(self) -> None:
        err = ConflictError()
        result = repr(err)
        assert result.startswith("ConflictError(")
        assert "status_code=409" in result

    def test_unauthorized_error_shows_class_name(self) -> None:
        err = UnauthorizedError()
        result = repr(err)
        assert result.startswith("UnauthorizedError(")
        assert "status_code=401" in result

    def test_forbidden_error_shows_class_name(self) -> None:
        err = ForbiddenError()
        result = repr(err)
        assert result.startswith("ForbiddenError(")
        assert "status_code=403" in result

    def test_service_error_shows_class_name(self) -> None:
        err = ServiceError()
        result = repr(err)
        assert result.startswith("ServiceError(")
        assert "status_code=500" in result

    def test_not_found_with_custom_message(self) -> None:
        err = NotFoundError(message="Index 'foo' not found")
        result = repr(err)
        assert result == "NotFoundError(status_code=404, message=\"Index 'foo' not found\")"

    def test_subclass_with_body(self) -> None:
        err = NotFoundError(
            message="Not found",
            body={"error": "resource missing"},
        )
        result = repr(err)
        assert "NotFoundError(" in result
        assert "body=" in result

    def test_subclass_accepts_reason_and_headers(self) -> None:
        headers = {"X-Request-Id": "abc"}

        not_found = NotFoundError(reason="test", headers=headers)
        assert not_found.reason == "test"
        assert not_found.headers == headers
        assert not_found.status_code == 404

        conflict = ConflictError(reason="test", headers=headers)
        assert conflict.reason == "test"
        assert conflict.headers == headers
        assert conflict.status_code == 409

        unauthorized = UnauthorizedError(reason="test", headers=headers)
        assert unauthorized.reason == "test"
        assert unauthorized.headers == headers
        assert unauthorized.status_code == 401

        forbidden = ForbiddenError(reason="test", headers=headers)
        assert forbidden.reason == "test"
        assert forbidden.headers == headers
        assert forbidden.status_code == 403

        service = ServiceError(reason="test", headers=headers)
        assert service.reason == "test"
        assert service.headers == headers
        assert service.status_code == 500


class TestValidationErrorPath:
    def test_value_error_with_path(self) -> None:
        err = PineconeValueError("score must be a number, got str", path="records[3].metadata.score")
        assert str(err) == "at records[3].metadata.score: score must be a number, got str"

    def test_value_error_without_path(self) -> None:
        err = PineconeValueError("invalid input")
        assert str(err) == "invalid input"

    def test_value_error_with_none_path(self) -> None:
        err = PineconeValueError("invalid", path=None)
        assert str(err) == "invalid"

    def test_value_error_with_empty_path(self) -> None:
        err = PineconeValueError("invalid", path="")
        assert str(err) == "invalid"

    def test_type_error_with_path(self) -> None:
        err = PineconeTypeError("expected int, got str", path="records[0].values")
        assert str(err) == "at records[0].values: expected int, got str"

    def test_type_error_without_path(self) -> None:
        err = PineconeTypeError("expected int, got str")
        assert str(err) == "expected int, got str"

    def test_str_does_not_raise_on_weird_path(self) -> None:
        err = PineconeValueError("msg", path="some.path")
        err.path = 123  # type: ignore[assignment]
        result = str(err)
        assert isinstance(result, str)

    def test_caught_as_value_error(self) -> None:
        caught = False
        try:
            raise PineconeValueError("x", path="y")
        except ValueError:
            caught = True
        assert caught

    def test_caught_as_type_error(self) -> None:
        caught = False
        try:
            raise PineconeTypeError("x", path="y")
        except TypeError:
            caught = True
        assert caught

    def test_path_attribute_accessible(self) -> None:
        err = PineconeValueError("msg", path="a.b.c")
        assert err.path == "a.b.c"

    def test_path_default_none(self) -> None:
        err = PineconeValueError("x")
        assert err.path is None


class TestResponseParsingErrorStr:
    def test_str_without_cause(self) -> None:
        err = ResponseParsingError("Failed to parse")
        assert str(err) == "Failed to parse"

    def test_str_with_cause(self) -> None:
        err = ResponseParsingError(
            "Failed to parse describe response",
            cause=ValueError("missing 'host' field"),
        )
        assert str(err) == "Failed to parse describe response (caused by ValueError: missing 'host' field)"

    def test_str_with_msgspec_validation_error(self) -> None:
        msgspec = pytest.importorskip("msgspec")

        class _Stub(msgspec.Struct):
            host: str

        try:
            msgspec.json.decode(b'{"foo": "bar"}', type=_Stub)
        except msgspec.ValidationError as e:
            cause = e
        else:
            pytest.skip("msgspec did not raise ValidationError as expected")

        err = ResponseParsingError("Failed to parse describe_index response", cause=cause)
        result = str(err)
        assert "Failed to parse describe_index response" in result
        assert "ValidationError" in result

    def test_cause_attribute_accessible(self) -> None:
        original = ValueError("original cause")
        err = ResponseParsingError("parse failed", cause=original)
        assert err.cause is original

    def test_str_does_not_raise_on_weird_cause(self) -> None:
        class BadStrError(Exception):
            def __str__(self) -> str:
                raise RuntimeError("nope")

        err = ResponseParsingError("parse failed", cause=BadStrError())
        result = str(err)
        assert result  # does not raise
        assert "<unrenderable>" in result

    def test_str_does_not_raise_on_none_cause(self) -> None:
        err = ResponseParsingError("parse failed", cause=None)
        result = str(err)
        assert result == "parse failed"

    def test_caught_as_pineconeerror(self) -> None:
        caught = False
        try:
            raise ResponseParsingError("x")
        except PineconeError:
            caught = True
        assert caught
