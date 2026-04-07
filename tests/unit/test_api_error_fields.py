"""Tests for ApiError reason and headers fields."""

from __future__ import annotations

import httpx
import pytest

from pinecone._internal.http_client import _raise_for_status
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ServiceError,
    UnauthorizedError,
)


class TestApiErrorFields:
    def test_api_error_exposes_reason(self) -> None:
        err = ApiError(
            message="not found",
            status_code=404,
            body=None,
            reason="Not Found",
        )
        assert err.reason == "Not Found"

    def test_api_error_exposes_headers(self) -> None:
        headers = {"x-request-id": "abc123", "content-type": "application/json"}
        err = ApiError(
            message="error",
            status_code=500,
            body=None,
            headers=headers,
        )
        assert err.headers == headers

    def test_api_error_defaults_none(self) -> None:
        err = ApiError(message="error", status_code=400)
        assert err.reason is None
        assert err.headers is None

    def test_subclass_propagates_reason_and_headers(self) -> None:
        headers = {"x-request-id": "def456"}
        for cls, code in [
            (NotFoundError, 404),
            (ConflictError, 409),
            (UnauthorizedError, 401),
            (ForbiddenError, 403),
            (ServiceError, 500),
        ]:
            err = cls(
                message="test",
                status_code=code,
                body=None,
                reason="Test Reason",
                headers=headers,
            )
            assert err.reason == "Test Reason"
            assert err.headers == headers


class TestRaiseForStatusPopulatesFields:
    def test_raise_for_status_populates_reason_and_headers(self) -> None:
        response = httpx.Response(
            status_code=404,
            json={"message": "Index not found"},
            headers={"x-request-id": "req-123", "content-type": "application/json"},
        )
        with pytest.raises(NotFoundError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.reason == "Not Found"
        assert err.headers is not None
        assert err.headers["x-request-id"] == "req-123"

    def test_raise_for_status_401(self) -> None:
        response = httpx.Response(
            status_code=401,
            json={"message": "Invalid API key"},
        )
        with pytest.raises(UnauthorizedError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.reason == "Unauthorized"
        assert err.headers is not None

    def test_raise_for_status_500(self) -> None:
        response = httpx.Response(
            status_code=500,
            json={"message": "Internal error"},
        )
        with pytest.raises(ServiceError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.reason == "Internal Server Error"
        assert err.headers is not None

    def test_raise_for_status_generic_error(self) -> None:
        response = httpx.Response(
            status_code=429,
            json={"message": "Rate limited"},
        )
        with pytest.raises(ApiError) as exc_info:
            _raise_for_status(response)
        err = exc_info.value
        assert err.reason == "Too Many Requests"
        assert err.headers is not None

    def test_raise_for_status_success_no_raise(self) -> None:
        response = httpx.Response(status_code=200)
        _raise_for_status(response)  # Should not raise
