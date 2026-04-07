"""Pinecone SDK exception hierarchy."""

from __future__ import annotations

from typing import Any


class PineconeError(Exception):
    """Base exception for all Pinecone SDK errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ApiError(PineconeError):
    """Server returned an error response."""

    def __init__(
        self,
        message: str,
        status_code: int,
        body: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class NotFoundError(ApiError):
    """404 Not Found."""

    def __init__(
        self,
        message: str = "Resource not found",
        status_code: int = 404,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, body=body)


class ConflictError(ApiError):
    """409 Conflict."""

    def __init__(
        self,
        message: str = "Resource conflict",
        status_code: int = 409,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, body=body)


class UnauthorizedError(ApiError):
    """401 Unauthorized."""

    def __init__(
        self,
        message: str = "Invalid or missing API key",
        status_code: int = 401,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, body=body)


class ForbiddenError(ApiError):
    """403 Forbidden."""

    def __init__(
        self,
        message: str = "Forbidden",
        status_code: int = 403,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, body=body)


class ServiceError(ApiError):
    """5xx server error."""

    def __init__(
        self,
        message: str = "Internal server error",
        status_code: int = 500,
        body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, body=body)


class IndexInitFailedError(PineconeError):
    """Raised when an index fails to initialize."""

    def __init__(self, index_name: str) -> None:
        super().__init__(f"Index '{index_name}' entered InitializationFailed state")
        self.index_name = index_name


class PineconeTimeoutError(PineconeError):
    """Raised when an operation exceeds its timeout."""

    pass


class PineconeConnectionError(PineconeError):
    """Raised when a network-level connection fails.

    Covers DNS resolution failures, connection refused, read/write errors,
    and other transport-level problems.
    """

    pass


class PineconeValueError(PineconeError, ValueError):
    """Input validation failed — invalid value."""

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


class PineconeTypeError(PineconeError, TypeError):
    """Input validation failed — wrong type."""

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


class ResponseParsingError(PineconeError):
    """Raised when the SDK fails to parse an API response body.

    Wraps the underlying deserialization error (e.g. ``msgspec.ValidationError``)
    so that callers' ``except PineconeError`` blocks always catch it.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        self.cause = cause
        super().__init__(message)


# Backwards-compatible alias — most validation is value validation
ValidationError = PineconeValueError
