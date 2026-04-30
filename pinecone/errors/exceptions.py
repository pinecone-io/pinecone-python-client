"""Pinecone SDK exception hierarchy."""

from __future__ import annotations

TYPE_CHECKING = False

if TYPE_CHECKING:
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
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.reason = reason
        self.headers = headers
        self.error_code = error_code
        self.request_id = request_id
        super().__init__(message)

    def __str__(self) -> str:
        try:
            prefix = f"{self.status_code}"
            if self.error_code:
                prefix = f"{prefix} {self.error_code}"
            base = f"[{prefix}] {self.message}"
            if self.request_id:
                base = f"{base} (request_id: {self.request_id})"
            return base
        except Exception:
            # Never let __str__ raise — that would mask the original error.
            try:
                return f"[{self.status_code}] {self.message}"
            except Exception:
                return "<ApiError: unrenderable>"

    def __repr__(self) -> str:
        try:
            msg = self.message
            if len(msg) > 100:
                msg = msg[:97] + "..."
        except Exception:
            msg = "<unrenderable>"
        parts = [
            f"status_code={self.status_code}",
            f"message={msg!r}",
        ]
        if self.error_code is not None:
            parts.append(f"error_code={self.error_code!r}")
        if self.request_id is not None:
            parts.append(f"request_id={self.request_id!r}")
        if self.body is not None:
            try:
                parts.append(f"body={self.body!r}")
            except Exception:
                parts.append("body=<unrenderable>")
        return f"{type(self).__name__}({', '.join(parts)})"


class NotFoundError(ApiError):
    """404 Not Found."""

    def __init__(
        self,
        message: str = "Resource not found",
        status_code: int = 404,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class ConflictError(ApiError):
    """409 Conflict."""

    def __init__(
        self,
        message: str = "Resource conflict",
        status_code: int = 409,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class UnauthorizedError(ApiError):
    """401 Unauthorized."""

    def __init__(
        self,
        message: str = "Invalid or missing API key",
        status_code: int = 401,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class ForbiddenError(ApiError):
    """403 Forbidden."""

    def __init__(
        self,
        message: str = "Forbidden",
        status_code: int = 403,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class ServiceError(ApiError):
    """5xx server error."""

    def __init__(
        self,
        message: str = "Internal server error",
        status_code: int = 500,
        body: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            body=body,
            reason=reason,
            headers=headers,
            error_code=error_code,
            request_id=request_id,
        )


class IndexInitFailedError(PineconeError):
    """Raised when an index fails to initialize."""

    def __init__(self, index_name: str) -> None:
        super().__init__(f"Index '{index_name}' entered InitializationFailed state")
        self.index_name = index_name


class PineconeTimeoutError(PineconeError, TimeoutError):
    """Raised when an operation exceeds its timeout.

    Multiply inherits from Python's built-in :class:`TimeoutError` so that
    ``except TimeoutError`` blocks in caller code catch SDK timeouts without
    having to import a Pinecone-specific class. This is the same pattern used
    by :class:`PineconeValueError` (extends :class:`ValueError`).
    """


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
ValidationError = PineconeValueError  # Deprecated: use PineconeValueError instead

# ---------------------------------------------------------------------------
# Legacy name aliases — :meta private:
# New code should use the canonical names above.
# ---------------------------------------------------------------------------

# Backcompat alias, :meta private:
PineconeException = PineconeError
# Backcompat alias, :meta private:
PineconeApiException = ApiError
# Backcompat alias, :meta private:
NotFoundException = NotFoundError
# Backcompat alias, :meta private:
UnauthorizedException = UnauthorizedError
# Backcompat alias, :meta private:
ForbiddenException = ForbiddenError
# Backcompat alias, :meta private:
ServiceException = ServiceError
# Backcompat alias, :meta private:
PineconeConfigurationError = PineconeValueError
# Backcompat alias, :meta private:
PineconeProtocolError = PineconeError
# Backcompat alias, :meta private:
PineconeApiTypeError = PineconeTypeError
# Backcompat alias, :meta private:
PineconeApiValueError = PineconeValueError
# Backcompat alias, :meta private:
PineconeApiAttributeError = PineconeError
# Backcompat alias, :meta private:
PineconeApiKeyError = PineconeError
# Backcompat alias, :meta private:
ListConversionException = PineconeError
