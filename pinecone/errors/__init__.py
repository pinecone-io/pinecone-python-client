"""Import path for the SDK's exceptions.

``from pinecone.errors import NotFoundError`` and
``from pinecone import NotFoundError`` reach the same class. The classes are
defined in :mod:`pinecone.errors.exceptions`, and
`Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__
covers which call produces which.
"""

from __future__ import annotations

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    FailedPreconditionError,
    ForbiddenError,
    IndexInitFailedError,
    IndexTerminatedError,
    NotFoundError,
    PayloadTooLargeError,
    PaymentRequiredError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    PineconeTypeError,
    PineconeValueError,
    RateLimitError,
    RateLimitException,
    ResponseParsingError,
    ServiceError,
    UnauthorizedError,
    ValidationError,
)

__all__ = [
    "ApiError",
    "ConflictError",
    "FailedPreconditionError",
    "ForbiddenError",
    "IndexInitFailedError",
    "IndexTerminatedError",
    "NotFoundError",
    "PayloadTooLargeError",
    "PaymentRequiredError",
    "PineconeConnectionError",
    "PineconeError",
    "PineconeTimeoutError",
    "PineconeTypeError",
    "PineconeValueError",
    "RateLimitError",
    "RateLimitException",
    "ResponseParsingError",
    "ServiceError",
    "UnauthorizedError",
    "ValidationError",
]
