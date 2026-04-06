"""Exception hierarchy for the Pinecone SDK."""

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    IndexInitFailedError,
    NotFoundError,
    PineconeError,
    PineconeTimeoutError,
    PineconeTypeError,
    PineconeValueError,
    ServiceError,
    UnauthorizedError,
    ValidationError,
)

__all__ = [
    "ApiError",
    "ConflictError",
    "ForbiddenError",
    "IndexInitFailedError",
    "NotFoundError",
    "PineconeError",
    "PineconeTimeoutError",
    "PineconeTypeError",
    "PineconeValueError",
    "ServiceError",
    "UnauthorizedError",
    "ValidationError",
]
