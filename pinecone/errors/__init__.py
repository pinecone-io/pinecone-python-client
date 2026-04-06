"""Exception hierarchy for the Pinecone SDK."""

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    IndexInitFailedError,
    NotFoundError,
    PineconeError,
    PineconeTimeoutError,
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
    "ServiceError",
    "UnauthorizedError",
    "ValidationError",
]
