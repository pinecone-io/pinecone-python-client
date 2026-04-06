"""Exception hierarchy for the Pinecone SDK."""

from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    IndexInitFailedError,
    NotFoundError,
    PineconeError,
    PineconeTimeoutError,
    UnauthorizedError,
    ValidationError,
)

__all__ = [
    "ApiError",
    "ConflictError",
    "IndexInitFailedError",
    "NotFoundError",
    "PineconeError",
    "PineconeTimeoutError",
    "UnauthorizedError",
    "ValidationError",
]
