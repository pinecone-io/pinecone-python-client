from .core.client.exceptions import (
    OpenApiException,
    ApiAttributeError,
    ApiTypeError,
    ApiValueError,
    ApiKeyError,
    ApiException,
    NotFoundException,
    UnauthorizedException,
    ForbiddenException,
    ServiceException,
)

class PineconeException(Exception):
    """The base exception class for all Pinecone client exceptions."""

class PineconeProtocolError(PineconeException):
    """Raised when something unexpected happens mid-request/response."""

__all__ = [
    "PineconeException",
    "PineconeProtocolError",
    "OpenApiException",
    "ApiAttributeError",
    "ApiTypeError",
    "ApiValueError",
    "ApiKeyError",
    "ApiException",
    "NotFoundException",
    "UnauthorizedException",
    "ForbiddenException",
    "ServiceException",
]
