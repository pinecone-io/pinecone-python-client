from pinecone.core.openapi.shared.exceptions import (
    PineconeException,
    PineconeApiAttributeError,
    PineconeApiTypeError,
    PineconeApiValueError,
    PineconeApiKeyError,
    PineconeApiException,
    NotFoundException,
    UnauthorizedException,
    ForbiddenException,
    ServiceException,
)
from .exceptions import PineconeConfigurationError, PineconeProtocolError, ListConversionException

PineconeNotFoundException = NotFoundException

__all__ = [
    "PineconeConfigurationError",
    "PineconeProtocolError",
    "PineconeException",
    "PineconeApiAttributeError",
    "PineconeApiTypeError",
    "PineconeApiValueError",
    "PineconeApiKeyError",
    "PineconeApiException",
    "NotFoundException",
    "PineconeNotFoundException",
    "UnauthorizedException",
    "ForbiddenException",
    "ServiceException",
    "ListConversionException",
]
