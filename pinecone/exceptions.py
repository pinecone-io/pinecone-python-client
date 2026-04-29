"""Backwards-compatibility shim for :mod:`pinecone.errors.exceptions`.

Re-exports exception classes that used to live at :mod:`pinecone.exceptions`
before the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module or directly from :mod:`pinecone`.

:meta private:
"""

from __future__ import annotations

from pinecone.errors.exceptions import (
    ForbiddenException,
    ListConversionException,
    NotFoundException,
    PineconeApiAttributeError,
    PineconeApiException,
    PineconeApiKeyError,
    PineconeApiTypeError,
    PineconeApiValueError,
    PineconeConfigurationError,
    PineconeException,
    PineconeProtocolError,
    ServiceException,
    UnauthorizedException,
)

__all__ = [
    "ForbiddenException",
    "ListConversionException",
    "NotFoundException",
    "PineconeApiAttributeError",
    "PineconeApiException",
    "PineconeApiKeyError",
    "PineconeApiTypeError",
    "PineconeApiValueError",
    "PineconeConfigurationError",
    "PineconeException",
    "PineconeProtocolError",
    "ServiceException",
    "UnauthorizedException",
]
