"""Pinecone Python SDK."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

# Lightweight imports — exception classes and config are small and always needed.
from pinecone._internal.config import PineconeConfig
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    NotFoundError,
    PineconeError,
    UnauthorizedError,
    ValidationError,
)

__all__ = [
    "__version__",
    "ApiError",
    "AsyncPinecone",
    "ConflictError",
    "Index",
    "NotFoundError",
    "Pinecone",
    "PineconeConfig",
    "PineconeError",
    "UnauthorizedError",
    "ValidationError",
]

# Lazy-load heavy classes to keep cold import under 10ms.
# Importing Pinecone/AsyncPinecone/Index eagerly pulls in httpx (~120ms).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Pinecone": ("pinecone._client", "Pinecone"),
    "AsyncPinecone": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "Index": ("pinecone.index", "Index"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        value = getattr(mod, attr)
        # Cache on the module so subsequent accesses skip __getattr__
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pinecone' has no attribute {name!r}")
