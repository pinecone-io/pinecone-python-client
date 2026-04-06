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
from pinecone.models.enums import (
    CloudProvider,
    DeletionProtection,
    Metric,
    PodType,
    VectorType,
)

__all__ = [
    "__version__",
    "ApiError",
    "AsyncIndex",
    "AsyncPinecone",
    "CloudProvider",
    "CollectionList",
    "CollectionModel",
    "ConflictError",
    "DeletionProtection",
    "Index",
    "IndexList",
    "IndexModel",
    "Metric",
    "NotFoundError",
    "Pinecone",
    "PineconeConfig",
    "PineconeError",
    "PodSpec",
    "PodType",
    "QueryResponse",
    "ServerlessSpec",
    "UnauthorizedError",
    "ValidationError",
    "VectorType",
]

# Lazy-load heavy classes to keep cold import under 10ms.
# Importing Pinecone/AsyncPinecone/Index eagerly pulls in httpx (~120ms).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Pinecone": ("pinecone._client", "Pinecone"),
    "AsyncIndex": ("pinecone.async_client.async_index", "AsyncIndex"),
    "AsyncPinecone": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "Index": ("pinecone.index", "Index"),
    "CollectionList": ("pinecone.models.collections.collection_list", "CollectionList"),
    "CollectionModel": ("pinecone.models.collections.collection_model", "CollectionModel"),
    "IndexList": ("pinecone.models.indexes.list", "IndexList"),
    "IndexModel": ("pinecone.models.indexes.index", "IndexModel"),
    "PodSpec": ("pinecone.models.indexes.specs", "PodSpec"),
    "QueryResponse": ("pinecone.models.vectors.responses", "QueryResponse"),
    "ServerlessSpec": ("pinecone.models.indexes.specs", "ServerlessSpec"),
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
