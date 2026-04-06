"""Pinecone Python SDK."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

# Lightweight imports — exception classes and config are small and always needed.
from pinecone._internal.config import PineconeConfig
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
    "ByocSpec",
    "CloudProvider",
    "CollectionList",
    "CollectionModel",
    "ConflictError",
    "DeletionProtection",
    "EmbedConfig",
    "EmbedModel",
    "IndexInitFailedError",
    "Index",
    "IndexList",
    "IndexModel",
    "IntegratedSpec",
    "Metric",
    "NotFoundError",
    "Pinecone",
    "PineconeConfig",
    "PineconeError",
    "PineconeTimeoutError",
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
    "CollectionList": ("pinecone.models.collections.list", "CollectionList"),
    "CollectionModel": ("pinecone.models.collections.model", "CollectionModel"),
    "IndexList": ("pinecone.models.indexes.list", "IndexList"),
    "IndexModel": ("pinecone.models.indexes.index", "IndexModel"),
    "ByocSpec": ("pinecone.models.indexes.specs", "ByocSpec"),
    "EmbedConfig": ("pinecone.models.indexes.specs", "EmbedConfig"),
    "EmbedModel": ("pinecone.models.enums", "EmbedModel"),
    "IntegratedSpec": ("pinecone.models.indexes.specs", "IntegratedSpec"),
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


def __dir__() -> list[str]:
    import builtins

    return builtins.list({*globals(), *__all__, *_LAZY_IMPORTS})
