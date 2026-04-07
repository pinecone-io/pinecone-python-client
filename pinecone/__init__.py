"""Pinecone Python SDK — vector database for similarity search.

Quick Start::

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key="your-api-key")  # or set PINECONE_API_KEY env var

    # Control plane: manage indexes
    pc.indexes.create(
        name="movie-recommendations",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Data plane: operate on vectors
    index = pc.index("movie-recommendations")
    index.upsert(vectors=[("movie-42", [0.012, -0.087, 0.153, ...])])
    results = index.query(vector=[0.012, -0.087, 0.153, ...], top_k=5)

    # Integrated inference: search with text (server-side embedding)
    index = pc.index("my-integrated-index")
    results = index.search(namespace="default", top_k=5, inputs={"text": "search query"})

The :class:`Pinecone` client manages indexes (control plane). Call
``pc.index(name)`` to get an :class:`Index` for vector operations (data plane).

For async usage, see :class:`AsyncPinecone`. For admin/org management,
see :class:`Admin`.
"""

from __future__ import annotations

import logging as _logging
import os as _os
from typing import Any

__version__ = "0.1.0"

if _os.environ.get("PINECONE_DEBUG"):
    _logging.getLogger("pinecone").setLevel(_logging.DEBUG)

# Lightweight imports — exception classes and config are small and always needed.
from pinecone._internal.config import PineconeConfig, RetryConfig
from pinecone.errors.exceptions import (
    ApiError,
    ConflictError,
    ForbiddenError,
    IndexInitFailedError,
    NotFoundError,
    PineconeConnectionError,
    PineconeError,
    PineconeTimeoutError,
    PineconeTypeError,
    PineconeValueError,
    ResponseParsingError,
    ServiceError,
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
    "Admin",
    "ApiError",
    "AsyncIndex",
    "AsyncPinecone",
    "BackupList",
    "BackupModel",
    "ByocSpec",
    "ByocSpecInfo",
    "CloudProvider",
    "CollectionList",
    "CollectionModel",
    "ConflictError",
    "CreateIndexFromBackupResponse",
    "DeletionProtection",
    "DescribeIndexStatsResponse",
    "EmbedConfig",
    "EmbedModel",
    "EmbeddingsList",
    "FetchByMetadataResponse",
    "FetchResponse",
    "Field",
    "ForbiddenError",
    "GrpcIndex",
    "PineconeFuture",
    "ImportList",
    "ImportModel",
    "Index",
    "IndexInitFailedError",
    "IndexList",
    "IndexModel",
    "IndexSpec",
    "IntegratedSpec",
    "ListNamespacesResponse",
    "ListResponse",
    "Metric",
    "ModelInfo",
    "ModelInfoList",
    "NamespaceDescription",
    "NotFoundError",
    "Pinecone",
    "PineconeConfig",
    "PineconeConnectionError",
    "PineconeError",
    "PineconeTimeoutError",
    "PineconeTypeError",
    "PineconeValueError",
    "PodSpec",
    "PodSpecInfo",
    "PodType",
    "QueryNamespacesResults",
    "QueryResponse",
    "QueryResultsAggregator",
    "RerankModel",
    "RetryConfig",
    "RerankResult",
    "ResponseInfo",
    "ResponseParsingError",
    "RestoreJobList",
    "RestoreJobModel",
    "SearchRecordsResponse",
    "ServerlessSpec",
    "ServerlessSpecInfo",
    "ServiceError",
    "SparseValues",
    "StartImportResponse",
    "UnauthorizedError",
    "UpdateResponse",
    "UpsertRecordsResponse",
    "UpsertResponse",
    "ValidationError",
    "Vector",
    "VectorType",
]

# Lazy-load heavy classes to keep cold import under 10ms.
# Importing Pinecone/AsyncPinecone/Index eagerly pulls in httpx (~120ms).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Admin": ("pinecone.admin", "Admin"),
    "AsyncIndex": ("pinecone.async_client.async_index", "AsyncIndex"),
    "AsyncPinecone": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "BackupList": ("pinecone.models.backups.list", "BackupList"),
    "BackupModel": ("pinecone.models.backups.model", "BackupModel"),
    "ByocSpec": ("pinecone.models.indexes.specs", "ByocSpec"),
    "ByocSpecInfo": ("pinecone.models.indexes.index", "ByocSpecInfo"),
    "CollectionList": ("pinecone.models.collections.list", "CollectionList"),
    "CollectionModel": ("pinecone.models.collections.model", "CollectionModel"),
    "CreateIndexFromBackupResponse": (
        "pinecone.models.backups.model",
        "CreateIndexFromBackupResponse",
    ),
    "DescribeIndexStatsResponse": (
        "pinecone.models.vectors.responses",
        "DescribeIndexStatsResponse",
    ),
    "EmbedConfig": ("pinecone.models.indexes.specs", "EmbedConfig"),
    "EmbedModel": ("pinecone.models.enums", "EmbedModel"),
    "EmbeddingsList": ("pinecone.models.inference.embed", "EmbeddingsList"),
    "FetchByMetadataResponse": (
        "pinecone.models.vectors.responses",
        "FetchByMetadataResponse",
    ),
    "FetchResponse": ("pinecone.models.vectors.responses", "FetchResponse"),
    "Field": ("pinecone.utils.filter_builder", "Field"),
    "GrpcIndex": ("pinecone.grpc", "GrpcIndex"),
    "PineconeFuture": ("pinecone.grpc.future", "PineconeFuture"),
    "ImportList": ("pinecone.models.imports.list", "ImportList"),
    "ImportModel": ("pinecone.models.imports.model", "ImportModel"),
    "Index": ("pinecone.index", "Index"),
    "IndexList": ("pinecone.models.indexes.list", "IndexList"),
    "IndexModel": ("pinecone.models.indexes.index", "IndexModel"),
    "IndexSpec": ("pinecone.models.indexes.index", "IndexSpec"),
    "IntegratedSpec": ("pinecone.models.indexes.specs", "IntegratedSpec"),
    "ListNamespacesResponse": (
        "pinecone.models.namespaces.models",
        "ListNamespacesResponse",
    ),
    "ListResponse": ("pinecone.models.vectors.responses", "ListResponse"),
    "ModelInfo": ("pinecone.models.inference.models", "ModelInfo"),
    "ModelInfoList": ("pinecone.models.inference.model_list", "ModelInfoList"),
    "NamespaceDescription": (
        "pinecone.models.namespaces.models",
        "NamespaceDescription",
    ),
    "Pinecone": ("pinecone._client", "Pinecone"),
    "PodSpec": ("pinecone.models.indexes.specs", "PodSpec"),
    "PodSpecInfo": ("pinecone.models.indexes.index", "PodSpecInfo"),
    "QueryNamespacesResults": (
        "pinecone.models.vectors.query_aggregator",
        "QueryNamespacesResults",
    ),
    "QueryResponse": ("pinecone.models.vectors.responses", "QueryResponse"),
    "QueryResultsAggregator": (
        "pinecone.models.vectors.query_aggregator",
        "QueryResultsAggregator",
    ),
    "RerankModel": ("pinecone.models.enums", "RerankModel"),
    "RerankResult": ("pinecone.models.inference.rerank", "RerankResult"),
    "ResponseInfo": ("pinecone.models.vectors.responses", "ResponseInfo"),
    "RestoreJobList": ("pinecone.models.backups.list", "RestoreJobList"),
    "RestoreJobModel": ("pinecone.models.backups.model", "RestoreJobModel"),
    "SearchRecordsResponse": (
        "pinecone.models.vectors.search",
        "SearchRecordsResponse",
    ),
    "ServerlessSpec": ("pinecone.models.indexes.specs", "ServerlessSpec"),
    "ServerlessSpecInfo": ("pinecone.models.indexes.index", "ServerlessSpecInfo"),
    "SparseValues": ("pinecone.models.vectors.sparse", "SparseValues"),
    "StartImportResponse": (
        "pinecone.models.imports.model",
        "StartImportResponse",
    ),
    "UpdateResponse": ("pinecone.models.vectors.responses", "UpdateResponse"),
    "UpsertRecordsResponse": (
        "pinecone.models.vectors.responses",
        "UpsertRecordsResponse",
    ),
    "UpsertResponse": ("pinecone.models.vectors.responses", "UpsertResponse"),
    "Vector": ("pinecone.models.vectors.vector", "Vector"),
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
