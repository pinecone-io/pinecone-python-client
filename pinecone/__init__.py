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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pinecone._client import Pinecone
    from pinecone.admin import Admin
    from pinecone.async_client.async_index import AsyncIndex
    from pinecone.async_client.pinecone import AsyncPinecone
    from pinecone.grpc import GrpcIndex
    from pinecone.grpc.future import PineconeFuture
    from pinecone.index import Index
    from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
    from pinecone.models.assistant.context import ContextResponse
    from pinecone.models.assistant.file_model import AssistantFileModel
    from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
    from pinecone.models.assistant.message import Message
    from pinecone.models.assistant.model import AssistantModel
    from pinecone.models.assistant.options import ContextOptions
    from pinecone.models.backups.list import BackupList, RestoreJobList
    from pinecone.models.backups.model import (
        BackupModel,
        CreateIndexFromBackupResponse,
        RestoreJobModel,
    )
    from pinecone.models.collections.list import CollectionList
    from pinecone.models.collections.model import CollectionModel
    from pinecone.models.enums import EmbedModel, RerankModel
    from pinecone.models.imports.list import ImportList
    from pinecone.models.imports.model import ImportModel, StartImportResponse
    from pinecone.models.indexes.index import (
        ByocSpecInfo,
        IndexModel,
        IndexSpec,
        PodSpecInfo,
        ServerlessSpecInfo,
    )
    from pinecone.models.indexes.list import IndexList
    from pinecone.models.indexes.specs import (
        ByocSpec,
        EmbedConfig,
        IntegratedSpec,
        PodSpec,
        ServerlessSpec,
    )
    from pinecone.models.inference.embed import EmbeddingsList
    from pinecone.models.inference.model_list import ModelInfoList
    from pinecone.models.inference.models import ModelInfo
    from pinecone.models.inference.rerank import RerankResult
    from pinecone.models.namespaces.models import (
        ListNamespacesResponse,
        NamespaceDescription,
    )
    from pinecone.models.vectors.query_aggregator import (
        QueryNamespacesResults,
        QueryResultsAggregator,
    )
    from pinecone.models.vectors.responses import (
        DescribeIndexStatsResponse,
        FetchByMetadataResponse,
        FetchResponse,
        ListResponse,
        QueryResponse,
        ResponseInfo,
        UpdateResponse,
        UpsertRecordsResponse,
        UpsertResponse,
    )
    from pinecone.models.vectors.search import SearchRecordsResponse
    from pinecone.models.vectors.sparse import SparseValues
    from pinecone.models.vectors.vector import Vector
    from pinecone.utils.filter_builder import Field

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
    "AssistantFileModel",
    "AssistantModel",
    "AsyncIndex",
    "AsyncPinecone",
    "BackupList",
    "BackupModel",
    "ByocSpec",
    "ByocSpecInfo",
    "ChatCompletionResponse",
    "ChatResponse",
    "CloudProvider",
    "CollectionList",
    "CollectionModel",
    "ConflictError",
    "ContextOptions",
    "ContextResponse",
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
    "ListAssistantsResponse",
    "ListFilesResponse",
    "ListResponse",
    "Message",
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
    "AssistantFileModel": ("pinecone.models.assistant.file_model", "AssistantFileModel"),
    "AssistantModel": ("pinecone.models.assistant.model", "AssistantModel"),
    "AsyncIndex": ("pinecone.async_client.async_index", "AsyncIndex"),
    "AsyncPinecone": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "BackupList": ("pinecone.models.backups.list", "BackupList"),
    "BackupModel": ("pinecone.models.backups.model", "BackupModel"),
    "ByocSpec": ("pinecone.models.indexes.specs", "ByocSpec"),
    "ByocSpecInfo": ("pinecone.models.indexes.index", "ByocSpecInfo"),
    "ChatCompletionResponse": ("pinecone.models.assistant.chat", "ChatCompletionResponse"),
    "ChatResponse": ("pinecone.models.assistant.chat", "ChatResponse"),
    "CollectionList": ("pinecone.models.collections.list", "CollectionList"),
    "CollectionModel": ("pinecone.models.collections.model", "CollectionModel"),
    "ContextOptions": ("pinecone.models.assistant.options", "ContextOptions"),
    "ContextResponse": ("pinecone.models.assistant.context", "ContextResponse"),
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
    "ListAssistantsResponse": (
        "pinecone.models.assistant.list",
        "ListAssistantsResponse",
    ),
    "ListFilesResponse": ("pinecone.models.assistant.list", "ListFilesResponse"),
    "ListResponse": ("pinecone.models.vectors.responses", "ListResponse"),
    "Message": ("pinecone.models.assistant.message", "Message"),
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
