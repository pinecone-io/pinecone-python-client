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
    index.upsert(vectors=[("movie-42", [0.012, -0.087, 0.153])])  # 1536-dim vector
    results = index.query(vector=[0.012, -0.087, 0.153], top_k=5)  # 1536-dim vector

    # Integrated inference: search with text (server-side embedding)
    index = pc.index("my-integrated-index")
    results = index.search(namespace="default", top_k=5, inputs={"text": "search query"})

The :class:`Pinecone` client manages indexes (control plane). Call
``pc.index(name)`` to get an :class:`Index` for vector operations (data plane).

Async Quick Start::

    from pinecone import AsyncPinecone, ServerlessSpec

    async with AsyncPinecone(api_key="your-api-key") as pc:
        # Control plane: manage indexes
        indexes = await pc.indexes.list()

        # Data plane: resolve host first, then create index client
        desc = await pc.indexes.describe("my-index")
        index = pc.index(host=desc.host)

        async with index:
            results = await index.query(vector=[0.012, -0.087, 0.153], top_k=5)

For async usage, see :class:`AsyncPinecone`. For admin/org management,
see :class:`Admin`.
"""

from __future__ import annotations

import os as _os

# Avoid importing typing at runtime — its transitive deps (re, enum,
# collections, contextlib, functools, warnings) add ~28ms to cold import.
# All annotations use PEP 563 (from __future__ import annotations), so
# typing.Any is a string at runtime and never evaluated.
# mypy recognises a module-level `TYPE_CHECKING = False` as a type-checking
# guard, so the if-block below is analysed by type checkers but skipped at
# runtime.
TYPE_CHECKING = False

if TYPE_CHECKING:
    from typing import Any

    from pinecone._client import Pinecone
    from pinecone._internal.config import PineconeConfig, RetryConfig
    from pinecone.admin import Admin
    from pinecone.async_client.async_index import AsyncIndex
    from pinecone.async_client.pinecone import AsyncPinecone
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
    )
    from pinecone.grpc import GrpcIndex
    from pinecone.grpc.future import PineconeFuture
    from pinecone.index import Index
    from pinecone.models.admin.api_key import APIKeyList, APIKeyModel, APIKeyWithSecret
    from pinecone.models.admin.organization import OrganizationList, OrganizationModel
    from pinecone.models.admin.project import ProjectList, ProjectModel
    from pinecone.models.assistant.chat import ChatCompletionResponse, ChatResponse
    from pinecone.models.assistant.context import ContextResponse
    from pinecone.models.assistant.evaluation import AlignmentResult
    from pinecone.models.assistant.file_model import AssistantFileModel
    from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
    from pinecone.models.assistant.message import Message
    from pinecone.models.assistant.model import AssistantModel
    from pinecone.models.assistant.options import ContextOptions
    from pinecone.models.assistant.streaming import (
        ChatCompletionStreamChunk,
        ChatStreamChunk,
        StreamCitationChunk,
        StreamContentChunk,
        StreamMessageEnd,
        StreamMessageStart,
    )
    from pinecone.models.backups.list import BackupList, RestoreJobList
    from pinecone.models.backups.model import (
        BackupModel,
        CreateIndexFromBackupResponse,
        RestoreJobModel,
    )
    from pinecone.models.collections.list import CollectionList
    from pinecone.models.collections.model import CollectionModel
    from pinecone.models.enums import (
        CloudProvider,
        DeletionProtection,
        EmbedModel,
        Metric,
        PodType,
        RerankModel,
        VectorType,
    )
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
    from pinecone.models.inference.embed import DenseEmbedding, EmbeddingsList, SparseEmbedding
    from pinecone.models.inference.model_list import ModelInfoList
    from pinecone.models.inference.models import ModelInfo
    from pinecone.models.inference.rerank import RankedDocument, RerankResult
    from pinecone.models.namespaces.models import (
        ListNamespacesResponse,
        NamespaceDescription,
    )
    from pinecone.models.pagination import AsyncPaginator, Page, Paginator
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
    from pinecone.models.vectors.search import (
        Hit,
        RerankConfig,
        SearchInputs,
        SearchRecordsResponse,
        SearchResult,
        SearchUsage,
    )
    from pinecone.models.vectors.sparse import SparseValues
    from pinecone.models.vectors.vector import Vector
    from pinecone.utils.filter_builder import Field

__version__ = "9.0.0"

if _os.environ.get("PINECONE_DEBUG"):
    import logging as _logging

    _logging.getLogger("pinecone").setLevel(_logging.DEBUG)

__all__ = [
    "APIKeyList",
    "APIKeyModel",
    "APIKeyWithSecret",
    "Admin",
    "AlignmentResult",
    "ApiError",
    "AssistantFileModel",
    "AssistantModel",
    "AsyncIndex",
    "AsyncPaginator",
    "AsyncPinecone",
    "BackupList",
    "BackupModel",
    "ByocSpec",
    "ByocSpecInfo",
    "ChatCompletionResponse",
    "ChatCompletionStreamChunk",
    "ChatResponse",
    "ChatStreamChunk",
    "CloudProvider",
    "CollectionList",
    "CollectionModel",
    "ConflictError",
    "ContextOptions",
    "ContextResponse",
    "CreateIndexFromBackupResponse",
    "DeletionProtection",
    "DenseEmbedding",
    "DescribeIndexStatsResponse",
    "EmbedConfig",
    "EmbedModel",
    "EmbeddingsList",
    "FetchByMetadataResponse",
    "FetchResponse",
    "Field",
    "ForbiddenError",
    "GrpcIndex",
    "Hit",
    "ImportList",
    "ImportModel",
    "Index",
    "IndexInitFailedError",
    "IndexList",
    "IndexModel",
    "IndexSpec",
    "IntegratedSpec",
    "ListAssistantsResponse",
    "ListFilesResponse",
    "ListNamespacesResponse",
    "ListResponse",
    "Message",
    "Metric",
    "ModelInfo",
    "ModelInfoList",
    "NamespaceDescription",
    "NotFoundError",
    "OrganizationList",
    "OrganizationModel",
    "Page",
    "Paginator",
    "Pinecone",
    "PineconeConfig",
    "PineconeConnectionError",
    "PineconeError",
    "PineconeFuture",
    "PineconeTimeoutError",
    "PineconeTypeError",
    "PineconeValueError",
    "PodSpec",
    "PodSpecInfo",
    "PodType",
    "ProjectList",
    "ProjectModel",
    "QueryNamespacesResults",
    "QueryResponse",
    "QueryResultsAggregator",
    "RankedDocument",
    "RerankConfig",
    "RerankModel",
    "RerankResult",
    "ResponseInfo",
    "ResponseParsingError",
    "RestoreJobList",
    "RestoreJobModel",
    "RetryConfig",
    "SearchInputs",
    "SearchRecordsResponse",
    "SearchResult",
    "SearchUsage",
    "ServerlessSpec",
    "ServerlessSpecInfo",
    "ServiceError",
    "SparseEmbedding",
    "SparseValues",
    "StartImportResponse",
    "StreamCitationChunk",
    "StreamContentChunk",
    "StreamMessageEnd",
    "StreamMessageStart",
    "UnauthorizedError",
    "UpdateResponse",
    "UpsertRecordsResponse",
    "UpsertResponse",
    "Vector",
    "VectorType",
    "__version__",
]

# Lazy-load heavy classes to keep cold import under 10ms.
# Importing Pinecone/AsyncPinecone/Index eagerly pulls in httpx (~120ms).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ApiError": ("pinecone.errors.exceptions", "ApiError"),
    "ConflictError": ("pinecone.errors.exceptions", "ConflictError"),
    "ForbiddenError": ("pinecone.errors.exceptions", "ForbiddenError"),
    "IndexInitFailedError": ("pinecone.errors.exceptions", "IndexInitFailedError"),
    "NotFoundError": ("pinecone.errors.exceptions", "NotFoundError"),
    "PineconeConnectionError": ("pinecone.errors.exceptions", "PineconeConnectionError"),
    "PineconeError": ("pinecone.errors.exceptions", "PineconeError"),
    "PineconeTimeoutError": ("pinecone.errors.exceptions", "PineconeTimeoutError"),
    "PineconeTypeError": ("pinecone.errors.exceptions", "PineconeTypeError"),
    "PineconeValueError": ("pinecone.errors.exceptions", "PineconeValueError"),
    "ResponseParsingError": ("pinecone.errors.exceptions", "ResponseParsingError"),
    "ServiceError": ("pinecone.errors.exceptions", "ServiceError"),
    "UnauthorizedError": ("pinecone.errors.exceptions", "UnauthorizedError"),
    "Admin": ("pinecone.admin", "Admin"),
    "APIKeyList": ("pinecone.models.admin.api_key", "APIKeyList"),
    "APIKeyModel": ("pinecone.models.admin.api_key", "APIKeyModel"),
    "APIKeyWithSecret": ("pinecone.models.admin.api_key", "APIKeyWithSecret"),
    "AlignmentResult": ("pinecone.models.assistant.evaluation", "AlignmentResult"),
    "AsyncPaginator": ("pinecone.models.pagination", "AsyncPaginator"),
    "AssistantFileModel": ("pinecone.models.assistant.file_model", "AssistantFileModel"),
    "AssistantModel": ("pinecone.models.assistant.model", "AssistantModel"),
    "AsyncIndex": ("pinecone.async_client.async_index", "AsyncIndex"),
    "AsyncPinecone": ("pinecone.async_client.pinecone", "AsyncPinecone"),
    "BackupList": ("pinecone.models.backups.list", "BackupList"),
    "BackupModel": ("pinecone.models.backups.model", "BackupModel"),
    "ByocSpec": ("pinecone.models.indexes.specs", "ByocSpec"),
    "ByocSpecInfo": ("pinecone.models.indexes.index", "ByocSpecInfo"),
    "CloudProvider": ("pinecone.models.enums", "CloudProvider"),
    "ChatCompletionResponse": ("pinecone.models.assistant.chat", "ChatCompletionResponse"),
    "ChatCompletionStreamChunk": (
        "pinecone.models.assistant.streaming",
        "ChatCompletionStreamChunk",
    ),
    "ChatResponse": ("pinecone.models.assistant.chat", "ChatResponse"),
    "ChatStreamChunk": ("pinecone.models.assistant.streaming", "ChatStreamChunk"),
    "CollectionList": ("pinecone.models.collections.list", "CollectionList"),
    "CollectionModel": ("pinecone.models.collections.model", "CollectionModel"),
    "ContextOptions": ("pinecone.models.assistant.options", "ContextOptions"),
    "ContextResponse": ("pinecone.models.assistant.context", "ContextResponse"),
    "CreateIndexFromBackupResponse": (
        "pinecone.models.backups.model",
        "CreateIndexFromBackupResponse",
    ),
    "DeletionProtection": ("pinecone.models.enums", "DeletionProtection"),
    "DenseEmbedding": ("pinecone.models.inference.embed", "DenseEmbedding"),
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
    "Hit": ("pinecone.models.vectors.search", "Hit"),
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
    "Metric": ("pinecone.models.enums", "Metric"),
    "ModelInfo": ("pinecone.models.inference.models", "ModelInfo"),
    "ModelInfoList": ("pinecone.models.inference.model_list", "ModelInfoList"),
    "NamespaceDescription": (
        "pinecone.models.namespaces.models",
        "NamespaceDescription",
    ),
    "OrganizationList": ("pinecone.models.admin.organization", "OrganizationList"),
    "OrganizationModel": ("pinecone.models.admin.organization", "OrganizationModel"),
    "Page": ("pinecone.models.pagination", "Page"),
    "Paginator": ("pinecone.models.pagination", "Paginator"),
    "Pinecone": ("pinecone._client", "Pinecone"),
    "PineconeConfig": ("pinecone._internal.config", "PineconeConfig"),
    "PodSpec": ("pinecone.models.indexes.specs", "PodSpec"),
    "PodType": ("pinecone.models.enums", "PodType"),
    "ProjectList": ("pinecone.models.admin.project", "ProjectList"),
    "ProjectModel": ("pinecone.models.admin.project", "ProjectModel"),
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
    "RerankConfig": ("pinecone.models.vectors.search", "RerankConfig"),
    "RankedDocument": ("pinecone.models.inference.rerank", "RankedDocument"),
    "RerankModel": ("pinecone.models.enums", "RerankModel"),
    "RerankResult": ("pinecone.models.inference.rerank", "RerankResult"),
    "ResponseInfo": ("pinecone.models.vectors.responses", "ResponseInfo"),
    "RestoreJobList": ("pinecone.models.backups.list", "RestoreJobList"),
    "RestoreJobModel": ("pinecone.models.backups.model", "RestoreJobModel"),
    "RetryConfig": ("pinecone._internal.config", "RetryConfig"),
    "SearchInputs": ("pinecone.models.vectors.search", "SearchInputs"),
    "SearchRecordsResponse": (
        "pinecone.models.vectors.search",
        "SearchRecordsResponse",
    ),
    "SearchResult": ("pinecone.models.vectors.search", "SearchResult"),
    "SearchUsage": ("pinecone.models.vectors.search", "SearchUsage"),
    "ServerlessSpec": ("pinecone.models.indexes.specs", "ServerlessSpec"),
    "ServerlessSpecInfo": ("pinecone.models.indexes.index", "ServerlessSpecInfo"),
    "SparseEmbedding": ("pinecone.models.inference.embed", "SparseEmbedding"),
    "SparseValues": ("pinecone.models.vectors.sparse", "SparseValues"),
    "StartImportResponse": (
        "pinecone.models.imports.model",
        "StartImportResponse",
    ),
    "StreamCitationChunk": (
        "pinecone.models.assistant.streaming",
        "StreamCitationChunk",
    ),
    "StreamContentChunk": (
        "pinecone.models.assistant.streaming",
        "StreamContentChunk",
    ),
    "StreamMessageEnd": ("pinecone.models.assistant.streaming", "StreamMessageEnd"),
    "StreamMessageStart": ("pinecone.models.assistant.streaming", "StreamMessageStart"),
    "UpdateResponse": ("pinecone.models.vectors.responses", "UpdateResponse"),
    "UpsertRecordsResponse": (
        "pinecone.models.vectors.responses",
        "UpsertRecordsResponse",
    ),
    "UpsertResponse": ("pinecone.models.vectors.responses", "UpsertResponse"),
    "Vector": ("pinecone.models.vectors.vector", "Vector"),
    "VectorType": ("pinecone.models.enums", "VectorType"),
}


def __getattr__(name: str) -> Any:
    if name == "ValidationError":
        import warnings

        warnings.warn(
            "ValidationError is deprecated; use PineconeValueError instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from pinecone.errors.exceptions import ValidationError

        globals()["ValidationError"] = ValidationError
        return ValidationError
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
