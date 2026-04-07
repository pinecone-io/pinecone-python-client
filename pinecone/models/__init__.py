"""msgspec.Struct models for the Pinecone SDK."""

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
from pinecone.models.indexes.index import IndexModel, IndexStatus
from pinecone.models.indexes.list import IndexList
from pinecone.models.indexes.specs import (
    ByocSpec,
    EmbedConfig,
    IntegratedSpec,
    PodSpec,
    ServerlessSpec,
)
from pinecone.models.inference import (
    DenseEmbedding,
    Embedding,
    EmbeddingsList,
    EmbedUsage,
    ModelInfo,
    ModelInfoList,
    ModelInfoSupportedParameter,
    RankedDocument,
    RerankResult,
    RerankUsage,
    SparseEmbedding,
)
from pinecone.models.namespaces.models import ListNamespacesResponse, NamespaceDescription
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.responses import (
    DescribeIndexStatsResponse,
    FetchByMetadataResponse,
    FetchResponse,
    ListItem,
    ListResponse,
    NamespaceSummary,
    Pagination,
    QueryResponse,
    ResponseInfo,
    UpdateResponse,
    UpsertRecordsResponse,
    UpsertResponse,
)
from pinecone.models.vectors.search import SearchRecordsResponse
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.usage import Usage
from pinecone.models.vectors.vector import ScoredVector, Vector

__all__ = [
    "BackupList",
    "BackupModel",
    "ByocSpec",
    "CloudProvider",
    "CollectionList",
    "CollectionModel",
    "CreateIndexFromBackupResponse",
    "DeletionProtection",
    "DenseEmbedding",
    "DescribeIndexStatsResponse",
    "Embedding",
    "EmbedConfig",
    "EmbeddingsList",
    "EmbedModel",
    "EmbedUsage",
    "FetchByMetadataResponse",
    "FetchResponse",
    "ImportList",
    "ImportModel",
    "IndexList",
    "IndexModel",
    "IndexStatus",
    "IntegratedSpec",
    "ListItem",
    "ListNamespacesResponse",
    "ListResponse",
    "Metric",
    "ModelInfo",
    "ModelInfoList",
    "ModelInfoSupportedParameter",
    "NamespaceDescription",
    "NamespaceSummary",
    "Pagination",
    "PodSpec",
    "PodType",
    "QueryNamespacesResults",
    "QueryResponse",
    "RankedDocument",
    "RerankModel",
    "RerankResult",
    "RerankUsage",
    "ResponseInfo",
    "RestoreJobList",
    "RestoreJobModel",
    "ScoredVector",
    "SearchRecordsResponse",
    "ServerlessSpec",
    "SparseEmbedding",
    "SparseValues",
    "StartImportResponse",
    "UpdateResponse",
    "UpsertRecordsResponse",
    "UpsertResponse",
    "Usage",
    "Vector",
    "VectorType",
]
