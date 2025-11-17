from pinecone.config import Config
from pinecone.config import ConfigBuilder
from pinecone.config import PineconeConfig
from pinecone.exceptions import (
    PineconeException,
    PineconeApiTypeError,
    PineconeApiValueError,
    PineconeApiAttributeError,
    PineconeApiKeyError,
    PineconeApiException,
    NotFoundException,
    UnauthorizedException,
    ForbiddenException,
    ServiceException,
    PineconeProtocolError,
    PineconeConfigurationError,
    ListConversionException,
)
from pinecone.inference import (
    Inference,
    AsyncioInference,
    RerankModel,
    EmbedModel,
    ModelInfo,
    ModelInfoList,
    EmbeddingsList,
    RerankResult,
)
from pinecone.db_data.dataclasses import (
    Vector,
    SparseValues,
    SearchQuery,
    SearchQueryVector,
    SearchRerank,
    FetchResponse,
    FetchByMetadataResponse,
    QueryResponse,
    UpsertResponse,
    UpdateResponse,
)
from pinecone.db_data.models import (
    DeleteRequest,
    DescribeIndexStatsRequest,
    IndexDescription as DescribeIndexStatsResponse,
    RpcStatus,
    ScoredVector,
    SingleQueryResults,
    QueryRequest,
    UpdateRequest,
)
from pinecone.core.openapi.db_data.models import NamespaceDescription
from pinecone.db_data.resources.sync.bulk_import import ImportErrorMode
from pinecone.db_data.errors import (
    VectorDictionaryMissingKeysError,
    VectorDictionaryExcessKeysError,
    VectorTupleLengthError,
    SparseValuesTypeError,
    SparseValuesMissingKeysError,
    SparseValuesDictionaryExpectedError,
)
from pinecone.db_control.enums import (
    CloudProvider,
    AwsRegion,
    GcpRegion,
    AzureRegion,
    PodIndexEnvironment,
    Metric,
    VectorType,
    DeletionProtection,
    PodType,
)
from pinecone.db_control.models import (
    CollectionDescription,
    CollectionList,
    IndexList,
    IndexModel,
    IndexEmbed,
    ServerlessSpec,
    ServerlessSpecDefinition,
    PodSpec,
    PodSpecDefinition,
    ByocSpec,
    BackupModel,
    BackupList,
    RestoreJobModel,
    RestoreJobList,
)
from pinecone.db_control.models.serverless_spec import (
    ScalingConfigManualDict,
    ReadCapacityDedicatedConfigDict,
    ReadCapacityOnDemandDict,
    ReadCapacityDedicatedDict,
    ReadCapacityDict,
    MetadataSchemaFieldConfig,
)
from pinecone.db_data.filter_builder import FilterBuilder
from pinecone.db_control.types import ConfigureIndexEmbed, CreateIndexForModelEmbedTypedDict
from pinecone.pinecone import Pinecone
from pinecone.pinecone_asyncio import PineconeAsyncio
from pinecone.admin import Admin
from pinecone.utils import __version__

# Deprecated top-level functions
def init(*args: object, **kwargs: object) -> None: ...
def create_index(*args: object, **kwargs: object) -> None: ...
def delete_index(*args: object, **kwargs: object) -> None: ...
def list_indexes(*args: object, **kwargs: object) -> None: ...
def describe_index(*args: object, **kwargs: object) -> None: ...
def configure_index(*args: object, **kwargs: object) -> None: ...
def scale_index(*args: object, **kwargs: object) -> None: ...
def create_collection(*args: object, **kwargs: object) -> None: ...
def delete_collection(*args: object, **kwargs: object) -> None: ...
def describe_collection(*args: object, **kwargs: object) -> None: ...
def list_collections(*args: object, **kwargs: object) -> None: ...

# Re-export all the types
__all__ = [
    "__version__",
    # Deprecated top-level functions
    "init",
    "create_index",
    "delete_index",
    "list_indexes",
    "describe_index",
    "configure_index",
    "scale_index",
    "create_collection",
    "delete_collection",
    "describe_collection",
    "list_collections",
    # Primary client classes
    "Pinecone",
    "PineconeAsyncio",
    "Admin",
    # Config classes
    "Config",
    "ConfigBuilder",
    "PineconeConfig",
    # Exceptions
    "PineconeException",
    "PineconeApiTypeError",
    "PineconeApiValueError",
    "PineconeApiAttributeError",
    "PineconeApiKeyError",
    "PineconeApiException",
    "NotFoundException",
    "UnauthorizedException",
    "ForbiddenException",
    "ServiceException",
    "PineconeProtocolError",
    "PineconeConfigurationError",
    "ListConversionException",
    # Inference classes
    "Inference",
    "AsyncioInference",
    "RerankModel",
    "EmbedModel",
    "ModelInfo",
    "ModelInfoList",
    "EmbeddingsList",
    "RerankResult",
    # Data classes
    "Vector",
    "SparseValues",
    "SearchQuery",
    "SearchQueryVector",
    "SearchRerank",
    # Data response classes
    "FetchResponse",
    "FetchByMetadataResponse",
    "QueryResponse",
    "UpsertResponse",
    "UpdateResponse",
    # Model classes
    "DeleteRequest",
    "DescribeIndexStatsRequest",
    "DescribeIndexStatsResponse",
    "RpcStatus",
    "ScoredVector",
    "SingleQueryResults",
    "QueryRequest",
    "UpdateRequest",
    "NamespaceDescription",
    "ImportErrorMode",
    "FilterBuilder",
    # Error classes
    "VectorDictionaryMissingKeysError",
    "VectorDictionaryExcessKeysError",
    "VectorTupleLengthError",
    "SparseValuesTypeError",
    "SparseValuesMissingKeysError",
    "SparseValuesDictionaryExpectedError",
    # Control plane enums
    "CloudProvider",
    "AwsRegion",
    "GcpRegion",
    "AzureRegion",
    "PodIndexEnvironment",
    "Metric",
    "VectorType",
    "DeletionProtection",
    "PodType",
    # Control plane models
    "CollectionDescription",
    "CollectionList",
    "IndexList",
    "IndexModel",
    "IndexEmbed",
    "ServerlessSpec",
    "ServerlessSpecDefinition",
    "PodSpec",
    "PodSpecDefinition",
    "ByocSpec",
    "BackupModel",
    "BackupList",
    "RestoreJobModel",
    "RestoreJobList",
    # Control plane types
    "ConfigureIndexEmbed",
    "CreateIndexForModelEmbedTypedDict",
    "ScalingConfigManualDict",
    "ReadCapacityDedicatedConfigDict",
    "ReadCapacityOnDemandDict",
    "ReadCapacityDedicatedDict",
    "ReadCapacityDict",
    "MetadataSchemaFieldConfig",
]
