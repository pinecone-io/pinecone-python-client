# Complete List of Importable Objects from the Pinecone Python Client

This document provides a comprehensive list of every object that can be imported from the `pinecone` package. This is useful for creating replacement packages or understanding the public API surface.

## Import from Root Package (`import pinecone` or `from pinecone import ...`)

### Version
- `__version__` (str) - Package version string

### Primary Client Classes
- `Pinecone` - Main synchronous client class
- `PineconeAsyncio` - Async client class  
- `Admin` - Admin API client class

### Deprecated Top-Level Functions
These functions raise `AttributeError` with migration messages when called:
- `init` - Deprecated, use `Pinecone()` instead
- `create_index` - Deprecated, use `pc.create_index()` instead
- `delete_index` - Deprecated, use `pc.delete_index()` instead
- `list_indexes` - Deprecated, use `pc.list_indexes()` instead
- `describe_index` - Deprecated, use `pc.describe_index()` instead
- `configure_index` - Deprecated, use `pc.configure_index()` instead
- `scale_index` - Deprecated, use `pc.configure_index()` instead
- `create_collection` - Deprecated, use `pc.create_collection()` instead
- `delete_collection` - Deprecated, use `pc.delete_collection()` instead
- `describe_collection` - Deprecated, use `pc.describe_collection()` instead
- `list_collections` - Deprecated, use `pc.list_collections()` instead

### Configuration Classes
- `Config` - Configuration class
- `ConfigBuilder` - Configuration builder class
- `PineconeConfig` - Main configuration class

### Inference Classes and Models
- `Inference` - Synchronous inference client (accessible via lazy import)
- `AsyncioInference` - Async inference client (accessible via lazy import)
- `RerankModel` - Enum/class for rerank models
- `EmbedModel` - Enum/class for embedding models
- `ModelInfo` - Model information class
- `ModelInfoList` - List of model information
- `EmbeddingsList` - List of embeddings
- `RerankResult` - Rerank result class

### Data Plane Classes (db_data)
#### Dataclasses
- `Vector` - Vector dataclass
- `SparseValues` - Sparse values dataclass
- `SearchQuery` - Search query dataclass
- `SearchQueryVector` - Search query vector dataclass
- `SearchRerank` - Search rerank dataclass
- `FetchResponse` - Fetch response dataclass

#### Models (from OpenAPI generated code)
- `DeleteRequest` - Delete request model
- `DescribeIndexStatsRequest` - Describe index stats request model
- `DescribeIndexStatsResponse` - Describe index stats response (alias for `IndexDescription`)
- `RpcStatus` - RPC status model
- `ScoredVector` - Scored vector model
- `SingleQueryResults` - Single query results model
- `QueryRequest` - Query request model
- `QueryResponse` - Query response model
- `UpsertResponse` - Upsert response model
- `UpdateRequest` - Update request model
- `NamespaceDescription` - Namespace description model
- `ImportErrorMode` - Import error mode enum/class

### Control Plane Classes (db_control)
#### Enums
- `CloudProvider` - Cloud provider enum
- `AwsRegion` - AWS region enum
- `GcpRegion` - GCP region enum
- `AzureRegion` - Azure region enum
- `PodIndexEnvironment` - Pod index environment enum
- `Metric` - Metric enum (cosine, euclidean, dotproduct)
- `VectorType` - Vector type enum
- `DeletionProtection` - Deletion protection enum
- `PodType` - Pod type enum

#### Models
- `CollectionDescription` - Collection description model
- `CollectionList` - Collection list model
- `IndexList` - Index list model
- `IndexModel` - Index model
- `IndexEmbed` - Index embed configuration
- `ByocSpec` - Bring Your Own Cloud spec
- `ServerlessSpec` - Serverless index spec
- `ServerlessSpecDefinition` - Serverless spec definition
- `PodSpec` - Pod index spec
- `PodSpecDefinition` - Pod spec definition
- `BackupModel` - Backup model
- `BackupList` - Backup list model
- `RestoreJobModel` - Restore job model
- `RestoreJobList` - Restore job list model

#### Types (TypedDict)
- `ConfigureIndexEmbed` - Configure index embed TypedDict
- `CreateIndexForModelEmbedTypedDict` - Create index for model embed TypedDict

### Exception Classes
- `PineconeException` - Base exception class
- `PineconeApiException` - Base API exception
- `PineconeConfigurationError` - Configuration error
- `PineconeProtocolError` - Protocol error
- `PineconeApiAttributeError` - API attribute error
- `PineconeApiTypeError` - API type error
- `PineconeApiValueError` - API value error
- `PineconeApiKeyError` - API key error
- `NotFoundException` - Not found exception (404)
- `UnauthorizedException` - Unauthorized exception (401)
- `ForbiddenException` - Forbidden exception (403)
- `ServiceException` - Service exception (5xx)
- `ListConversionException` - List conversion exception

#### Data Plane Error Classes
- `VectorDictionaryMissingKeysError` - Vector dictionary missing keys error
- `VectorDictionaryExcessKeysError` - Vector dictionary excess keys error
- `VectorTupleLengthError` - Vector tuple length error
- `SparseValuesTypeError` - Sparse values type error
- `SparseValuesMissingKeysError` - Sparse values missing keys error
- `SparseValuesDictionaryExpectedError` - Sparse values dictionary expected error

## Additional Objects Importable via Submodules

### From `pinecone.db_data`
These can be imported directly from the submodule:
- `Index` - Index client class (synchronous)
- `IndexAsyncio` - Index client class (async)
- `_Index` - Alias for `Index` (backwards compatibility)
- `_IndexAsyncio` - Alias for `IndexAsyncio` (backwards compatibility)
- `QueryResponse` - Query response class
- `MetadataDictionaryExpectedError` - Metadata dictionary expected error

### From `pinecone.db_data.types`
These TypedDict classes are not exported from root but can be imported from submodule:
- `SparseVectorTypedDict`
- `VectorTypedDict`
- `VectorMetadataTypedDict`
- `VectorTuple`
- `VectorTupleWithMetadata`
- `FilterTypedDict`
- `SearchRerankTypedDict`
- `SearchQueryTypedDict`
- `SearchQueryVectorTypedDict`

### From `pinecone.db_control`
These can be imported directly from the submodule:
- `DBControl` - Control plane client (synchronous)
- `DBControlAsyncio` - Control plane client (async)

### From `pinecone.inference`
These can be imported directly from the submodule (and are also lazily loaded in root):
- `Inference` - Synchronous inference client
- `AsyncioInference` - Async inference client

### From `pinecone.config`
- `OpenApiConfiguration` - OpenAPI configuration class (not in root __all__ but available)

### From `pinecone.openapi_support`
These are internal support classes but may be importable:
- `ApiClient`
- `AsyncioApiClient`
- `Endpoint`
- `AsyncioEndpoint`
- `Configuration`
- `OpenApiModel`
- Various other OpenAPI support utilities

### From `pinecone.grpc`
These GRPC-related classes can be imported from the submodule:
- `GRPCIndex`
- `PineconeGRPC`
- `GRPCDeleteResponse`
- `GRPCClientConfig`
- `GRPCVector`
- `GRPCSparseValues`
- `PineconeGrpcFuture`
- `ListNamespacesResponse`

### From `pinecone.utils`
Various utility functions and classes (internal but importable):
- `PluginAware`
- `normalize_host`
- `docslinks`
- `require_kwargs`
- `get_user_agent`
- `warn_deprecated`
- `fix_tuple_length`
- `convert_to_list`
- `setup_openapi_client`
- `setup_async_openapi_client`
- `build_plugin_setup_client`
- `parse_non_empty_args`
- `install_json_repr_override`
- `validate_and_convert_errors`
- `convert_enum_to_string`
- `filter_dict`
- `tqdm` (from `pinecone.utils.tqdm`)

## Summary Count

### From Root Package (`from pinecone import ...`)
- **Total items in `__all__`**: ~85-90 items (including lazy imports)

### Categories:
- **Primary clients**: 3 (Pinecone, PineconeAsyncio, Admin)
- **Config classes**: 3 (Config, ConfigBuilder, PineconeConfig)
- **Deprecated functions**: 11 (all raise errors)
- **Inference**: 8 (Inference, AsyncioInference, RerankModel, EmbedModel, ModelInfo, ModelInfoList, EmbeddingsList, RerankResult)
- **Data plane classes**: ~20 (dataclasses, models, errors)
- **Control plane classes**: ~20 (enums, models, types)
- **Exceptions**: 14
- **Version**: 1 (__version__)

### Additional via Submodules
- **Data types (TypedDict)**: 9
- **GRPC classes**: 8
- **Utility functions**: ~15

## Notes

1. **Lazy Imports**: Many classes are loaded lazily via the `_setup_lazy_imports` mechanism in `__init__.py`. They appear in `__all__` but are only imported when first accessed.

2. **Deprecated Functions**: The deprecated top-level functions (`init`, `create_index`, etc.) are in `__all__` but raise `AttributeError` with helpful migration messages when called. They exist to provide better error messages for users migrating from older versions.

3. **Backwards Compatibility**: Some aliases exist for backwards compatibility (e.g., `_Index` for `Index`).

4. **Type Stubs**: The `__init__.pyi` file provides type hints and should match the actual exports.

5. **Submodule Imports**: While many objects can be imported directly from root, they can also be imported from their specific submodules (e.g., `from pinecone.db_data import Index`).

6. **Internal vs Public**: Some objects in submodules like `pinecone.openapi_support` and `pinecone.utils` are technically importable but are considered internal implementation details. Only items in the root `__all__` or documented in the official docs should be considered part of the public API.

