# Index Creation and Configuration

Operations for creating and configuring Pinecone indexes. The `create_index()` method supports serverless, pod, and BYOC configurations, while `create_index_for_model()` creates serverless indexes with integrated inference. The `configure_index()` method enables modification of pod count, pod type, deletion protection, tags, integrated inference settings, and serverless read capacity.

---

## `Pinecone.create_index()`

Creates a Pinecone index with the specified configuration.

**Source:** `pinecone/pinecone.py:378-508`

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a new index. The index is immediately available for configuration but may take time to be ready for data operations.

### Signature

```python
def create_index(
    self,
    name: str,
    spec: Dict | ServerlessSpec | PodSpec | ByocSpec,
    dimension: int | None = None,
    metric: (Metric | str) | None = "cosine",
    timeout: int | None = None,
    deletion_protection: (DeletionProtection | str) | None = "disabled",
    vector_type: (VectorType | str) | None = "dense",
    tags: dict[str, str] | None = None,
) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string (1–45 chars)` | Yes | — | v1.0 | No | The name of the index to create. Must be unique within your project and cannot be changed once created. Names must contain only lowercase letters, numbers, and hyphens, and cannot start or end with a hyphen (validated by the API). |
| `spec` | `Dict \| ServerlessSpec \| PodSpec \| ByocSpec` | Yes | — | v1.0 | No | Configuration describing how the index should be deployed. For serverless, provide region and cloud; optionally specify `read_capacity` (OnDemand or Dedicated) and `schema` for filterable metadata fields. For pod, specify replicas, shards, pods, pod_type, metadata_config, and source_collection. |
| `dimension` | `integer (int32, 1–20000)` | No (Required for `vector_type="dense"`) | — | v1.0 | No | The dimensionality of vectors in the index. Must match the embeddings you will insert. Examples: 1536 for OpenAI's text-embedding-3-small, 768 for open-source models. Required when `vector_type="dense"`. Omit when `vector_type="sparse"`. |
| `metric` | `(Metric \| string) \| None` | No | `"cosine"` | v1.0 | No | The similarity metric used when querying vectors. Affects which queries are most efficient and how similarity scores are computed. |
| `timeout` | `integer (int32) \| None` | No | `None` | v1.0 | No | The number of seconds to wait for the index to reach ready state. When `None`, wait indefinitely. When `>= 0`, time out after this many seconds and raise `TimeoutError`. When `-1`, return immediately without waiting. |
| `deletion_protection` | `(DeletionProtection \| string) \| None` | No | `"disabled"` | v1.0 | No | If `"enabled"`, the index cannot be deleted. If `"disabled"`, the index can be deleted. This setting can later be changed with `configure_index()`. |
| `vector_type` | `(VectorType \| string) \| None` | No | `"dense"` | v2.0 | No | The type of vectors stored in the index. `"dense"` for fixed-dimension embeddings, `"sparse"` for variable-length sparse vectors (requires `dimension` to be omitted). |
| `tags` | `dict[str, str] \| None` | No | `None` | v1.0 | No | Key-value pairs to organize and identify the index. Example use cases: tag with model name (`model: text-embedding-3-small`), creation date (`created: 2024-03-15`), or purpose (`env: production`). |

### Returns

**Type:** `IndexModel` — A description of the newly created index, containing fields like `name`, `dimension`, `metric`, `host`, `status`, `spec`, and `deletion_protection`.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `dimension` is provided with `vector_type="sparse"`. |
| `PineconeApiException` (400 Bad Request) | Input validation failed. May occur if `name` exceeds 45 characters, contains invalid characters, or starts/ends with hyphens; if `dimension` is outside the valid range; if `metric` is invalid; or if other parameters are malformed. |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission to create indexes. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |
| `Exception` | Index initialization failed with status `InitializationFailed`. |

### Behavior

- The index is marked as "Initializing" in the response and transitions to "Ready" state asynchronously. Use `describe_index()` to poll the status.
- By default (`timeout=None`), the method polls `describe_index()` every 5 seconds until the index status is `ready`. This is blocking.
- When `timeout=-1`, returns the API response without polling. When `timeout` is a positive integer, polls for that many seconds then raises `TimeoutError`.
- The index name is case-insensitive for uniqueness checks, but the returned `IndexModel` preserves the case you provided.
- When using metadata schema, only fields listed in the schema with `filterable: True` can be used in filter expressions during queries.
- Caches the index endpoint address locally for use by subsequent operations.

### Example

```python
import os
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    DeletionProtection,
    VectorType
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create a serverless index with dedicated read capacity and metadata schema
index_description = pc.create_index(
    name="movie-index",
    dimension=512,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_WEST_2,
        read_capacity={
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {"shards": 2, "replicas": 2},
            },
        },
        schema={
            "genre": {"filterable": True},
            "year": {"filterable": True},
            "rating": {"filterable": True},
        },
    ),
    deletion_protection=DeletionProtection.DISABLED,
    vector_type=VectorType.DENSE,
    tags={
        "app": "movie-recommendations",
        "env": "production"
    }
)

print(f"Index '{index_description.name}' created successfully")
print(f"Status: {index_description.status}")
print(f"Host: {index_description.host}")
```

### Notes

- When `timeout=None`, the method waits indefinitely. For production code, consider setting an explicit timeout to avoid unbounded waits.
- Repeated identical calls with the same name raise an error (name conflict). Use `has_index()` if you need idempotent behavior.

---

## `PineconeAsyncio.create_index()`

Asynchronous version of `Pinecone.create_index()`. Creates a Pinecone index with the specified configuration.

**Source:** `pinecone/pinecone_asyncio.py:423-563`

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a new index. The index is immediately available for configuration but may take time to be ready for data operations.

### Signature

```python
async def create_index(
    self,
    name: str,
    spec: Dict | ServerlessSpec | PodSpec | ByocSpec,
    dimension: int | None = None,
    metric: (Metric | str) | None = "cosine",
    timeout: int | None = None,
    deletion_protection: (DeletionProtection | str) | None = "disabled",
    vector_type: (VectorType | str) | None = "dense",
    tags: dict[str, str] | None = None,
) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string (1–45 chars)` | Yes | — | v1.0 | No | The name of the index to create. Must be unique within your project and cannot be changed once created. Names must contain only lowercase letters, numbers, and hyphens, and cannot start or end with a hyphen (validated by the API). |
| `spec` | `Dict \| ServerlessSpec \| PodSpec \| ByocSpec` | Yes | — | v1.0 | No | Configuration describing how the index should be deployed. For serverless, provide region and cloud; optionally specify `read_capacity` (OnDemand or Dedicated) and `schema` for filterable metadata fields. For pod, specify replicas, shards, pods, pod_type, metadata_config, and source_collection. |
| `dimension` | `integer (int32, 1–20000)` | No (Required for `vector_type="dense"`) | — | v1.0 | No | The dimensionality of vectors in the index. Must match the embeddings you will insert. Required when `vector_type="dense"`. Omit when `vector_type="sparse"`. |
| `metric` | `(Metric \| string) \| None` | No | `"cosine"` | v1.0 | No | The similarity metric used when querying vectors. Affects which queries are most efficient and how similarity scores are computed. |
| `timeout` | `integer (int32) \| None` | No | `None` | v1.0 | No | The number of seconds to wait for the index to reach ready state. When `None`, wait indefinitely. When `>= 0`, time out after this many seconds and raise `TimeoutError`. When `-1`, return immediately without waiting. |
| `deletion_protection` | `(DeletionProtection \| string) \| None` | No | `"disabled"` | v1.0 | No | If `"enabled"`, the index cannot be deleted. If `"disabled"`, the index can be deleted. This setting can later be changed with `configure_index()`. |
| `vector_type` | `(VectorType \| string) \| None` | No | `"dense"` | v2.0 | No | The type of vectors stored in the index. `"dense"` for fixed-dimension embeddings, `"sparse"` for variable-length sparse vectors. |
| `tags` | `dict[str, str] \| None` | No | `None` | v1.0 | No | Key-value pairs to organize and identify the index. |

### Returns

**Type:** `IndexModel` — Awaitable that resolves to a description of the newly created index.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `dimension` is provided with `vector_type="sparse"`. |
| `PineconeApiException` (400 Bad Request) | Input validation failed (invalid name, dimension, metric, or other parameters). |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission to create indexes. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |

### Example

```python
import os
import asyncio
from pinecone import (
    PineconeAsyncio,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
)

async def main():
    async with PineconeAsyncio(api_key=os.environ.get("PINECONE_API_KEY")) as pc:
        # Create a serverless index
        index_description = await pc.create_index(
            name="async-movie-index",
            dimension=768,
            metric=Metric.COSINE,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
            ),
            tags={"app": "async-movies"}
        )

        print(f"Index created: {index_description.name}")

asyncio.run(main())
```

### Notes

- The index name is case-insensitive for uniqueness checks, but the returned `IndexModel` preserves the case you provided.
- When `timeout=None`, the method waits indefinitely. For production code, consider setting an explicit timeout to avoid unbounded waits.
- The index is marked as "Initializing" in the response and transitions to "Ready" state asynchronously. Use `describe_index()` to poll the status.
- When using metadata schema, only fields listed in the schema with `filterable: True` can be used in filter expressions during queries.
- The async version uses Enum defaults instead of string defaults: `metric=Metric.COSINE`, `deletion_protection=DeletionProtection.DISABLED`, `vector_type=VectorType.DENSE`. Both formats (strings and Enum values) are accepted by the API.

---

## `Pinecone.create_index_for_model()`

Creates a serverless index optimized for use with Pinecone's integrated inference models. The index is automatically configured with an embedding model that transforms input data before indexing and querying.

**Source:** `pinecone/pinecone.py:510-655`

**Added:** v1.5
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a serverless index with integrated inference enabled. The specified embedding model becomes the default for upsert and query operations on this index.

### Signature

```python
def create_index_for_model(
    self,
    name: str,
    cloud: CloudProvider | str,
    region: AwsRegion | GcpRegion | AzureRegion | str,
    embed: IndexEmbed | CreateIndexForModelEmbedTypedDict,
    tags: dict[str, str] | None = None,
    deletion_protection: (DeletionProtection | str) | None = "disabled",
    read_capacity: ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | None = None,
    schema: dict[str, MetadataSchemaFieldConfig] | dict[str, dict[str, Any]] | MetadataSchema | None = None,
    timeout: int | None = None,
) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string (1–45 chars)` | Yes | — | v1.5 | No | The name of the index to create. Must be unique within your project. |
| `cloud` | `CloudProvider \| string (enum: aws, gcp, azure)` | Yes | — | v1.5 | No | The cloud provider for the serverless index. Use `CloudProvider.AWS`, `CloudProvider.GCP`, or `CloudProvider.AZURE`, or pass the string value directly. |
| `region` | `AwsRegion \| GcpRegion \| AzureRegion \| string` | Yes | — | v1.5 | No | The cloud region for the index. Enum classes `AwsRegion`, `GcpRegion`, and `AzureRegion` provide region constants. Alternatively, pass region names as strings (e.g., `"us-east-1"` for AWS). |
| `embed` | `IndexEmbed \| CreateIndexForModelEmbedTypedDict` | Yes | — | v1.5 | No | The embedding configuration. Specify `model` (e.g., `EmbedModel.Multilingual_E5_Large`), `field_map` to map input field names to the fields your embedding model expects (required), and optionally `metric` (e.g., `Metric.COSINE`). |
| `tags` | `dict[str, str] \| None` | No | `None` | v1.5 | No | Key-value pairs to organize and identify the index (e.g., `{"model": "e5-large", "app": "search"}`). |
| `deletion_protection` | `(DeletionProtection \| string) \| None` | No | `"disabled"` | v1.5 | No | If `"enabled"`, the index cannot be deleted. If `"disabled"`, the index can be deleted. Can be changed later with `configure_index()`. |
| `read_capacity` | `ReadCapacityDict \| ReadCapacity \| ReadCapacityOnDemandSpec \| ReadCapacityDedicatedSpec \| None` | No | `None` (on-demand) | v1.5 | No | Optional read capacity configuration. Omit for on-demand scaling, or provide a dictionary/object with mode (`"OnDemand"` or `"Dedicated"`) and associated settings (node_type, scaling mode, shards, replicas). |
| `schema` | `dict[str, MetadataSchemaFieldConfig] \| dict[str, dict[str, Any]] \| MetadataSchema \| None` | No | `None` | v1.5 | No | Optional metadata schema defining which fields are filterable during queries. Provide as a dictionary mapping field names to their configuration (e.g., `{"genre": {"filterable": True}}`), optionally with a `"fields"` wrapper. |
| `timeout` | `integer (int32) \| None` | No | `None` | v1.5 | No | The number of seconds to wait for the index to reach ready state. When `None`, wait indefinitely. When `>= 0`, time out after this many seconds. When `-1`, return immediately. |

### Returns

**Type:** `IndexModel` — A description of the newly created index, including name, dimension (inferred from the model), metric, host, status, and spec.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `field_map` is not provided in the `embed` argument. |
| `PineconeApiException` (400 Bad Request) | Input validation failed. May occur if `cloud` is invalid, `region` is not valid for the specified cloud, `embed.model` is not recognized, or other parameters are malformed. |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission to create indexes or access the specified embedding model. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |

### Example

```python
from pinecone import (
    Pinecone,
    IndexEmbed,
    EmbedModel,
    CloudProvider,
    AwsRegion,
    Metric,
    DeletionProtection,
)

pc = Pinecone()

# Create a serverless index with integrated inference (multilingual E5 embeddings)
if not pc.has_index("book-search"):
    index_description = pc.create_index_for_model(
        name="book-search",
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1,
        embed=IndexEmbed(
            model=EmbedModel.Multilingual_E5_Large,
            metric=Metric.COSINE,
            field_map={
                "text": "description",  # Map input field 'description' to 'text' for the model
            },
        ),
        tags={
            "model": "e5-large",
            "app": "book-search",
            "env": "production"
        }
    )

    print(f"Index '{index_description.name}' created with inference enabled")
    print(f"Host: {index_description.host}")
```

```python
# Example with dedicated read capacity and metadata schema
from pinecone import (
    Pinecone,
    IndexEmbed,
    EmbedModel,
    CloudProvider,
    AwsRegion,
    Metric,
)

pc = Pinecone()

index_description = pc.create_index_for_model(
    name="product-search",
    cloud=CloudProvider.AWS,
    region=AwsRegion.US_EAST_1,
    embed=IndexEmbed(
        model=EmbedModel.Multilingual_E5_Large,
        metric=Metric.COSINE,
        field_map={
            "text": "description",
        },
    ),
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 2, "replicas": 2},
        },
    },
    schema={
        "product_category": {"filterable": True},
        "brand": {"filterable": True},
        "price": {"filterable": True},
    },
)

print(f"Index created with dedicated read capacity")
```

### Notes

- The resulting index is always a serverless index. Pod and BYOC indexes are not supported with integrated inference.
- The embedding model's dimension is automatically set and cannot be overridden.
- The `field_map` is required and must map input field names to the fields your embedding model expects.
- After creation, interact with the index using `Pinecone.Index()` to get an index client for upsert, query, and delete operations.
- For a list of available embedding models, call `pc.inference.list_models()` or visit the [Model Gallery](https://docs.pinecone.io/models/overview).

---

## `PineconeAsyncio.create_index_for_model()`

Asynchronous version of `Pinecone.create_index_for_model()`. Creates a serverless index optimized for use with Pinecone's integrated inference models.

**Source:** `pinecone/pinecone_asyncio.py:565-720`

**Added:** v1.5
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a serverless index with integrated inference enabled.

### Signature

```python
async def create_index_for_model(
    self,
    name: str,
    cloud: CloudProvider | str,
    region: AwsRegion | GcpRegion | AzureRegion | str,
    embed: IndexEmbed | CreateIndexForModelEmbedTypedDict,
    tags: dict[str, str] | None = None,
    deletion_protection: (DeletionProtection | str) | None = "disabled",
    read_capacity: ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | None = None,
    schema: dict[str, MetadataSchemaFieldConfig] | dict[str, dict[str, Any]] | MetadataSchema | None = None,
    timeout: int | None = None,
) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string (1–45 chars)` | Yes | — | v1.5 | No | The name of the index to create. Must be unique within your project. |
| `cloud` | `CloudProvider \| string (enum: aws, gcp, azure)` | Yes | — | v1.5 | No | The cloud provider for the serverless index. |
| `region` | `AwsRegion \| GcpRegion \| AzureRegion \| string` | Yes | — | v1.5 | No | The cloud region for the index. Use enum classes or pass region names as strings. |
| `embed` | `IndexEmbed \| CreateIndexForModelEmbedTypedDict` | Yes | — | v1.5 | No | The embedding configuration. Specify `model` (required), `field_map` to map input field names to the fields your embedding model expects (required), and optionally `metric`. |
| `tags` | `dict[str, str] \| None` | No | `None` | v1.5 | No | Key-value pairs to organize and identify the index. |
| `deletion_protection` | `(DeletionProtection \| string) \| None` | No | `"disabled"` | v1.5 | No | If `"enabled"`, the index cannot be deleted. |
| `read_capacity` | `ReadCapacityDict \| ReadCapacity \| ReadCapacityOnDemandSpec \| ReadCapacityDedicatedSpec \| None` | No | `None` (on-demand) | v1.5 | No | Optional read capacity configuration (OnDemand or Dedicated with node settings). |
| `schema` | `dict[str, MetadataSchemaFieldConfig] \| dict[str, dict[str, Any]] \| MetadataSchema \| None` | No | `None` | v1.5 | No | Optional metadata schema defining which fields are filterable. |
| `timeout` | `integer (int32) \| None` | No | `None` | v1.5 | No | The number of seconds to wait for the index to reach ready state. |

### Returns

**Type:** `IndexModel` — Awaitable that resolves to a description of the newly created index.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `field_map` is not provided in the `embed` argument. |
| `PineconeApiException` (400 Bad Request) | Input validation failed (invalid cloud, region, model, or other parameters). |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |

### Example

```python
import asyncio
from pinecone import (
    PineconeAsyncio,
    IndexEmbed,
    EmbedModel,
    CloudProvider,
    AwsRegion,
    Metric,
)

async def main():
    async with PineconeAsyncio() as pc:
        # Create a serverless index with integrated inference
        index_description = await pc.create_index_for_model(
            name="async-book-search",
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1,
            embed=IndexEmbed(
                model=EmbedModel.Multilingual_E5_Large,
                metric=Metric.COSINE,
                field_map={"text": "description"},
            ),
            tags={"app": "async-search"}
        )

        print(f"Async index created: {index_description.name}")

asyncio.run(main())
```

### Notes

- The resulting index is always a serverless index. Pod and BYOC indexes are not supported with integrated inference.
- The embedding model's dimension is automatically set and cannot be overridden.
- The `field_map` is required and must map input field names to the fields your embedding model expects.
- After creation, interact with the index using `PineconeAsyncio.Index()` to get an index client for upsert, query, and delete operations.
- For a list of available embedding models, call `pc.inference.list_models()` or visit the [Model Gallery](https://docs.pinecone.io/models/overview).

---

## `Pinecone.configure_index()`

Modifies the configuration of an existing index without waiting for the operation to complete.

**Source:** `pinecone/pinecone.py:870-1026`

**Added:** v3.0
**Deprecated:** No
**Idempotency:** Safe to retry. Repeated calls with identical parameters produce the same final result. Each call is sent to the server; the method does not skip requests if the configuration is already in the desired state.
**Side effects:** Modifies the index configuration. Changes are applied asynchronously; use `describe_index()` to monitor status.

### Signature

```python
def configure_index(
    self,
    name: str,
    replicas: int | None = None,
    pod_type: (PodType | str) | None = None,
    deletion_protection: (DeletionProtection | str) | None = None,
    tags: dict[str, str] | None = None,
    embed: (ConfigureIndexEmbed | Dict) | None = None,
    read_capacity: ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | None = None,
) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v3.0 | No | The name of the index to configure. |
| `replicas` | `integer (int32, 0–)` | No | `None` | v3.0 | No | The desired number of replicas for pod-based indexes. When omitted, replicas are not modified. Only applies to pod-based indexes. |
| `pod_type` | `string (enum: p1.x1, p1.x2, p1.x4, p1.x8, p2.x1, p2.x2, p2.x4, p2.x8, s1.x1, s1.x2, s1.x4, s1.x8) \| PodType` | No | `None` | v3.0 | No | The new pod type for pod-based indexes. When omitted, pod type is not modified. Only applies to pod-based indexes. Valid values depend on the pod-based index environment. |
| `deletion_protection` | `string (enum: enabled, disabled) \| DeletionProtection` | No | `None` | v3.0 | No | Whether the index is protected from deletion. When set to `"enabled"` or `DeletionProtection.ENABLED`, the index cannot be deleted via `delete_index()`. When set to `"disabled"` or `DeletionProtection.DISABLED`, deletion protection is removed. When omitted, deletion protection status is not modified. |
| `tags` | `dict[str, str]` | No | `None` | v3.0 | No | Tags to add, update, or remove from the index. Tag updates are merged with existing tags; to remove a tag, set its value to an empty string `""`. When omitted, tags are not modified. |
| `embed` | `ConfigureIndexEmbed \| dict` | No | `None` | v3.0 | No | Enables or updates integrated inference embeddings on the index, specifying the embedding model and configuring field mapping and read/write parameters. Once set, the embedding model cannot be changed. Only applies to serverless indexes. When omitted, embedding configuration is not modified. |
| `read_capacity` | `ReadCapacityDict \| ReadCapacity \| ReadCapacityOnDemandSpec \| ReadCapacityDedicatedSpec \| None` | No | `None` | v3.0 | No | Read capacity configuration for serverless indexes, specifying whether to use on-demand or dedicated mode. When omitted, read capacity configuration is not modified. Only applies to serverless indexes. See examples for detailed structure. |

### Returns

**Type:** `None` — Configuration changes are processed asynchronously. The method returns immediately after the server accepts the request, not after changes are applied.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | No index with the given `name` exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `ValueError` | `deletion_protection` has an invalid value, or `read_capacity` configuration is invalid (e.g., missing required fields in dedicated mode). |
| `PineconeApiException` | The index modification fails on the server side (e.g., incompatible configuration changes). |

### Behavior

- Configuration changes are processed asynchronously. After calling `configure_index()`, call `describe_index()` to check the current status; the status field will show `"initializing"` or `"ready"` as changes are being applied.
- Multiple parameters can be modified in a single call; all requested changes are batched together.
- Tag merging: The `tags` parameter merges with existing tags rather than replacing them. To remove a tag, set its value to an empty string.
- Pod type changes (vertical scaling) may incur a brief period where the index is temporarily unavailable.
- For serverless indexes, `replicas` and `pod_type` are not applicable; use `read_capacity` instead.
- For pod-based indexes, `read_capacity` and `embed` are not applicable.
- Queries and mutations are not interrupted during configuration changes. Scaling happens in a rolling manner with zero downtime.

### Example

```python
from pinecone import Pinecone, DeletionProtection, PodType

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Scale a pod-based index by adding replicas
pc.configure_index(
    name="my-pod-index",
    replicas=3
)

# Change pod type (vertical scaling)
pc.configure_index(
    name="my-pod-index",
    pod_type=PodType.P1_X2
)

# Enable deletion protection
pc.configure_index(
    name="my-pod-index",
    deletion_protection=DeletionProtection.ENABLED
)

# Disable deletion protection
pc.configure_index(
    name="my-pod-index",
    deletion_protection=DeletionProtection.DISABLED
)

# Add or update tags
pc.configure_index(
    name="my-pod-index",
    tags={"environment": "production", "team": "search"}
)

# Remove a tag by setting its value to empty string
pc.configure_index(
    name="my-pod-index",
    tags={"old-tag": ""}
)

# Configure integrated inference embeddings on a serverless index
pc.configure_index(
    name="my-serverless-index",
    embed={"model": "multilingual-e5-large"}
)

# Configure serverless read capacity to on-demand mode
pc.configure_index(
    name="my-serverless-index",
    read_capacity={"mode": "OnDemand"}
)

# Configure serverless read capacity to dedicated mode with manual scaling
pc.configure_index(
    name="my-serverless-index",
    read_capacity={
        "mode": "Dedicated",
        "dedicated": {
            "node_type": "t1",
            "scaling": "Manual",
            "manual": {"shards": 1, "replicas": 1}
        }
    }
)

# Verify configuration was accepted (changes apply asynchronously)
desc = pc.describe_index("my-pod-index")
print(f"Replicas: {desc.spec.pod.replicas}")  # May not yet reflect new value
print(f"Deletion protection: {desc.deletion_protection}")
```

---

## `PineconeAsyncio.configure_index()`

Asynchronous version of `Pinecone.configure_index()`. Modifies the configuration of an existing index without waiting for the operation to complete.

**Source:** `pinecone/pinecone_asyncio.py:941-1108`

**Added:** v3.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** Modifies the index configuration. Changes are applied asynchronously; use `describe_index()` to monitor status.

### Signature

```python
async def configure_index(
    self,
    name: str,
    replicas: int | None = None,
    pod_type: (PodType | str) | None = None,
    deletion_protection: (DeletionProtection | str) | None = None,
    tags: dict[str, str] | None = None,
    embed: (ConfigureIndexEmbed | Dict) | None = None,
    read_capacity: ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | None = None,
) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v3.0 | No | The name of the index to configure. |
| `replicas` | `integer (int32, 0–)` | No | `None` | v3.0 | No | The desired number of replicas for pod-based indexes. When omitted, replicas are not modified. Only applies to pod-based indexes. |
| `pod_type` | `string (enum: p1.x1, p1.x2, p1.x4, p1.x8, p2.x1, p2.x2, p2.x4, p2.x8, s1.x1, s1.x2, s1.x4, s1.x8) \| PodType` | No | `None` | v3.0 | No | The new pod type for pod-based indexes. When omitted, pod type is not modified. Only applies to pod-based indexes. Valid values depend on the pod-based index environment. |
| `deletion_protection` | `string (enum: enabled, disabled) \| DeletionProtection` | No | `None` | v3.0 | No | Whether the index is protected from deletion. When set to `"enabled"` or `DeletionProtection.ENABLED`, the index cannot be deleted via `delete_index()`. When set to `"disabled"` or `DeletionProtection.DISABLED`, deletion protection is removed. When omitted, deletion protection status is not modified. |
| `tags` | `dict[str, str]` | No | `None` | v3.0 | No | Tags to add, update, or remove from the index. Tag updates are merged with existing tags; to remove a tag, set its value to an empty string `""`. When omitted, tags are not modified. |
| `embed` | `ConfigureIndexEmbed \| dict` | No | `None` | v3.0 | No | Enables or updates integrated inference embeddings on the index, specifying the embedding model and configuring field mapping and read/write parameters. Once set, the embedding model cannot be changed. Only applies to serverless indexes. When omitted, embedding configuration is not modified. |
| `read_capacity` | `ReadCapacityDict \| ReadCapacity \| ReadCapacityOnDemandSpec \| ReadCapacityDedicatedSpec \| None` | No | `None` | v3.0 | No | Read capacity configuration for serverless indexes, specifying whether to use on-demand or dedicated mode. When omitted, read capacity configuration is not modified. Only applies to serverless indexes. See examples for detailed structure. |

### Returns

**Type:** `None` — An awaitable that completes when the server accepts the request. Configuration changes are processed asynchronously after the method returns.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | No index with the given `name` exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `ValueError` | `deletion_protection` has an invalid value, or `read_capacity` configuration is invalid (e.g., missing required fields in dedicated mode). |
| `PineconeApiException` | The index modification fails on the server side (e.g., incompatible configuration changes). |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio, DeletionProtection, PodType

async def main():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    # Scale a pod-based index by adding replicas
    await pc.configure_index(
        name="my-pod-index",
        replicas=3
    )

    # Change pod type (vertical scaling)
    await pc.configure_index(
        name="my-pod-index",
        pod_type=PodType.P1_X2
    )

    # Enable deletion protection
    await pc.configure_index(
        name="my-pod-index",
        deletion_protection=DeletionProtection.ENABLED
    )

    # Disable deletion protection
    await pc.configure_index(
        name="my-pod-index",
        deletion_protection=DeletionProtection.DISABLED
    )

    # Add or update tags
    await pc.configure_index(
        name="my-pod-index",
        tags={"environment": "production", "team": "search"}
    )

    # Remove a tag by setting its value to empty string
    await pc.configure_index(
        name="my-pod-index",
        tags={"old-tag": ""}
    )

    # Configure integrated inference embeddings on a serverless index
    await pc.configure_index(
        name="my-serverless-index",
        embed={"model": "multilingual-e5-large"}
    )

    # Configure serverless read capacity to on-demand mode
    await pc.configure_index(
        name="my-serverless-index",
        read_capacity={"mode": "OnDemand"}
    )

    # Configure serverless read capacity to dedicated mode with manual scaling
    await pc.configure_index(
        name="my-serverless-index",
        read_capacity={
            "mode": "Dedicated",
            "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {"shards": 1, "replicas": 1}
            }
        }
    )

    # Verify configuration was accepted
    desc = await pc.describe_index("my-pod-index")
    print(f"Replicas: {desc.spec.pod.replicas}")

asyncio.run(main())
```

### Notes

- Configuration changes are processed asynchronously. After awaiting `configure_index()`, call `describe_index()` to check the current status.
- All notes from the synchronous version apply equally to the asynchronous version.

---

## Enumerations

### Metric

Distance metric for similarity search. Set during index creation; cannot be changed.

| Value | Description |
|-------|-------------|
| `"cosine"` | Cosine similarity (1 - (u · v) / (‖u‖ ‖v‖)). Recommended for most embedding models. |
| `"euclidean"` | Euclidean distance (√(Σ(uᵢ - vᵢ)²)). Useful when absolute magnitude matters. |
| `"dotproduct"` | Dot product (u · v). Useful for magnitude-sensitive embeddings and assumes normalized vectors. |

**Source:** `pinecone/db_control/enums/metric.py:4-11`

### VectorType

Type of vectors stored in the index. Set during index creation; cannot be changed.

| Value | Description |
|-------|-------------|
| `"dense"` | Dense vectors with values in most or all dimensions. Use for most embedding models. |
| `"sparse"` | Sparse vectors with non-zero values in few dimensions. Use for sparse embeddings and BM25 models. |

**Source:** `pinecone/db_control/enums/vector_type.py:4-14`

### DeletionProtection

Whether the index is protected from deletion. Can be enabled or disabled at any time via `configure()`.

| Value | Description |
|-------|-------------|
| `"enabled"` | Calling `delete()` raises an error. The index can only be deleted after calling `configure()` with `deletion_protection="disabled"`. |
| `"disabled"` | Calling `delete()` immediately deletes the index. |

**Source:** `pinecone/db_control/enums/deletion_protection.py:4-15`

### PodType

Pod types for pod-based indexes. Different types provide different memory, throughput, and cost trade-offs.

| Value | Generation | Memory | Description |
|-------|-----------|--------|-------------|
| `"p1.x1"` | P1 (Legacy) | 1GB | Older generation, lower cost. |
| `"p1.x2"` | P1 (Legacy) | 2GB | Older generation, lower cost. |
| `"p1.x4"` | P1 (Legacy) | 4GB | Older generation, lower cost. |
| `"p1.x8"` | P1 (Legacy) | 8GB | Older generation, lower cost. |
| `"s1.x1"` | S1 (Storage) | 1GB | Storage-optimized for large indexes with lower QPS requirements. |
| `"s1.x2"` | S1 (Storage) | 2GB | Storage-optimized for large indexes with lower QPS requirements. |
| `"s1.x4"` | S1 (Storage) | 4GB | Storage-optimized for large indexes with lower QPS requirements. |
| `"s1.x8"` | S1 (Storage) | 8GB | Storage-optimized for large indexes with lower QPS requirements. |
| `"p2.x1"` | P2 (Current) | 1GB | Current generation, recommended for most workloads. |
| `"p2.x2"` | P2 (Current) | 2GB | Current generation, recommended for most workloads. |
| `"p2.x4"` | P2 (Current) | 4GB | Current generation, recommended for most workloads. |
| `"p2.x8"` | P2 (Current) | 8GB | Current generation, recommended for most workloads. |

**Source:** `pinecone/db_control/enums/pod_type.py:4-20`

---

## Data Models

### IndexModel

Represents a created or described index. Proxies to the underlying OpenAPI model.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `name` | `string` | No | The index name. |
| `dimension` | `integer (int32)` | No | The vector dimension. |
| `metric` | `string (enum: cosine, euclidean, dotproduct)` | No | The distance metric. |
| `host` | `string (uri)` | No | The service endpoint for queries and mutations. Include this in client initialization. |
| `status` | `IndexStatus` | No | Current index status (ready, initializing, failed). |
| `spec` | `ServerlessSpec \| PodSpec \| ByocSpec` | No | The infrastructure specification. |
| `tags` | `dict[string, string]` | No | User-defined tags. Empty dict if no tags are set. |
| `vector_type` | `string (enum: dense, sparse)` | No | The vector type. |
| `deletion_protection` | `string (enum: enabled, disabled)` | No | Whether deletion is protected. |

**Source:** `pinecone/db_control/models/index_model.py:22-35`

### IndexStatus

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `state` | `string (enum: Initializing, InitializationFailed, ScalingUp, ScalingDown, ScalingUpPodSize, ScalingDownPodSize, Terminating, Ready, Disabled)` | No | Current lifecycle state. Initializing when the index is being created; Ready when operational; Disabled when the index is removed or inactive. |
| `ready` | `boolean` | No | `true` if state is `Ready`, `false` otherwise. Convenience flag for checking operational status. |

**Source:** `pinecone/core/openapi/db_control/model/index_model_status.py:120-122`
