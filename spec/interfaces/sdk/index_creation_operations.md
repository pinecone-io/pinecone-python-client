# Index Creation Operations

Documents index creation methods on the Pinecone and PineconeAsyncio clients: `create_index()` for general purpose indexes with serverless, pod, or BYOC configurations, and `create_index_for_model()` for serverless indexes with integrated inference.

## Overview

**Language / runtime:** Python 3.8+
**Package:** `pinecone`
**Module:** `pinecone`
**Classes:** `Pinecone` and `PineconeAsyncio`
**Version:** v3.0.0+
**Breaking change definition:** Removing a method, changing a method's return type, or changing a method signature in a backward-incompatible way (e.g., making a previously optional parameter required without a deprecation period).

## Methods

### `Pinecone.create_index(name: str, spec: Dict | ServerlessSpec | PodSpec | ByocSpec, dimension: int | None = None, metric: Metric | str = "cosine", timeout: int | None = None, deletion_protection: DeletionProtection | str = "disabled", vector_type: VectorType | str = "dense", tags: dict[str, str] | None = None) -> IndexModel`

Creates a Pinecone index with the specified configuration.

**Source:** `pinecone/pinecone.py:378-508`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a new index. The index is immediately available for configuration but may take time to be ready for data operations.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| name | string (1–45 chars) | Yes | — | v1.0 | No | The name of the index to create. Must be unique within your project and cannot be changed once created. Names must contain only lowercase letters, numbers, and hyphens, and cannot start or end with a hyphen (validated by the API). |
| spec | Dict \| ServerlessSpec \| PodSpec \| ByocSpec | Yes | — | v1.0 | No | Configuration describing how the index should be deployed. For serverless, provide region and cloud; optionally specify `read_capacity` (OnDemand or Dedicated) and `schema` for filterable metadata fields. For pod, specify replicas, shards, pods, pod_type, metadata_config, and source_collection. |
| dimension | integer (int32, 1–20000) | No (Required for `vector_type="dense"`) | — | v1.0 | No | The dimensionality of vectors in the index. Must match the embeddings you will insert. Examples: 1536 for OpenAI's text-embedding-3-small, 768 for open-source models. Required when `vector_type="dense"`. Omit when `vector_type="sparse"`. |
| metric | string (enum: `cosine`, `dotproduct`, `euclidean`) | No | `"cosine"` | v1.0 | No | The similarity metric used when querying vectors. Affects which queries are most efficient and how similarity scores are computed. |
| timeout | integer (int32) \| None | No | None | v1.0 | No | The number of seconds to wait for the index to reach ready state. When `None`, wait indefinitely. When `>= 0`, time out after this many seconds and raise `TimeoutError`. When `-1`, return immediately without waiting. |
| deletion_protection | string (enum: `enabled`, `disabled`) \| DeletionProtection | No | `"disabled"` | v1.0 | No | If `"enabled"`, the index cannot be deleted. If `"disabled"`, the index can be deleted. This setting can later be changed with `configure_index()`. |
| vector_type | string (enum: `dense`, `sparse`) \| VectorType | No | `"dense"` | v2.0 | No | The type of vectors stored in the index. `"dense"` for fixed-dimension embeddings, `"sparse"` for variable-length sparse vectors (requires `dimension` to be omitted). |
| tags | dict[str, str] \| None | No | None | v1.0 | No | Key-value pairs to organize and identify the index. Example use cases: tag with model name (`model: text-embedding-3-small`), creation date (`created: 2024-03-15`), or purpose (`env: production`). |

**Returns:** `IndexModel` — A description of the newly created index, containing fields like `name`, `dimension`, `metric`, `host`, `status`, `spec`, and `deletion_protection`.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `dimension` is provided with `vector_type="sparse"`. |
| `PineconeApiException` (400 Bad Request) | Input validation failed. May occur if `name` exceeds 45 characters, contains invalid characters, or starts/ends with hyphens; if `dimension` is outside the valid range; if `metric` is invalid; or if other parameters are malformed. |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission to create indexes. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |

**Example**

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

**Notes**

- The index name is case-insensitive for uniqueness checks, but the returned `IndexModel` preserves the case you provided.
- When `timeout=None`, the method waits indefinitely. For production code, consider setting an explicit timeout to avoid unbounded waits.
- The index is marked as "Initializing" in the response and transitions to "Ready" state asynchronously. Use `describe_index()` to poll the status.
- When using metadata schema, only fields listed in the schema with `filterable: True` can be used in filter expressions during queries.

---

### `PineconeAsyncio.create_index(name: str, spec: Dict | ServerlessSpec | PodSpec | ByocSpec, dimension: int | None = None, metric: Metric | str = "cosine", timeout: int | None = None, deletion_protection: DeletionProtection | str = "disabled", vector_type: VectorType | str = "dense", tags: dict[str, str] | None = None) -> Awaitable[IndexModel]`

Asynchronous version of `Pinecone.create_index()`. Creates a Pinecone index with the specified configuration.

**Source:** `pinecone/pinecone_asyncio.py:423-563`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a new index. The index is immediately available for configuration but may take time to be ready for data operations.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| name | string (1–45 chars) | Yes | — | v1.0 | No | The name of the index to create. Must be unique within your project and cannot be changed once created. Names must contain only lowercase letters, numbers, and hyphens, and cannot start or end with a hyphen (validated by the API). |
| spec | Dict \| ServerlessSpec \| PodSpec \| ByocSpec | Yes | — | v1.0 | No | Configuration describing how the index should be deployed. For serverless, provide region and cloud; optionally specify `read_capacity` (OnDemand or Dedicated) and `schema` for filterable metadata fields. For pod, specify replicas, shards, pods, pod_type, metadata_config, and source_collection. |
| dimension | integer (int32, 1–20000) | No (Required for `vector_type="dense"`) | — | v1.0 | No | The dimensionality of vectors in the index. Must match the embeddings you will insert. Required when `vector_type="dense"`. Omit when `vector_type="sparse"`. |
| metric | string (enum: `cosine`, `dotproduct`, `euclidean`) | No | `"cosine"` | v1.0 | No | The similarity metric used when querying vectors. Affects which queries are most efficient and how similarity scores are computed. |
| timeout | integer (int32) \| None | No | None | v1.0 | No | The number of seconds to wait for the index to reach ready state. When `None`, wait indefinitely. When `>= 0`, time out after this many seconds and raise `TimeoutError`. When `-1`, return immediately without waiting. |
| deletion_protection | string (enum: `enabled`, `disabled`) \| DeletionProtection | No | `"disabled"` | v1.0 | No | If `"enabled"`, the index cannot be deleted. If `"disabled"`, the index can be deleted. This setting can later be changed with `configure_index()`. |
| vector_type | string (enum: `dense`, `sparse`) \| VectorType | No | `"dense"` | v2.0 | No | The type of vectors stored in the index. `"dense"` for fixed-dimension embeddings, `"sparse"` for variable-length sparse vectors. |
| tags | dict[str, str] \| None | No | None | v1.0 | No | Key-value pairs to organize and identify the index. |

**Returns:** `Awaitable[IndexModel]` — Awaitable that resolves to a description of the newly created index.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `dimension` is provided with `vector_type="sparse"`. |
| `PineconeApiException` (400 Bad Request) | Input validation failed (invalid name, dimension, metric, or other parameters). |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission to create indexes. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |

**Example**

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

**Notes**

- The index name is case-insensitive for uniqueness checks, but the returned `IndexModel` preserves the case you provided.
- When `timeout=None`, the method waits indefinitely. For production code, consider setting an explicit timeout to avoid unbounded waits.
- The index is marked as "Initializing" in the response and transitions to "Ready" state asynchronously. Use `describe_index()` to poll the status.
- When using metadata schema, only fields listed in the schema with `filterable: True` can be used in filter expressions during queries.

---

### `Pinecone.create_index_for_model(name: str, cloud: CloudProvider | str, region: AwsRegion | GcpRegion | AzureRegion | str, embed: IndexEmbed | CreateIndexForModelEmbedTypedDict, tags: dict[str, str] | None = None, deletion_protection: DeletionProtection | str = "disabled", read_capacity: ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | None = None, schema: dict[str, MetadataSchemaFieldConfig] | dict[str, dict[str, Any]] | MetadataSchema | None = None, timeout: int | None = None) -> IndexModel`

Creates a serverless index optimized for use with Pinecone's integrated inference models. The index is automatically configured with an embedding model that transforms input data before indexing and querying.

**Source:** `pinecone/pinecone.py:510-655`
**Added:** v1.5
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a serverless index with integrated inference enabled. The specified embedding model becomes the default for upsert and query operations on this index.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| name | string (1–45 chars) | Yes | — | v1.5 | No | The name of the index to create. Must be unique within your project. |
| cloud | CloudProvider \| string (enum: `aws`, `gcp`, `azure`) | Yes | — | v1.5 | No | The cloud provider for the serverless index. Use `CloudProvider.AWS`, `CloudProvider.GCP`, or `CloudProvider.AZURE`, or pass the string value directly. |
| region | AwsRegion \| GcpRegion \| AzureRegion \| string | Yes | — | v1.5 | No | The cloud region for the index. Enum classes `AwsRegion`, `GcpRegion`, and `AzureRegion` provide region constants. Alternatively, pass region names as strings (e.g., `"us-east-1"` for AWS). |
| embed | IndexEmbed \| CreateIndexForModelEmbedTypedDict | Yes | — | v1.5 | No | The embedding configuration. Specify `model` (e.g., `EmbedModel.Multilingual_E5_Large`), `field_map` to map input field names to the fields your embedding model expects (required), and optionally `metric` (e.g., `Metric.COSINE`). |
| tags | dict[str, str] \| None | No | None | v1.5 | No | Key-value pairs to organize and identify the index (e.g., `{"model": "e5-large", "app": "search"}`). |
| deletion_protection | string (enum: `enabled`, `disabled`) \| DeletionProtection | No | `"disabled"` | v1.5 | No | If `"enabled"`, the index cannot be deleted. If `"disabled"`, the index can be deleted. Can be changed later with `configure_index()`. |
| read_capacity | ReadCapacityDict \| ReadCapacity \| ReadCapacityOnDemandSpec \| ReadCapacityDedicatedSpec \| None | No | None (on-demand) | v1.5 | No | Optional read capacity configuration. Omit for on-demand scaling, or provide a dictionary/object with mode (`"OnDemand"` or `"Dedicated"`) and associated settings (node_type, scaling mode, shards, replicas). |
| schema | dict[str, MetadataSchemaFieldConfig] \| dict[str, dict[str, Any]] \| MetadataSchema \| None | No | None | v1.5 | No | Optional metadata schema defining which fields are filterable during queries. Provide as a dictionary mapping field names to their configuration (e.g., `{"genre": {"filterable": True}}`), optionally with a `"fields"` wrapper. |
| timeout | integer (int32) \| None | No | None | v1.5 | No | The number of seconds to wait for the index to reach ready state. When `None`, wait indefinitely. When `>= 0`, time out after this many seconds. When `-1`, return immediately. |

**Returns:** `IndexModel` — A description of the newly created index, including name, dimension (inferred from the model), metric, host, status, and spec.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `field_map` is not provided in the `embed` argument. |
| `PineconeApiException` (400 Bad Request) | Input validation failed. May occur if `cloud` is invalid, `region` is not valid for the specified cloud, `embed.model` is not recognized, or other parameters are malformed. |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission to create indexes or access the specified embedding model. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |

**Example**

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

**Example with dedicated read capacity and metadata schema**

```python
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

**Notes**

- The resulting index is always a serverless index. Pod and BYOC indexes are not supported with integrated inference.
- The embedding model's dimension is automatically set and cannot be overridden.
- The `field_map` is required and must map input field names to the fields your embedding model expects.
- After creation, interact with the index using `Pinecone.Index()` to get an index client for upsert, query, and delete operations.
- For a list of available embedding models, call `pc.inference.list_models()` or visit the [Model Gallery](https://docs.pinecone.io/models/overview).

---

### `PineconeAsyncio.create_index_for_model(name: str, cloud: CloudProvider | str, region: AwsRegion | GcpRegion | AzureRegion | str, embed: IndexEmbed | CreateIndexForModelEmbedTypedDict, tags: dict[str, str] | None = None, deletion_protection: DeletionProtection | str = "disabled", read_capacity: ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | None = None, schema: dict[str, MetadataSchemaFieldConfig] | dict[str, dict[str, Any]] | MetadataSchema | None = None, timeout: int | None = None) -> Awaitable[IndexModel]`

Asynchronous version of `Pinecone.create_index_for_model()`. Creates a serverless index optimized for use with Pinecone's integrated inference models.

**Source:** `pinecone/pinecone_asyncio.py:565-720`
**Added:** v1.5
**Deprecated:** No
**Idempotency:** Non-idempotent. Calling with the same parameters when an index with that name already exists raises a `PineconeApiException` with HTTP status 409 (Conflict).
**Side effects:** Creates a serverless index with integrated inference enabled.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| name | string (1–45 chars) | Yes | — | v1.5 | No | The name of the index to create. Must be unique within your project. |
| cloud | CloudProvider \| string (enum: `aws`, `gcp`, `azure`) | Yes | — | v1.5 | No | The cloud provider for the serverless index. |
| region | AwsRegion \| GcpRegion \| AzureRegion \| string | Yes | — | v1.5 | No | The cloud region for the index. Use enum classes or pass region names as strings. |
| embed | IndexEmbed \| CreateIndexForModelEmbedTypedDict | Yes | — | v1.5 | No | The embedding configuration. Specify `model` (required), `field_map` to map input field names to the fields your embedding model expects (required), and optionally `metric`. |
| tags | dict[str, str] \| None | No | None | v1.5 | No | Key-value pairs to organize and identify the index. |
| deletion_protection | string (enum: `enabled`, `disabled`) \| DeletionProtection | No | `"disabled"` | v1.5 | No | If `"enabled"`, the index cannot be deleted. |
| read_capacity | ReadCapacityDict \| ReadCapacity \| ReadCapacityOnDemandSpec \| ReadCapacityDedicatedSpec \| None | No | None (on-demand) | v1.5 | No | Optional read capacity configuration (OnDemand or Dedicated with node settings). |
| schema | dict[str, MetadataSchemaFieldConfig] \| dict[str, dict[str, Any]] \| MetadataSchema \| None | No | None | v1.5 | No | Optional metadata schema defining which fields are filterable. |
| timeout | integer (int32) \| None | No | None | v1.5 | No | The number of seconds to wait for the index to reach ready state. |

**Returns:** `Awaitable[IndexModel]` — Awaitable that resolves to a description of the newly created index.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` (409 Conflict) | An index with the given `name` already exists. |
| `ValueError` | `field_map` is not provided in the `embed` argument. |
| `PineconeApiException` (400 Bad Request) | Input validation failed (invalid cloud, region, model, or other parameters). |
| `PineconeApiException` (401 Unauthorized) | The API key is missing or invalid. |
| `PineconeApiException` (403 Forbidden) | The API key lacks permission. |
| `TimeoutError` | The index did not reach ready state within the specified `timeout`. |

**Example**

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

**Notes**

- The resulting index is always a serverless index. Pod and BYOC indexes are not supported with integrated inference.
- The embedding model's dimension is automatically set and cannot be overridden.
- The `field_map` is required and must map input field names to the fields your embedding model expects.
- After creation, interact with the index using `PineconeAsyncio.Index()` to get an index client for upsert, query, and delete operations.
- For a list of available embedding models, call `pc.inference.list_models()` or visit the [Model Gallery](https://docs.pinecone.io/models/overview).
