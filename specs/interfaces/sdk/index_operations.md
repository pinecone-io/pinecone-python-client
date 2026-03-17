# Index Creation and Configuration

Methods for creating and configuring Pinecone indexes via the `IndexResource` class (available on `Pinecone` and `PineconeAsyncio` client instances as `.indexes`).

---

## IndexResource.create

Creates a new index with the specified configuration.

**Source:** `pinecone/db_control/resources/sync/index.py:74-101`, `pinecone/db_control/resources/asyncio/index.py:58-85` (async equivalent)

**Added:** v1.0
**Deprecated:** No

### Signature

```python
def create(
    self,
    *,
    name: str,
    spec: Dict | ServerlessSpec | PodSpec | ByocSpec,
    dimension: int | None = None,
    metric: Metric | str = "cosine",
    timeout: int | None = None,
    deletion_protection: DeletionProtection | str = "disabled",
    vector_type: VectorType | str = "dense",
    tags: dict[str, str] | None = None,
) -> IndexModel:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index. Must be unique within the workspace and match the pattern `[a-z0-9_-]{1,45}`. |
| `spec` | `dict \| ServerlessSpec \| PodSpec \| ByocSpec` | Yes | — | v1.0 | No | The infrastructure specification for the index. Defines deployment mode (serverless, pod-based, or BYOC), compute resources, and region. Pass a dict or use the provided spec classes. |
| `dimension` | `int \| None` | No | `None` | v1.0 | No | The number of dimensions for vectors in the index. Required when creating serverless indexes unless embedding is configured via the spec. Must be between 1 and 20,000. |
| `metric` | `Metric \| str` | No | `"cosine"` | v1.0 | No | The distance metric used for similarity search. One of `cosine`, `euclidean`, or `dotproduct`. Cannot be changed after index creation. |
| `deletion_protection` | `DeletionProtection \| str` | No | `"disabled"` | v1.0 | No | Whether the index is protected from deletion. One of `enabled` or `disabled`. When enabled, calling `delete()` raises an error. Can be changed later via `configure()`. |
| `vector_type` | `VectorType \| str` | No | `"dense"` | v1.0 | No | The type of vectors to store in the index. One of `dense` or `sparse`. Cannot be changed after index creation. |
| `tags` | `dict[str, str] \| None` | No | `None` | v1.0 | No | Key-value tags to attach to the index for organization and filtering. User-defined; no validation or auto-generation. |
| `timeout` | `int \| None` | No | `None` | v1.0 | No | Seconds to wait for the index to reach `ready` status. `None` polls indefinitely; `-1` returns immediately without polling; positive integer polls for that many seconds then raises `TimeoutError`. |

### Returns

**Type:** `IndexModel`

The created index with metadata: name, dimension, metric, host endpoint, status, spec, tags, vector type, and deletion protection.

### Raises

| Exception | Condition |
|-----------|-----------|
| `pinecone.PineconeApiException` | Index name already exists, or invalid parameters (spec, dimension out of range, invalid metric, unsupported region, etc.). |
| `ValueError` | Invalid parameter values (e.g., deletion_protection not "enabled"/"disabled", spec missing required keys, dimension specified for sparse vectors). |
| `TypeError` | Invalid parameter type (e.g., `spec` is not dict, ServerlessSpec, PodSpec, or ByocSpec). |
| `TimeoutError` | Index was not ready within the specified `timeout` seconds. Message includes elapsed time. Only raised if `timeout` is a positive integer. |
| `Exception` | Index initialization failed with status `InitializationFailed`. |

### Idempotency

Non-idempotent. Repeated identical calls with the same name raise an error (name conflict). Use the index name as a uniqueness key if you need idempotent behavior.

### Side Effects

- Creates a new index in the Pinecone service
- Allocates compute resources according to the `spec`
- Caches the index endpoint address locally for use by subsequent operations
- If `timeout >= 0`, polls `describe()` every 5 seconds until ready or timeout

### Example

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# Create a serverless index
index = pc.indexes.create(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    timeout=120
)

print(f"Index '{index.name}' is ready: {index.status.ready}")
```

### Notable Behavior

- **Polling:** By default, the method polls `describe()` every 5 seconds until the index status is `ready`. This is blocking.
- **Timeout behavior:** When `timeout=-1`, returns the API response without polling. When `timeout=None`, polls indefinitely. When `timeout` is a positive integer, polls for that many seconds then raises `TimeoutError`.
- **Async default types:** The async version (`PineconeAsyncio.indexes.create()`) uses Enum defaults instead of string defaults: `metric=Metric.COSINE`, `deletion_protection=DeletionProtection.DISABLED`, `vector_type=VectorType.DENSE`. Both formats (strings and Enum values) are accepted by the API.

---

## IndexResource.configure

Updates the configuration of an existing index without interrupting running queries.

**Source:** `pinecone/db_control/resources/sync/index.py:259-289`, `pinecone/db_control/resources/asyncio/index.py:220-248` (async equivalent)

**Added:** v1.0
**Deprecated:** No

### Signature

```python
def configure(
    self,
    *,
    name: str,
    replicas: int | None = None,
    pod_type: PodType | str | None = None,
    deletion_protection: DeletionProtection | str | None = None,
    tags: dict[str, str] | None = None,
    embed: ConfigureIndexEmbed | Dict | None = None,
    read_capacity: ReadCapacityDict | ReadCapacity | ReadCapacityOnDemandSpec | ReadCapacityDedicatedSpec | None = None,
) -> None:
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index to configure. |
| `replicas` | `int \| None` | No | `None` | v1.0 | No | Number of replicas for pod-based indexes only. Ignored for serverless indexes. When `None`, the current replica count is preserved. |
| `pod_type` | `PodType \| str \| None` | No | `None` | v1.0 | No | Pod type for pod-based indexes (e.g., `p1.x1`, `p2.x4`). Ignored for serverless indexes. When `None`, the current pod type is preserved. |
| `deletion_protection` | `DeletionProtection \| str \| None` | No | `None` | v1.0 | No | Whether the index is protected from deletion. One of `enabled` or `disabled`. When `None`, the current protection status is preserved. When set, overrides the existing value. |
| `tags` | `dict[str, str] \| None` | No | `None` | v1.0 | No | Key-value tags for the index. When provided, **replaces all existing tags**. When `None`, existing tags are preserved. Pass `{}` to remove all tags. |
| `embed` | `ConfigureIndexEmbed \| dict \| None` | No | `None` | v1.0 | No | Embedding API configuration for serverless indexes only. Enables automatic embedding via a specified model. Ignored for pod-based indexes. When `None`, current embed config is preserved. |
| `read_capacity` | `ReadCapacityDict \| ReadCapacity \| ... \| None` | No | `None` | v1.0 | No | Read capacity configuration for serverless indexes only. Pass a dict with mode specified: `{"mode": "OnDemand"}` for auto-scaling or `{"mode": "Dedicated", "value": N}` for fixed capacity. Ignored for pod-based indexes. When `None`, current read capacity is preserved. |

### Returns

**Type:** `None`

No return value. Call `describe()` to verify configuration changes took effect.

### Raises

| Exception | Condition |
|-----------|-----------|
| `pinecone.NotFoundException` | The index does not exist. |
| `pinecone.PineconeApiException` | Invalid parameter values (e.g., incompatible pod type, unsupported replica count). |
| `ValueError` | Invalid parameter values (e.g., deletion_protection not "enabled"/"disabled", invalid read_capacity configuration). |

### Idempotency

Idempotent in effect: repeated calls with identical parameters produce the same final result. Each call is sent to the server; the method does not skip requests if the configuration is already in the desired state.

### Side Effects

- Modifies index configuration on the Pinecone service
- May trigger resource scaling (pods, replicas, read capacity)
- Queries and mutations are not interrupted during configuration changes
- Scaling happens in a rolling manner with zero downtime

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

# Update deletion protection and tags
pc.indexes.configure(
    name="my-index",
    deletion_protection="enabled",
    tags={"environment": "production", "team": "ml"}
)

# For pod-based indexes, scale up
pc.indexes.configure(
    name="my-pod-index",
    replicas=3,
    pod_type="p2.x2"
)

# Verify the changes
index = pc.indexes.describe("my-index")
print(f"Deletion protection: {index.deletion_protection}")
```


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
