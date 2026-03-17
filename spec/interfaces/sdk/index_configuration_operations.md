# Index Configuration Operations

Documents index configuration modification methods on the Pinecone and PineconeAsyncio clients. The `configure_index()` method enables modification of pod count, pod type, deletion protection, tags, integrated inference settings, and serverless read capacity.

## Overview

**Language / runtime:** Python 3.8+
**Package:** `pinecone`
**Version:** v3.x

---

## Methods

### `Pinecone.configure_index(name: str, replicas: int | None = None, pod_type: PodType | str | None = None, deletion_protection: DeletionProtection | str | None = None, tags: dict[str, str] | None = None, embed: ConfigureIndexEmbed | Dict | None = None, read_capacity: dict | None = None) -> None`

Modifies the configuration of an existing index without waiting for the operation to complete.

**Source:** `pinecone/pinecone.py:870-1026`

**Added:** v3.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** Modifies the index configuration. Changes are applied asynchronously; use `describe_index()` to monitor status.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| name | string | Yes | — | v3.0 | No | The name of the index to configure. |
| replicas | integer (int32, 0–) | No | `None` | v3.0 | No | The desired number of replicas for pod-based indexes. When omitted, replicas are not modified. Only applies to pod-based indexes. |
| pod_type | string (enum: p1.x1, p1.x2, p1.x4, p1.x8, p2.x1, p2.x2, p2.x4, p2.x8, s1.x1, s1.x2, s1.x4, s1.x8) or `PodType` enum | No | `None` | v3.0 | No | The new pod type for pod-based indexes. When omitted, pod type is not modified. Only applies to pod-based indexes. Valid values depend on the pod-based index environment. |
| deletion_protection | string (enum: enabled, disabled) or `DeletionProtection` enum | No | `None` | v3.0 | No | Whether the index is protected from deletion. When set to `"enabled"` or `DeletionProtection.ENABLED`, the index cannot be deleted via `delete_index()`. When set to `"disabled"` or `DeletionProtection.DISABLED`, deletion protection is removed. When omitted, deletion protection status is not modified. |
| tags | dict[string, string] | No | `None` | v3.0 | No | Tags to add, update, or remove from the index. Tag updates are merged with existing tags; to remove a tag, set its value to an empty string `""`. When omitted, tags are not modified. |
| embed | `ConfigureIndexEmbed` or dict | No | `None` | v3.0 | No | Enables or updates integrated inference embeddings on the index, specifying the embedding model and configuring field mapping and read/write parameters. Once set, the embedding model cannot be changed. Only applies to serverless indexes. When omitted, embedding configuration is not modified. |
| read_capacity | dict | No | `None` | v3.0 | No | Read capacity configuration for serverless indexes, specifying whether to use on-demand or dedicated mode. When omitted, read capacity configuration is not modified. Only applies to serverless indexes. See examples for detailed structure. |

**Returns:** `None` — Configuration changes are processed asynchronously. The method returns immediately after the server accepts the request, not after changes are applied.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | No index with the given `name` exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `ValueError` | `deletion_protection` has an invalid value, or `read_capacity` configuration is invalid (e.g., missing required fields in dedicated mode). |
| `PineconeApiException` | The index modification fails on the server side (e.g., incompatible configuration changes). |

**Example**

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

**Notes**

- Configuration changes are processed asynchronously. After calling `configure_index()`, immediately call `describe_index()` to check the current status; the status field will show `"initializing"` or `"ready"` as changes are being applied.
- Multiple parameters can be modified in a single call; all requested changes are batched together.
- Tag merging: The `tags` parameter merges with existing tags rather than replacing them. To remove a tag, set its value to an empty string.
- Pod type changes (vertical scaling) may incur a brief period where the index is temporarily unavailable.
- For serverless indexes, `replicas` and `pod_type` are not applicable; use `read_capacity` instead.
- For pod-based indexes, `read_capacity` and `embed` are not applicable.

---

### `PineconeAsyncio.configure_index(name: str, replicas: int | None = None, pod_type: PodType | str | None = None, deletion_protection: DeletionProtection | str | None = None, tags: dict[str, str] | None = None, embed: ConfigureIndexEmbed | Dict | None = None, read_capacity: dict | None = None) -> None`

Asynchronous version of `configure_index()`. Modifies the configuration of an existing index without waiting for the operation to complete.

**Source:** `pinecone/pinecone_asyncio.py:941-1108`

**Added:** v3.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** Modifies the index configuration. Changes are applied asynchronously; use `describe_index()` to monitor status.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| name | string | Yes | — | v3.0 | No | The name of the index to configure. |
| replicas | integer (int32, 0–) | No | `None` | v3.0 | No | The desired number of replicas for pod-based indexes. When omitted, replicas are not modified. Only applies to pod-based indexes. |
| pod_type | string (enum: p1.x1, p1.x2, p1.x4, p1.x8, p2.x1, p2.x2, p2.x4, p2.x8, s1.x1, s1.x2, s1.x4, s1.x8) or `PodType` enum | No | `None` | v3.0 | No | The new pod type for pod-based indexes. When omitted, pod type is not modified. Only applies to pod-based indexes. Valid values depend on the pod-based index environment. |
| deletion_protection | string (enum: enabled, disabled) or `DeletionProtection` enum | No | `None` | v3.0 | No | Whether the index is protected from deletion. When set to `"enabled"` or `DeletionProtection.ENABLED`, the index cannot be deleted via `delete_index()`. When set to `"disabled"` or `DeletionProtection.DISABLED`, deletion protection is removed. When omitted, deletion protection status is not modified. |
| tags | dict[string, string] | No | `None` | v3.0 | No | Tags to add, update, or remove from the index. Tag updates are merged with existing tags; to remove a tag, set its value to an empty string `""`. When omitted, tags are not modified. |
| embed | `ConfigureIndexEmbed` or dict | No | `None` | v3.0 | No | Enables or updates integrated inference embeddings on the index, specifying the embedding model and configuring field mapping and read/write parameters. Once set, the embedding model cannot be changed. Only applies to serverless indexes. When omitted, embedding configuration is not modified. |
| read_capacity | dict | No | `None` | v3.0 | No | Read capacity configuration for serverless indexes, specifying whether to use on-demand or dedicated mode. When omitted, read capacity configuration is not modified. Only applies to serverless indexes. See examples for detailed structure. |

**Returns:** `None` — An awaitable that completes when the server accepts the request. Configuration changes are processed asynchronously after the method returns.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `NotFoundException` | No index with the given `name` exists. |
| `UnauthorizedException` | The API key is invalid or missing. |
| `ValueError` | `deletion_protection` has an invalid value, or `read_capacity` configuration is invalid (e.g., missing required fields in dedicated mode). |
| `PineconeApiException` | The index modification fails on the server side (e.g., incompatible configuration changes). |

**Example**

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

**Notes**

- Configuration changes are processed asynchronously. After awaiting `configure_index()`, call `describe_index()` to check the current status.
- All notes from the synchronous version apply equally to the asynchronous version.
