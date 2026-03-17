# Index Namespace Operations

This module documents namespace management methods on the Index class for serverless indexes: creating and inspecting namespaces. Namespaces allow logical separation of vector data within an index, useful for multi-tenant applications or organizing data by category.

## Overview

**Language / runtime:** Python 3.8+
**Package:** `pinecone`
**Version:** v8.1.0
**Breaking change definition:** Changing the return type or return value structure of any method, removing a method, or renaming a parameter.

## Classes

### `Index.namespace`

Accessor for namespace management operations on an index.

**Import:** Accessed via `Index.namespace`, not imported directly.
**Source:** `pinecone/db_data/index.py:200-212`

Provides methods to create, describe, delete, and list namespaces within a serverless index. This resource is only available for serverless indexes; pod-based indexes do not support namespace operations.

**Methods**

#### `namespace.create(name: str, schema: Any | None = None, **kwargs) -> NamespaceDescription`

Create a namespace in a serverless index.

**Source:** `pinecone/db_data/resources/sync/namespace.py:28-47`
**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Creates a new namespace in the index. If a namespace with the same name already exists, returns an error.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| name | string | Yes | — | v8.1.0 | No | The name of the namespace to create. Must be a non-empty string (whitespace-only strings are rejected). Namespace names must be unique within the index. |
| schema | Any \| None | No | `None` | v8.1.0 | No | Optional schema configuration for the namespace. Can be a dictionary defining metadata field schemas for the namespace. When `None`, no schema constraints are applied. |
| **kwargs | Any | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

**Returns:** `NamespaceDescription` — Information about the created namespace, including:
- `name` (string): The namespace name
- `record_count` (integer): The vector count in the namespace (zero for newly created namespaces)
- `schema` (object): The schema configuration if provided
- `indexed_fields` (object): Metadata fields configured with indexes

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `name` is not a string, or `name` is a string that is empty or contains only whitespace. |
| `PineconeApiException` | Failed to create the namespace (e.g., namespace already exists, invalid schema, or index does not support namespaces). |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index("serverless-index")

# Create a simple namespace
namespace_info = index.namespace.create(name="user-data")
print(namespace_info.name)  # "user-data"
print(namespace_info.record_count)  # 0

# Create a namespace with metadata schema
schema = {
    "indexed": ["user_tier", "region"]
}
ns_with_schema = index.namespace.create(name="premium-users", schema=schema)
```

**Notes**

- This operation is only supported for serverless indexes. Pod-based indexes do not support namespace operations.
- Namespace names must be unique within the index. Creating a namespace with a name that already exists will raise an error.
- The `record_count` of a newly created namespace is always zero.

---

#### `namespace.describe(namespace: str, **kwargs) -> NamespaceDescription`

Describe a namespace within an index, showing metadata and vector statistics.

**Source:** `pinecone/db_data/resources/sync/namespace.py:49-64`
**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| namespace | string | Yes | — | v8.1.0 | No | The name of the namespace to describe. Must be a string. |
| **kwargs | Any | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

**Returns:** `NamespaceDescription` — Information about the namespace, including:
- `name` (string): The namespace name
- `record_count` (integer): The total number of vectors in the namespace
- `schema` (object): The schema configuration, or `None` if no schema was applied
- `indexed_fields` (object): Metadata fields configured with indexes

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `namespace` is not a string. |
| `NotFoundException` | The namespace does not exist in the index. |
| `PineconeApiException` | Failed to retrieve namespace information from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index("serverless-index")

# Describe a namespace
ns_info = index.namespace.describe(namespace="user-data")
print(f"Namespace: {ns_info.name}")  # "user-data"
print(f"Vector count: {ns_info.record_count}")  # 1000
print(f"Schema: {ns_info.schema}")  # Schema object or None

# Check if a namespace is empty
if ns_info.record_count == 0:
    print("This namespace contains no vectors")
```

**Notes**

- This is a read-only operation that does not modify the namespace or index state.
- The `record_count` reflects the current vector count and is updated in real-time as vectors are upserted or deleted.
- If a namespace was created without a schema, the `schema` field will be `None` in the response.

---

#### `namespace.delete(namespace: str, **kwargs) -> dict[str, Any]`

Delete a namespace from an index.

**Source:** `pinecone/db_data/resources/sync/namespace.py:66-75`
**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Deletes the namespace and all vectors within it from the index.

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| namespace | string | Yes | — | v8.1.0 | No | The name of the namespace to delete. Must be a string. |
| **kwargs | Any | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

**Returns:** `dict[str, Any]` — A response dictionary from the API. The structure is not strongly typed but typically indicates successful deletion.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `namespace` is not a string. |
| `NotFoundException` | The namespace does not exist in the index. |
| `PineconeApiException` | Failed to delete the namespace from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index("serverless-index")

# Delete a namespace and all its vectors
index.namespace.delete(namespace="old-namespace")
print("Namespace deleted successfully")

# Attempting to describe the deleted namespace will raise NotFoundException
try:
    index.namespace.describe(namespace="old-namespace")
except Exception as e:
    print(f"Error: {type(e).__name__}")  # NotFoundException
```

**Notes**

- This operation is destructive and cannot be undone. All vectors in the namespace are permanently deleted.
- After deletion, the namespace can be recreated with the same name, but it will start empty.
- Deleting a namespace does not affect other namespaces in the index.

---

#### `namespace.list(limit: int | None = None, **kwargs) -> Iterator[ListNamespacesResponse]`

List all namespaces in an index (generator).

**Source:** `pinecone/db_data/resources/sync/namespace.py:77-111`
**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| limit | integer (int32) \| None | No | — | v8.1.0 | No | The maximum number of namespaces to fetch in each network call. When omitted, the server uses a default value. Pagination is handled automatically. |
| **kwargs | Any | No | — | v8.1.0 | No | Additional keyword arguments for the API call, including `pagination_token` for resuming iteration (rarely used directly). Must be passed as keyword arguments, not positional. |

**Returns:** `Iterator[ListNamespacesResponse]` — A generator that yields namespace objects automatically across all pages. Pagination is handled transparently as you iterate.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `PineconeApiException` | Failed to retrieve the namespace list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index("serverless-index")

# Iterate over all namespaces
for ns in index.namespace.list():
    print(f"Namespace: {ns.name}, Vectors: {ns.record_count}")

# Convert to list (be cautious with very large numbers of namespaces)
all_namespaces = list(index.namespace.list(limit=100))
print(f"Total namespaces: {len(all_namespaces)}")

# Iterate with custom page size
for ns in index.namespace.list(limit=10):
    print(ns.name)
```

**Notes**

- The `list()` method is a generator and automatically handles pagination tokens on your behalf.
- The generator will fetch namespaces in pages as you iterate, reducing memory usage compared to fetching all at once.
- Namespace order in the results is not guaranteed to be stable across calls.
