# Index Namespace Operations

This module documents namespace management methods on the Index class for serverless indexes: creating and inspecting namespaces. Namespaces allow logical separation of vector data within an index, useful for multi-tenant applications or organizing data by category.

**Source:** `pinecone/db_data/index.py:200-212`

---

## `Index.namespace.create()`

Create a namespace in a serverless index.

**Source:** `pinecone/db_data/resources/sync/namespace.py:28-47`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Creates a new namespace in the index. If a namespace with the same name already exists, returns an error.

### Signature

```python
def create(
    self,
    name: str,
    schema: Any | None = None,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to create. Must be a non-empty string (whitespace-only strings are rejected). Namespace names must be unique within the index. |
| `schema` | `Any \| None` | No | `None` | v8.1.0 | No | Optional schema configuration for the namespace. Can be a dictionary defining metadata field schemas for the namespace. When `None`, no schema constraints are applied. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `NamespaceDescription` — Information about the created namespace, including:
- `name` (string): The namespace name
- `record_count` (integer): The vector count in the namespace (zero for newly created namespaces)
- `schema` (object): The schema configuration if provided
- `indexed_fields` (object): Metadata fields configured with indexes

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `name` is not a string, or `name` is a string that is empty or contains only whitespace. |
| `PineconeApiException` | Failed to create the namespace (e.g., namespace already exists, invalid schema, or index does not support namespaces). |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

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

### Notes

- This operation is only supported for serverless indexes. Pod-based indexes do not support namespace operations.
- Namespace names must be unique within the index. Creating a namespace with a name that already exists will raise an error.
- The `record_count` of a newly created namespace is always zero.

---

## `IndexAsyncio.namespace.create()`

Async version of `Index.namespace.create()`. Creates a namespace in a serverless index.

**Source:** `pinecone/db_data/resources/asyncio/namespace_asyncio.py:19-37`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Creates a new namespace in the index. If a namespace with the same name already exists, returns an error.

### Signature

```python
async def create(
    self,
    name: str,
    schema: Any | None = None,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to create. Must be a non-empty string (whitespace-only strings are rejected). Namespace names must be unique within the index. |
| `schema` | `Any \| None` | No | `None` | v8.1.0 | No | Optional schema configuration for the namespace. Can be a dictionary defining metadata field schemas for the namespace. When `None`, no schema constraints are applied. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `NamespaceDescription` — Information about the created namespace, including:
- `name` (string): The namespace name
- `record_count` (integer): The vector count in the namespace (zero for newly created namespaces)
- `schema` (object): The schema configuration if provided
- `indexed_fields` (object): Metadata fields configured with indexes

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `name` is not a string, or `name` is a string that is empty or contains only whitespace. |
| `PineconeApiException` | Failed to create the namespace (e.g., namespace already exists, invalid schema, or index does not support namespaces). |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio("serverless-index") as index:
        namespace_info = await index.namespace.create(name="user-data")
        print(namespace_info.name)  # "user-data"
        print(namespace_info.record_count)  # 0

asyncio.run(main())
```

### Notes

- Identical behavior to the sync version; see `Index.namespace.create()` notes.

---

## `Index.namespace.describe()`

Describe a namespace within an index, showing metadata and vector statistics.

**Source:** `pinecone/db_data/resources/sync/namespace.py:49-64`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

### Signature

```python
def describe(
    self,
    namespace: str,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to describe. Must be a string. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `NamespaceDescription` — Information about the namespace, including:
- `name` (string): The namespace name
- `record_count` (integer): The total number of vectors in the namespace
- `schema` (object): The schema configuration, or `None` if no schema was applied
- `indexed_fields` (object): Metadata fields configured with indexes

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `namespace` is not a string. |
| `NotFoundException` | The namespace does not exist in the index. |
| `PineconeApiException` | Failed to retrieve namespace information from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

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

### Notes

- This is a read-only operation that does not modify the namespace or index state.
- The `record_count` reflects the current vector count and is updated in real-time as vectors are upserted or deleted.
- If a namespace was created without a schema, the `schema` field will be `None` in the response.

---

## `IndexAsyncio.namespace.describe()`

Async version of `Index.namespace.describe()`. Describes a namespace within an index.

**Source:** `pinecone/db_data/resources/asyncio/namespace_asyncio.py:39-54`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

### Signature

```python
async def describe(
    self,
    namespace: str,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to describe. Must be a string. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `NamespaceDescription` — Information about the namespace, including:
- `name` (string): The namespace name
- `record_count` (integer): The total number of vectors in the namespace
- `schema` (object): The schema configuration, or `None` if no schema was applied
- `indexed_fields` (object): Metadata fields configured with indexes

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `namespace` is not a string. |
| `NotFoundException` | The namespace does not exist in the index. |
| `PineconeApiException` | Failed to retrieve namespace information from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio("serverless-index") as index:
        ns_info = await index.namespace.describe(namespace="user-data")
        print(f"Namespace: {ns_info.name}, Vectors: {ns_info.record_count}")

asyncio.run(main())
```

### Notes

- Identical behavior to the sync version; see `Index.namespace.describe()` notes.

---

## `Index.namespace.delete()`

Delete a namespace from an index.

**Source:** `pinecone/db_data/resources/sync/namespace.py:66-75`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Deletes the namespace and all vectors within it from the index.

### Signature

```python
def delete(
    self,
    namespace: str,
    **kwargs
) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to delete. Must be a string. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `dict[str, Any]` — A response dictionary from the API. The structure is not strongly typed but typically indicates successful deletion.

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `namespace` is not a string. |
| `NotFoundException` | The namespace does not exist in the index. |
| `PineconeApiException` | Failed to delete the namespace from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

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

### Notes

- This operation is destructive and cannot be undone. All vectors in the namespace are permanently deleted.
- After deletion, the namespace can be recreated with the same name, but it will start empty.
- Deleting a namespace does not affect other namespaces in the index.

---

## `IndexAsyncio.namespace.delete()`

Async version of `Index.namespace.delete()`. Deletes a namespace from an index.

**Source:** `pinecone/db_data/resources/asyncio/namespace_asyncio.py:56-65`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Deletes the namespace and all vectors within it from the index.

### Signature

```python
async def delete(
    self,
    namespace: str,
    **kwargs
) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to delete. Must be a string. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `dict[str, Any]` — A response dictionary from the API. The structure is not strongly typed but typically indicates successful deletion.

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `ValueError` | `namespace` is not a string. |
| `NotFoundException` | The namespace does not exist in the index. |
| `PineconeApiException` | Failed to delete the namespace from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio("serverless-index") as index:
        await index.namespace.delete(namespace="old-namespace")
        print("Namespace deleted successfully")

asyncio.run(main())
```

### Notes

- Identical behavior to the sync version; see `Index.namespace.delete()` notes.

---

## `Index.namespace.list()`

List all namespaces in an index (generator).

**Source:** `pinecone/db_data/resources/sync/namespace.py:77-111`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

### Signature

```python
def list(
    self,
    limit: int | None = None,
    **kwargs
) -> Iterator[NamespaceDescription]
```

> **Note on type annotation:** The source code annotates the return type as `Iterator[ListNamespacesResponse]`, but the generator actually yields individual `NamespaceDescription` items extracted from each page's `namespaces` list. The effective yield type is `NamespaceDescription`.

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | — | v8.1.0 | No | The maximum number of namespaces to fetch in each network call. When omitted, the server uses a default value. Pagination is handled automatically. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call, including `pagination_token` for resuming iteration (rarely used directly). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `Iterator[NamespaceDescription]` — A generator that yields individual `NamespaceDescription` objects automatically across all pages. Each yielded item has:
- `name` (string): The namespace name
- `record_count` (integer): The vector count in the namespace

Pagination is handled transparently as you iterate.

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `PineconeApiException` | Failed to retrieve the namespace list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

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

### Notes

- The `list()` method is a generator and automatically handles pagination tokens on your behalf.
- The generator will fetch namespaces in pages as you iterate, reducing memory usage compared to fetching all at once.
- Namespace order in the results is not guaranteed to be stable across calls.

---

## `IndexAsyncio.namespace.list()`

Async version of `Index.namespace.list()`. Lists all namespaces in an index (async generator).

**Source:** `pinecone/db_data/resources/asyncio/namespace_asyncio.py:67-103`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

### Signature

```python
async def list(
    self,
    limit: int | None = None,
    **kwargs
) -> AsyncIterator[NamespaceDescription]
```

> **Note on type annotation:** The source code annotates the return type as `AsyncIterator[ListNamespacesResponse]`, but the generator actually yields individual `NamespaceDescription` items. The effective yield type is `NamespaceDescription`.

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | — | v8.1.0 | No | The maximum number of namespaces to fetch in each network call. When omitted, the server uses a default value. Pagination is handled automatically. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call, including `pagination_token` for resuming iteration (rarely used directly). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `AsyncIterator[NamespaceDescription]` — An async generator that yields individual `NamespaceDescription` objects automatically across all pages. Pagination is handled transparently.

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `PineconeApiException` | Failed to retrieve the namespace list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio("serverless-index") as index:
        async for ns in index.namespace.list():
            print(f"Namespace: {ns.name}, Vectors: {ns.record_count}")

        # Convert to list via async comprehension
        all_namespaces = [ns async for ns in index.namespace.list(limit=100)]
        print(f"Total namespaces: {len(all_namespaces)}")

asyncio.run(main())
```

### Notes

- Be cautious when collecting all results into a list using async comprehension, as this may trigger many network calls for indexes with many namespaces.
- Otherwise identical behavior to the sync version; see `Index.namespace.list()` notes.

---

## `Index.namespace.list_paginated()`

List namespaces in an index with manual pagination control.

**Source:** `pinecone/db_data/resources/sync/namespace.py:113-143`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

### Signature

```python
def list_paginated(
    self,
    limit: int | None = None,
    pagination_token: str | None = None,
    **kwargs
) -> ListNamespacesResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | `None` | v8.1.0 | No | The maximum number of namespaces to return in a single response. If unspecified, the server uses a default value. |
| `pagination_token` | `string \| None` | No | `None` | v8.1.0 | No | A token returned from a previous call to fetch the next page of results. Pass this to continue iterating through results. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `ListNamespacesResponse` — A response object containing:
- `namespaces` (list of `NamespaceDescription`): The namespaces in this page of results
- `pagination` (object or `None`): Pagination info with a `next` token if more results are available

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `PineconeApiException` | Failed to retrieve the namespace list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index("serverless-index")

# Get first page
results = index.namespace.list_paginated(limit=5)
for ns in results.namespaces:
    print(f"Namespace: {ns.name}")

# Get next page if available
if results.pagination and results.pagination.next:
    next_results = index.namespace.list_paginated(
        limit=5,
        pagination_token=results.pagination.next
    )
```

### Notes

- Consider using `namespace.list()` instead to avoid manual pagination token handling.
- Each call returns a single page of results. You must check `results.pagination.next` and pass it as `pagination_token` to get subsequent pages.

---

## `IndexAsyncio.namespace.list_paginated()`

Async version of `Index.namespace.list_paginated()`. Lists namespaces with manual pagination control.

**Source:** `pinecone/db_data/resources/asyncio/namespace_asyncio.py:105-135`

**Added:** v8.1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation)

### Signature

```python
async def list_paginated(
    self,
    limit: int | None = None,
    pagination_token: str | None = None,
    **kwargs
) -> ListNamespacesResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | `None` | v8.1.0 | No | The maximum number of namespaces to return in a single response. If unspecified, the server uses a default value. |
| `pagination_token` | `string \| None` | No | `None` | v8.1.0 | No | A token returned from a previous call to fetch the next page of results. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments for the API call (rarely used). Must be passed as keyword arguments, not positional. |

### Returns

**Type:** `ListNamespacesResponse` — A response object containing:
- `namespaces` (list of `NamespaceDescription`): The namespaces in this page of results
- `pagination` (object or `None`): Pagination info with a `next` token if more results are available

### Raises

| Exception | Condition |
|-----------|-----------|
| `TypeError` | Arguments are passed positionally instead of as keyword arguments. |
| `PineconeApiException` | Failed to retrieve the namespace list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio("serverless-index") as index:
        results = await index.namespace.list_paginated(limit=5)
        for ns in results.namespaces:
            print(f"Namespace: {ns.name}")

        if results.pagination and results.pagination.next:
            next_results = await index.namespace.list_paginated(
                limit=5,
                pagination_token=results.pagination.next
            )

asyncio.run(main())
```

### Notes

- Consider using `namespace.list()` instead to avoid manual pagination token handling.
- Identical behavior to the sync version; see `Index.namespace.list_paginated()` notes.

---

## Convenience Methods on `Index`

The following methods are defined directly on the `Index` class and delegate to the corresponding `namespace` resource methods. They apply the `@validate_and_convert_errors` decorator which normalizes API exceptions into SDK exception types. These are provided as a shorthand so callers do not need to access the `.namespace` sub-resource.

---

## `Index.create_namespace()`

Convenience wrapper for `Index.namespace.create()`.

**Source:** `pinecone/db_data/index.py:1661-1693`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `self.namespace.create(name=name, schema=schema, **kwargs)`

### Signature

```python
def create_namespace(
    self,
    name: str,
    schema: dict[str, Any] | None = None,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to create. |
| `schema` | `dict[str, Any] \| None` | No | `None` | v8.1.0 | No | Optional schema configuration for the namespace as a dictionary. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `NamespaceDescription` — Same as `Index.namespace.create()`.

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index("serverless-index")

namespace = index.create_namespace(name="my-namespace")
print(f"Created: {namespace.name}")
```

---

## `Index.describe_namespace()`

Convenience wrapper for `Index.namespace.describe()`.

**Source:** `pinecone/db_data/index.py:1695-1716`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `self.namespace.describe(namespace=namespace, **kwargs)`

### Signature

```python
def describe_namespace(
    self,
    namespace: str,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to describe. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `NamespaceDescription` — Same as `Index.namespace.describe()`.

### Example

```python
ns_info = index.describe_namespace(namespace="my-namespace")
print(f"Namespace: {ns_info.name}, Vectors: {ns_info.record_count}")
```

---

## `Index.delete_namespace()`

Convenience wrapper for `Index.namespace.delete()`.

**Source:** `pinecone/db_data/index.py:1718-1741`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `self.namespace.delete(namespace=namespace, **kwargs)`

### Signature

```python
def delete_namespace(
    self,
    namespace: str,
    **kwargs
) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to delete. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `dict[str, Any]` — Same as `Index.namespace.delete()`.

### Example

```python
index.delete_namespace(namespace="old-namespace")
```

---

## `Index.list_namespaces()`

Convenience wrapper for `Index.namespace.list()`.

**Source:** `pinecone/db_data/index.py:1743-1775`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `self.namespace.list(limit=limit, **kwargs)`

### Signature

```python
def list_namespaces(
    self,
    limit: int | None = None,
    **kwargs
) -> Iterator[NamespaceDescription]
```

> **Note on type annotation:** The source code annotates the return type as `Iterator[ListNamespacesResponse]`, but the generator actually yields individual `NamespaceDescription` items (delegated from `namespace.list()`).

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | `None` | v8.1.0 | No | The maximum number of namespaces to fetch in each network call. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `Iterator[NamespaceDescription]` — Same as `Index.namespace.list()`. A generator yielding individual `NamespaceDescription` objects with automatic pagination.

### Example

```python
for ns in index.list_namespaces(limit=10):
    print(f"Namespace: {ns.name}")
```

---

## `Index.list_namespaces_paginated()`

Convenience wrapper for `Index.namespace.list_paginated()`.

**Source:** `pinecone/db_data/index.py:1777-1816`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `self.namespace.list_paginated(limit=limit, pagination_token=pagination_token, **kwargs)`

### Signature

```python
def list_namespaces_paginated(
    self,
    limit: int | None = None,
    pagination_token: str | None = None,
    **kwargs
) -> ListNamespacesResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | `None` | v8.1.0 | No | The maximum number of namespaces to return in a single response. |
| `pagination_token` | `string \| None` | No | `None` | v8.1.0 | No | A token returned from a previous call to fetch the next page. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `ListNamespacesResponse` — Same as `Index.namespace.list_paginated()`.

### Example

```python
results = index.list_namespaces_paginated(limit=5)
for ns in results.namespaces:
    print(ns.name)

if results.pagination and results.pagination.next:
    next_page = index.list_namespaces_paginated(
        limit=5,
        pagination_token=results.pagination.next
    )
```

---

## Convenience Methods on `IndexAsyncio`

Async equivalents of the `Index` convenience methods. These are defined directly on `IndexAsyncio` and delegate to the corresponding `namespace` resource async methods. They apply the `@validate_and_convert_errors` decorator.

---

## `IndexAsyncio.create_namespace()`

Async convenience wrapper for `IndexAsyncio.namespace.create()`.

**Source:** `pinecone/db_data/index_asyncio.py:1627-1665`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `await self.namespace.create(name=name, schema=schema, **kwargs)`

### Signature

```python
async def create_namespace(
    self,
    name: str,
    schema: dict[str, Any] | None = None,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to create. |
| `schema` | `dict[str, Any] \| None` | No | `None` | v8.1.0 | No | Optional schema configuration for the namespace as a dictionary. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `NamespaceDescription` — Same as `Index.namespace.create()`.

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio("serverless-index") as index:
        namespace = await index.create_namespace(name="my-namespace")
        print(f"Created: {namespace.name}")

asyncio.run(main())
```

---

## `IndexAsyncio.describe_namespace()`

Async convenience wrapper for `IndexAsyncio.namespace.describe()`.

**Source:** `pinecone/db_data/index_asyncio.py:1667-1678`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `await self.namespace.describe(namespace=namespace, **kwargs)`

### Signature

```python
async def describe_namespace(
    self,
    namespace: str,
    **kwargs
) -> NamespaceDescription
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to describe. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `NamespaceDescription` — Same as `Index.namespace.describe()`.

### Example

```python
async with pc.IndexAsyncio("serverless-index") as index:
    ns_info = await index.describe_namespace(namespace="my-namespace")
    print(f"Vectors: {ns_info.record_count}")
```

---

## `IndexAsyncio.delete_namespace()`

Async convenience wrapper for `IndexAsyncio.namespace.delete()`.

**Source:** `pinecone/db_data/index_asyncio.py:1680-1694`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `await self.namespace.delete(namespace=namespace, **kwargs)`

### Signature

```python
async def delete_namespace(
    self,
    namespace: str,
    **kwargs
) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `string` | Yes | — | v8.1.0 | No | The name of the namespace to delete. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `dict[str, Any]` — Same as `Index.namespace.delete()`.

### Example

```python
async with pc.IndexAsyncio("serverless-index") as index:
    await index.delete_namespace(namespace="old-namespace")
```

---

## `IndexAsyncio.list_namespaces()`

Async convenience wrapper for `IndexAsyncio.namespace.list()`.

**Source:** `pinecone/db_data/index_asyncio.py:1696-1717`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `self.namespace.list(limit=limit, **kwargs)` (re-yields each item)

### Signature

```python
async def list_namespaces(
    self,
    limit: int | None = None,
    **kwargs
) -> AsyncIterator[NamespaceDescription]
```

> **Note on type annotation:** The source code annotates the return type as `AsyncIterator[ListNamespacesResponse]`, but the generator actually yields individual `NamespaceDescription` items.

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | `None` | v8.1.0 | No | The maximum number of namespaces to fetch in each network call. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `AsyncIterator[NamespaceDescription]` — An async generator yielding individual `NamespaceDescription` objects with automatic pagination.

### Example

```python
async with pc.IndexAsyncio("serverless-index") as index:
    async for ns in index.list_namespaces(limit=10):
        print(f"Namespace: {ns.name}")
```

---

## `IndexAsyncio.list_namespaces_paginated()`

Async convenience wrapper for `IndexAsyncio.namespace.list_paginated()`.

**Source:** `pinecone/db_data/index_asyncio.py:1719-1745`

**Added:** v8.1.0
**Deprecated:** No
**Delegates to:** `await self.namespace.list_paginated(limit=limit, pagination_token=pagination_token, **kwargs)`

### Signature

```python
async def list_namespaces_paginated(
    self,
    limit: int | None = None,
    pagination_token: str | None = None,
    **kwargs
) -> ListNamespacesResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `limit` | `integer (int32) \| None` | No | `None` | v8.1.0 | No | The maximum number of namespaces to return in a single response. |
| `pagination_token` | `string \| None` | No | `None` | v8.1.0 | No | A token returned from a previous call to fetch the next page. |
| `**kwargs` | `Any` | No | — | v8.1.0 | No | Additional keyword arguments passed through to the API call. |

### Returns

**Type:** `ListNamespacesResponse` — Same as `Index.namespace.list_paginated()`.

### Example

```python
async with pc.IndexAsyncio("serverless-index") as index:
    results = await index.list_namespaces_paginated(limit=5)
    for ns in results.namespaces:
        print(ns.name)

    if results.pagination and results.pagination.next:
        next_page = await index.list_namespaces_paginated(
            limit=5,
            pagination_token=results.pagination.next
        )
```
