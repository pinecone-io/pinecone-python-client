# Index Management Operations

This module documents index management methods on the Pinecone and PineconeAsyncio clients: listing, describing, deleting, and checking the existence of indexes. All four methods provide control plane access to query and manipulate index lifecycle.

---

## `Pinecone.list_indexes()`

Lists all indexes in your Pinecone project.

**Source:** `pinecone/pinecone.py:757-787`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def list_indexes(self) -> IndexList
```

### Returns

**Type:** `IndexList` — An iterable collection of all indexes in your project. The `IndexList` object wraps a list of `IndexModel` instances and provides a convenience method `names()` to extract just the index names.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | Failed to retrieve the index list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all indexes and access their properties
indexes = pc.list_indexes()
for index in indexes:
    print(f"Index: {index.name}, Dimension: {index.dimension}, Metric: {index.metric}")

# Get just the index names
index_names = indexes.names()
print(f"Available indexes: {index_names}")
```

### Notes

- The `IndexList` object is iterable; use it in a `for` loop to iterate over individual `IndexModel` instances.
- The `names()` method is a convenience method that returns a list of index names without needing to extract the name from each model.
- The results include all indexes regardless of their status (Ready, Initializing, Terminating, etc.).

---

## `PineconeAsyncio.list_indexes()`

Asynchronous version of `list_indexes()`. Lists all indexes in your Pinecone project.

**Source:** `pinecone/pinecone_asyncio.py:810-849`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def list_indexes(self) -> IndexList
```

### Returns

**Type:** `Awaitable[IndexList]` — An awaitable that resolves to an `IndexList` collection.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | Failed to retrieve the index list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def list_all_indexes():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    indexes = await pc.list_indexes()
    for index in indexes:
        print(f"Index: {index.name}, Status: {index.status.state}")

    await pc.close()

asyncio.run(list_all_indexes())
```

---

## `Pinecone.describe_index()`

Describes a specific Pinecone index by name.

**Source:** `pinecone/pinecone.py:789-843`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def describe_index(self, name: str) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index to describe. |

### Returns

**Type:** `IndexModel` — An object representing the index with properties including name, dimension, metric, host URL, status, and spec.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The specified index does not exist. |
| `PineconeApiException` | Failed to retrieve the index description from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Describe an index to get its configuration and status
index_name = "my-embeddings-index"
description = pc.describe_index(name=index_name)

print(f"Index name: {description.name}")
print(f"Dimension: {description.dimension}")
print(f"Metric: {description.metric}")
print(f"Host: {description.host}")
print(f"Status: {description.status.state}")
print(f"Deletion protection: {description.deletion_protection}")

# Use the host to connect to the index for data operations
index = pc.Index(host=description.host)
# Now you can call index.upsert(), index.query(), etc.
```

### Notes

- The returned `IndexModel` includes a `spec` property that describes the index's configuration (serverless, pod-based, or BYOC).
- The `host` property is the gRPC URL used to connect to the index for data plane operations.
- The `status` property contains `ready` (boolean) and `state` (string) fields indicating the index readiness.
- If the index is configured with `deletion_protection="enabled"`, it cannot be deleted until protection is disabled.

---

## `PineconeAsyncio.describe_index()`

Asynchronous version of `describe_index()`. Describes a specific Pinecone index by name.

**Source:** `pinecone/pinecone_asyncio.py:851-911`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def describe_index(self, name: str) -> IndexModel
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index to describe. |

### Returns

**Type:** `Awaitable[IndexModel]` — An awaitable that resolves to an `IndexModel` object.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The specified index does not exist. |
| `PineconeApiException` | Failed to retrieve the index description from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def get_index_host():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    description = await pc.describe_index(name="production-embeddings")
    print(f"Index is hosted at: {description.host}")
    print(f"Index status: {description.status.state}")

    # Check if deletion protection is enabled
    if description.deletion_protection == "enabled":
        print("This index is protected from deletion")

    await pc.close()

asyncio.run(get_index_host())
```

---

## `Pinecone.delete_index()`

Deletes a Pinecone index. This is an irreversible operation.

**Source:** `pinecone/pinecone.py:711-755`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent (idempotent in effect — calling twice will fail the second time if the index is already deleted)
**Side effects:** Deletes the index and all data it contains. The index transitions to "Terminating" state and is eventually removed.

### Signature

```python
def delete_index(self, name: str, timeout: int | None = None) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index to delete. |
| `timeout` | `int or None` | No | `None` | v1.0 | No | Number of seconds to wait for the delete operation to complete. If `None`, wait indefinitely. If >= 0, time out after that many seconds. If -1, return immediately without waiting for completion. |

### Returns

**Type:** `None`

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The specified index does not exist. |
| `PineconeApiException` | The delete operation failed (e.g., due to deletion protection). |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

index_name = "old-index"

# Check if the index exists
if pc.has_index(index_name):
    # Check if deletion protection is enabled
    description = pc.describe_index(name=index_name)
    if description.deletion_protection == "enabled":
        # Disable deletion protection first
        pc.configure_index(name=index_name, deletion_protection="disabled")

    # Delete the index and wait for completion (default behavior)
    pc.delete_index(name=index_name)
    print(f"Index '{index_name}' has been deleted")

# Alternative: delete without waiting
# pc.delete_index(name=index_name, timeout=-1)
# print("Delete request submitted; index will be removed asynchronously")
```

### Notes

- Deleting an index is irreversible; all data in the index will be lost.
- By default, the method blocks until the index is fully deleted (polls `describe_index()` to confirm deletion).
- After deletion is initiated, the index transitions to a "Terminating" state before being removed entirely.
- If the index has `deletion_protection="enabled"`, the delete operation will fail; you must call `configure_index()` with `deletion_protection="disabled"` first.
- Use `timeout=-1` to return immediately without waiting for the index to be fully deleted.

---

## `PineconeAsyncio.delete_index()`

Asynchronous version of `delete_index()`. Deletes a Pinecone index.

**Source:** `pinecone/pinecone_asyncio.py:757-808`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent (idempotent in effect)
**Side effects:** Deletes the index and all data it contains.

### Signature

```python
async def delete_index(self, name: str, timeout: int | None = None) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index to delete. |
| `timeout` | `int or None` | No | `None` | v1.0 | No | Number of seconds to wait for the delete operation to complete. If `None`, wait indefinitely. If >= 0, time out after that many seconds. If -1, return immediately without waiting. |

### Returns

**Type:** `Awaitable[None]`

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotFoundException` | The specified index does not exist. |
| `PineconeApiException` | The delete operation failed (e.g., due to deletion protection). |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def delete_index_safely():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    index_name = "temporary-index"

    try:
        # Delete the index and wait for completion
        await pc.delete_index(name=index_name)
        print(f"Index '{index_name}' deleted successfully")
    except Exception as e:
        print(f"Failed to delete index: {e}")
    finally:
        await pc.close()

asyncio.run(delete_index_safely())
```

---

## `Pinecone.has_index()`

Checks whether an index with the given name exists in your Pinecone project.

**Source:** `pinecone/pinecone.py:845-868`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def has_index(self, name: str) -> bool
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index to check for. |

### Returns

**Type:** `bool` — `True` if the index exists, `False` otherwise.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | Failed to check index existence due to an API error. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="sk-example-key-do-not-use")

index_name = "production-embeddings"

# Check if an index exists before creating it
if not pc.has_index(index_name):
    print(f"Index '{index_name}' does not exist. Creating...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created successfully")
else:
    print(f"Index '{index_name}' already exists")
```

### Notes

- This is a convenience method that returns a boolean, unlike `describe_index()` which raises `NotFoundException`.
- Use this method when you only need to check for existence without retrieving the full index description.

---

## `PineconeAsyncio.has_index()`

Asynchronous version of `has_index()`. Checks whether an index with the given name exists.

**Source:** `pinecone/pinecone_asyncio.py:913-939`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def has_index(self, name: str) -> bool
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the index to check for. |

### Returns

**Type:** `Awaitable[bool]` — An awaitable that resolves to `True` if the index exists, `False` otherwise.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | Failed to check index existence due to an API error. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec

async def ensure_index_exists():
    async with PineconeAsyncio(api_key="sk-example-key-do-not-use") as pc:
        index_name = "embeddings-index"

        if await pc.has_index(index_name):
            print(f"Index '{index_name}' is ready for use")
        else:
            print(f"Creating index '{index_name}'...")
            await pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )

asyncio.run(ensure_index_exists())
```

---

## Data Models

### `IndexModel`

Represents a single index in Pinecone.

**Source:** `pinecone/db_control/models/index_model.py:22-212`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `name` | `string` | No | v1.0 | No | The unique name of the index (1-45 characters). |
| `dimension` | `integer` | No | v1.0 | No | The vector dimension of the index (1-20000). |
| `metric` | `string` | No | v1.0 | No | The distance metric used by the index. One of: `"cosine"`, `"euclidean"`, `"dotproduct"`. |
| `host` | `string` | No | v1.0 | No | The gRPC host URL for connecting to the index for data operations. |
| `status` | `object` | No | v1.0 | No | Status object with `ready` (bool) and `state` (string) fields indicating readiness. |
| `spec` | `object` | No | v1.0 | No | Index configuration spec (serverless, pod-based, or BYOC). The exact structure depends on the index type. |
| `vector_type` | `string` | No | v1.0 | No | The vector type of the index. One of: `"dense"`, `"sparse"`. |
| `deletion_protection` | `string` | No | v1.0 | No | Whether deletion protection is enabled. One of: `"enabled"`, `"disabled"`. |
| `private_host` | `string` | No | v1.0 | No | The private gRPC host URL (only available for BYOC indexes). |
| `tags` | `object` | No | v1.0 | No | Dictionary of user-defined tags for the index. |
| `embed` | `object` | No | v1.0 | No | Embedded model configuration (if the index has integrated inference enabled). |

#### `to_dict()`

Converts the IndexModel instance to a dictionary representation.

**Source:** `pinecone/db_control/models/index_model.py:211-212`
**Added:** v1.0
**Deprecated:** No

**Returns:** `dict[str, Any]` — A dictionary containing all index properties with their current values.

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.describe_index(name="my-index")

# Convert to dictionary
index_dict = index.to_dict()
print(index_dict)
# Output: {"name": "my-index", "dimension": 1536, "metric": "cosine", "host": "...", ...}
```

### `IndexList`

An iterable collection of indexes.

**Source:** `pinecone/db_control/models/index_list.py:6-31`

The `IndexList` is created internally by the SDK when calling `list_indexes()`. Users do not instantiate it directly.

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `indexes` | `array of IndexModel` | No | v1.0 | No | The underlying list of `IndexModel` instances. |

#### `names()`

Convenience method that returns a list of index names without needing to extract the `name` property from each `IndexModel`.

**Source:** `pinecone/db_control/models/index_list.py:12-13`
**Added:** v1.0
**Deprecated:** No

**Returns:** `list[str]` — A list of index name strings.

#### `__iter__()`

Returns an iterator over the indexes in the list, allowing the IndexList to be used in `for` loops.

**Source:** `pinecone/db_control/models/index_list.py:21-22`
**Added:** v1.0
**Deprecated:** No

**Returns:** `Iterator[IndexModel]` — An iterator that yields `IndexModel` instances.

#### `__len__()`

Returns the number of indexes in the list.

**Source:** `pinecone/db_control/models/index_list.py:18-19`
**Added:** v1.0
**Deprecated:** No

**Returns:** `int` — The count of indexes in the list.

#### `__getitem__()`

Accesses a specific index by numeric position (0-based indexing).

**Source:** `pinecone/db_control/models/index_list.py:15-16`
**Added:** v1.0
**Deprecated:** No

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `index` | `integer` | Yes | — | v1.0 | No | The numeric index (0-based) of the index to retrieve. |

**Returns:** `IndexModel` — The `IndexModel` at the specified position.

**Raises**

| Exception | Condition |
|-----------|-----------|
| `IndexError` | The numeric index is out of bounds. |

#### `__str__()`

Returns a string representation of the IndexList.

**Source:** `pinecone/db_control/models/index_list.py:24-26`
**Added:** v1.0
**Deprecated:** No

**Returns:** `str` — A string representation of the list of indexes.

#### `__repr__()`

Returns a detailed string representation of the IndexList for debugging.

**Source:** `pinecone/db_control/models/index_list.py:27-28`
**Added:** v1.0
**Deprecated:** No

**Returns:** `str` — A detailed repr string showing the list structure and contained indexes.

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

indexes = pc.list_indexes()

# Iterate over indexes
for index in indexes:
    print(index.name)

# Get just the names
names = indexes.names()
print(names)

# Access by numeric index
first_index = indexes[0]
print(first_index.name)

# Get the count
count = len(indexes)
print(f"Total indexes: {count}")
```
