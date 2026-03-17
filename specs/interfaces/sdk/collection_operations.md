# Collection Operations

This module documents collection management operations on the Pinecone and PineconeAsyncio clients: creating, listing, deleting, and describing collections. Collections are serverless copies of pod-based indexes that provide a point-in-time snapshot of your data.

---

## `Pinecone.create_collection()`

Creates a collection from a pod-based index.

**Source:** `pinecone/pinecone.py:1028-1046`, `pinecone/pinecone_asyncio.py:1110-1116` (async equivalent)
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent — repeated calls will fail if the collection already exists
**Side effects:** Creates a new collection resource in the Pinecone API

### Signature

```python
def create_collection(self, name: str, source: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `str` | Yes | — | v1.0 | No | The name of the collection. Must be a valid collection name (alphanumeric characters, hyphens, underscores). |
| `source` | `str` | Yes | — | v1.0 | No | The name of the pod-based index to use as the source for the collection. The source index must exist and be pod-based. |

### Returns

**Type:** `None` — This method returns nothing on success. The collection is created asynchronously; use `describe_collection()` to poll for status.

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | The API key is invalid or missing. |
| `ForbiddenException` | The user lacks permission to create collections. |
| `NotFoundException` | The source index does not exist. |
| `PineconeApiException` | The source index is serverless (not pod-based), a collection with the same name already exists, or another API error occurs. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Create a collection from an existing pod-based index
pc.create_collection(name="my-collection", source="my-index")
print("Collection created successfully")
```

### Notes

- Collections provide a serverless snapshot of your data and can be used for backup or analysis purposes.
- The source index must be pod-based; serverless indexes cannot be used as sources.
- The source index remains fully operational; collections are read-only snapshots.
- Collection creation is asynchronous; the collection becomes available within a few moments of the API call returning.

---

## `PineconeAsyncio.create_collection()`

Asynchronous version of `create_collection()`. Creates a collection from a pod-based index.

**Source:** `pinecone/pinecone_asyncio.py:1110-1116`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent
**Side effects:** Creates a new collection resource in the Pinecone API

### Signature

```python
async def create_collection(self, name: str, source: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `str` | Yes | — | v1.0 | No | The name of the collection. |
| `source` | `str` | Yes | — | v1.0 | No | The name of the source pod-based index. |

### Returns

**Type:** `Awaitable[None]` — An awaitable that resolves to `None` on success.

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The request failed. May occur if the `name` or `source` parameters are invalid, if a collection with the same name already exists, or if the source index does not exist or is not pod-based. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def create_collection():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    try:
        await pc.create_collection(name="my-collection", source="my-index")
        print("Collection created successfully")
    finally:
        await pc.close()

asyncio.run(create_collection())
```

---

## `Pinecone.list_collections()`

Lists all collections in your Pinecone project.

**Source:** `pinecone/pinecone.py:1048-1068`, `pinecone/pinecone_asyncio.py:1118-1147` (async equivalent)
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def list_collections(self) -> CollectionList
```

### Returns

**Type:** `CollectionList` — An iterable collection of all collections. The `CollectionList` object wraps a list of collection items and provides a convenience method `names()` to extract just the collection names.

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | The API key is missing or invalid. |
| `ForbiddenException` | The user lacks permission to list collections. |
| `PineconeApiException` | Failed to retrieve the collection list from the API. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all collections and access their properties
collections = pc.list_collections()
for collection in collections:
    print(f"Collection: {collection['name']}, Source: {collection['source']}")

# Get just the collection names
collection_names = collections.names()
print(f"Available collections: {collection_names}")

# Get the number of collections
print(f"Total collections: {len(collections)}")
```

### Notes

- The `CollectionList` object is iterable; use it in a `for` loop to iterate over individual collection items.
- The `names()` method is a convenience method that returns a list of collection names without needing to extract the name from each item.
- Collections are returned regardless of their creation status.
- All collections are returned; there is no pagination or filtering.

---

## `PineconeAsyncio.list_collections()`

Asynchronous version of `list_collections()`. Lists all collections in your Pinecone project.

**Source:** `pinecone/pinecone_asyncio.py:1118-1147`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def list_collections(self) -> CollectionList
```

### Returns

**Type:** `CollectionList` — A collection of all collections (when awaited).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | Failed to retrieve the collection list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def list_all_collections():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    collections = await pc.list_collections()
    for collection in collections:
        print(f"Collection: {collection['name']}")

    await pc.close()

asyncio.run(list_all_collections())
```

---

## `Pinecone.delete_collection()`

Deletes a collection. This is an irreversible operation.

**Source:** `pinecone/pinecone.py:1070-1092`, `pinecone/pinecone_asyncio.py:1149-1171` (async equivalent)
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent (idempotent in effect — calling twice will fail the second time if the collection is already deleted)
**Side effects:** Deletes the collection resource. The collection transitions to a deleted state and is eventually removed.

### Signature

```python
def delete_collection(self, name: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `str` | Yes | — | v1.0 | No | The name of the collection to delete. |

### Returns

**Type:** `None`

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | The API key is missing or invalid. |
| `ForbiddenException` | The user lacks permission to delete collections. |
| `NotFoundException` | The collection does not exist. |
| `PineconeApiException` | The delete operation failed due to another API error. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

collection_name = "old-collection"

# Delete the collection
pc.delete_collection(name=collection_name)
print(f"Collection '{collection_name}' has been deleted")

# Verify deletion by checking that describe_collection raises NotFoundException
try:
    pc.describe_collection(name=collection_name)
except NotFoundException:
    print("Collection has been deleted")
```

### Notes

- Deleting a collection is irreversible; all data in the collection will be lost.
- Collection deletion is asynchronous; the collection becomes unavailable immediately, but cleanup may take a few moments.
- After sending a deletion request, use `describe_collection()` to verify the collection has been fully deleted (it will raise `NotFoundException` once deletion completes).

---

## `PineconeAsyncio.delete_collection()`

Asynchronous version of `delete_collection()`. Deletes a collection.

**Source:** `pinecone/pinecone_asyncio.py:1149-1171`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent (idempotent in effect)
**Side effects:** Deletes the collection and all data it contains.

### Signature

```python
async def delete_collection(self, name: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `str` | Yes | — | v1.0 | No | The name of the collection to delete. |

### Returns

**Type:** `None` — Returns `None` on success (when awaited).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | The delete operation failed, such as when the collection does not exist. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def delete_collection():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    try:
        await pc.delete_collection(name="old-collection")
        print("Collection deleted successfully")
    except Exception as e:
        print(f"Failed to delete collection: {e}")
    finally:
        await pc.close()

asyncio.run(delete_collection())
```

---

## `Pinecone.describe_collection()`

Describes a specific collection by name.

**Source:** `pinecone/pinecone.py:1094-1117`, `pinecone/pinecone_asyncio.py:1173-1195` (async equivalent)
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
def describe_collection(self, name: str) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `str` | Yes | — | v1.0 | No | The name of the collection to describe. |

### Returns

**Type:** `dict[str, Any]` — A dictionary representing the collection with properties including name, status, environment, size, dimension, and vector_count.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `name` | `string` | No | The name of the collection. |
| `status` | `string` | No | The status of the collection. One of `Initializing` or `Ready`. |
| `size` | `integer (int64)` | No | The size of the collection in bytes. |
| `vector_count` | `integer (int64)` | No | The number of vectors in the collection. |
| `environment` | `string` | No | The environment (region) where the collection resides. |
| `dimension` | `integer (int32)` | No | The number of dimensions for vectors in the collection. |

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | The API key is missing or invalid. |
| `ForbiddenException` | The user lacks permission to describe collections. |
| `NotFoundException` | The collection does not exist. |
| `PineconeApiException` | Another API error occurs. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Describe a collection to get its configuration and status
collection_name = "my-collection"
description = pc.describe_collection(name=collection_name)

print(f"Collection name: {description['name']}")
print(f"Status: {description['status']}")
print(f"Environment: {description['environment']}")
print(f"Dimension: {description['dimension']}")
print(f"Vector count: {description['vector_count']}")
print(f"Size: {description['size']} bytes")
```

### Notes

- Collections in `Initializing` status are being created from the source index and are not yet ready for use.
- The `vector_count` and `size` reflect the snapshot state at the time of collection creation.
- The `status` field indicates the operational state of the collection (e.g., "Ready", "Initializing", "Terminating").

---

## `PineconeAsyncio.describe_collection()`

Asynchronous version of `describe_collection()`. Describes a specific collection by name.

**Source:** `pinecone/pinecone_asyncio.py:1173-1194`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

### Signature

```python
async def describe_collection(self, name: str) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `str` | Yes | — | v1.0 | No | The name of the collection to describe. |

### Returns

**Type:** `dict[str, Any]` — A dictionary representing the collection (when awaited).

### Raises

| Exception | Condition |
|-----------|-----------|
| `PineconeApiException` | Failed to retrieve the collection description, such as when the collection does not exist. |
| `UnauthorizedException` | The API key is invalid or missing. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def describe_collection():
    pc = PineconeAsyncio(api_key="sk-example-key-do-not-use")

    description = await pc.describe_collection(name="my-collection")
    print(f"Collection name: {description['name']}")
    print(f"Status: {description.get('status')}")

    await pc.close()

asyncio.run(describe_collection())
```

---

## Error Handling

All collection operations may raise the following exceptions:

| Exception | Cause | Handling |
|-----------|-------|----------|
| `PineconeApiException` | Unexpected server error | Implement retry logic with exponential backoff |
| `BadRequestException` | Malformed request, invalid parameter values | Validate inputs before retrying |
| `UnauthorizedException` | Invalid or missing API key | Verify `PINECONE_API_KEY` environment variable or constructor argument |
| `NotFoundException` | Resource (collection or source index) does not exist | Confirm resource names; list operations to verify existence |
| `ConflictException` | Resource already exists (for create operations) | Check if resource exists before attempting creation |

---

## Usage Patterns

### Complete Workflow

```python
from pinecone import Pinecone

# Initialize the Pinecone client
pc = Pinecone(api_key="sk-example-key-do-not-use")

# Create a collection from a pod-based index
pc.create_collection(name="my-collection", source="my-index")
print("Collection created")

# List all collections
collections = pc.list_collections()
print(f"Collections: {collections.names()}")

# Describe a collection
description = pc.describe_collection(name="my-collection")
print(f"Collection status: {description.get('status')}")

# Delete the collection
pc.delete_collection(name="my-collection")
print("Collection deleted")
```

### Error Handling

```python
from pinecone import Pinecone
from pinecone.exceptions import NotFoundException, ConflictException

pc = Pinecone(api_key="sk-example-key-do-not-use")

try:
    pc.create_collection(name="my-collection", source="my-index")
except ConflictException:
    print("Collection already exists")
except NotFoundException:
    print("Source index not found")

try:
    description = pc.describe_collection(name="my-collection")
except NotFoundException:
    print("Collection not found")
```

---

## Data Models

### `CollectionList`

An iterable collection of collections returned by the `list_collections()` method.

**Source:** `pinecone/db_control/models/collection_list.py:7-36`

| Method/Operation | Description |
|------------------|-------------|
| `names()` | Returns a list of collection names (strings). |
| `__iter__()` | Iterate over individual collection objects. |
| `__len__()` | Get the number of collections. |
| `__getitem__(index)` | Access a collection by index. Raises `IndexError` if out of bounds. |
| `__str__()` | Returns the string representation of the underlying collection list. |
| `__repr__()` | Returns a JSON-formatted string of all collections with their properties. |

Each collection object has the following fields:
- `name` (string): The name of the collection
- `status` (string): The collection status (`Initializing` or `Ready`)
- `size` (integer): The size of the collection in bytes
- `vector_count` (integer): The number of vectors in the collection
- `environment` (string): The environment (region) where the collection resides
- `dimension` (integer): The number of dimensions for vectors in the collection

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

collections = pc.list_collections()

# Get just the names
names = collections.names()
print(names)  # Output: ['collection-1', 'collection-2']

# Iterate over collections
for collection in collections:
    print(f"Collection: {collection['name']}")

# Access by index
first_collection = collections[0]
print(f"First collection: {first_collection['name']}")

# Get count
print(f"Total collections: {len(collections)}")
```

---

### `CollectionDescription`

A NamedTuple representing the description of a collection (returned by the `describe_collection()` method in some contexts, though the primary return type is `dict[str, Any]`).

**Source:** `pinecone/db_control/models/collection_description.py:4-18`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| `name` | `str` | No | v1.0 | No | The name of the collection. |
| `source` | `str` | No | v1.0 | No | The name of the index used to create the collection. |
