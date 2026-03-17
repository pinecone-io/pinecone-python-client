# Collection Operations

This module documents collection management operations on the Pinecone and PineconeAsyncio clients: creating, listing, deleting, and describing collections. Collections are serverless copies of pod-based indexes that provide a point-in-time snapshot of your data.

## Overview

**Language / runtime:** Python 3.8+
**Package:** `pinecone`
**Module:** `pinecone`
**Class:** `Pinecone` and `PineconeAsyncio`
**Version:** v8.1.0
**Breaking change definition:** Changing the return type or return value structure of any method, removing a method, or renaming a parameter.

## Methods

### `Pinecone.create_collection(name: str, source: str) -> None`

Creates a collection from a pod-based index.

**Import:** `from pinecone import Pinecone`
**Source:** `pinecone/pinecone.py:1028-1046`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent — repeated calls will fail if the collection already exists
**Side effects:** Creates a new collection resource in the Pinecone API

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name of the collection. Must be a valid collection name (alphanumeric characters, hyphens, underscores). |
| source | str | Yes | — | The name of the pod-based index to use as the source for the collection. The source index must exist and be pod-based. |

**Returns:** `None` — This method returns nothing on success.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. May occur if the `name` or `source` parameters are invalid, if a collection with the same name already exists, if the source index does not exist, or if the source index is not pod-based. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Create a collection from an existing pod-based index
pc.create_collection(name="my-collection", source="my-index")
print("Collection created successfully")
```

**Notes**

- Collections provide a serverless snapshot of your data and can be used for backup or analysis purposes.
- The source index must be pod-based; serverless indexes cannot be used as sources.
- Collection creation is asynchronous; the collection becomes available within a few moments of the API call returning.

---

### `PineconeAsyncio.create_collection(name: str, source: str) -> None`

Asynchronous version of `create_collection()`. Creates a collection from a pod-based index.

**Import:** `from pinecone import PineconeAsyncio`
**Source:** `pinecone/pinecone_asyncio.py:1110-1116`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Not idempotent
**Side effects:** Creates a new collection resource in the Pinecone API

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name of the collection. |
| source | str | Yes | — | The name of the source pod-based index. |

**Returns:** `Awaitable[None]` — An awaitable that resolves to `None` on success.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The request failed. May occur if the `name` or `source` parameters are invalid, if a collection with the same name already exists, or if the source index does not exist or is not pod-based. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

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

### `Pinecone.list_collections() -> CollectionList`

Lists all collections in your Pinecone project.

**Import:** `from pinecone import Pinecone, CollectionList`
**Source:** `pinecone/pinecone.py:1048-1068`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Returns:** `CollectionList` — An iterable collection of all collections. The `CollectionList` object wraps a list of collection items and provides a convenience method `names()` to extract just the collection names.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | Failed to retrieve the collection list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

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
```

**Notes**

- The `CollectionList` object is iterable; use it in a `for` loop to iterate over individual collection items.
- The `names()` method is a convenience method that returns a list of collection names without needing to extract the name from each item.
- Collections are returned regardless of their creation status.

---

### `PineconeAsyncio.list_collections() -> CollectionList`

Asynchronous version of `list_collections()`. Lists all collections in your Pinecone project.

**Import:** `from pinecone import PineconeAsyncio, CollectionList`
**Source:** `pinecone/pinecone_asyncio.py:1118-1147`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Returns:** `CollectionList` — A collection of all collections (when awaited).

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | Failed to retrieve the collection list from the API. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

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

### `Pinecone.delete_collection(name: str) -> None`

Deletes a collection. This is an irreversible operation.

**Import:** `from pinecone import Pinecone`
**Source:** `pinecone/pinecone.py:1070-1092`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent (idempotent in effect — calling twice will fail the second time if the collection is already deleted)
**Side effects:** Deletes the collection resource. The collection transitions to a deleted state and is eventually removed.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name of the collection to delete. |

**Returns:** `None`

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The delete operation failed, such as when the collection does not exist. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

collection_name = "old-collection"

# Delete the collection
pc.delete_collection(name=collection_name)
print(f"Collection '{collection_name}' has been deleted")
```

**Notes**

- Deleting a collection is irreversible; all data in the collection will be lost.
- Collection deletion is asynchronous; the collection becomes unavailable immediately, but cleanup may take a few moments.
- Attempting to delete a non-existent collection raises `NotFoundException`.

---

### `PineconeAsyncio.delete_collection(name: str) -> None`

Asynchronous version of `delete_collection()`. Deletes a collection.

**Import:** `from pinecone import PineconeAsyncio`
**Source:** `pinecone/pinecone_asyncio.py:1149-1171`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent (idempotent in effect)
**Side effects:** Deletes the collection and all data it contains.

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name of the collection to delete. |

**Returns:** `None` — Returns `None` on success (when awaited).

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | The delete operation failed, such as when the collection does not exist. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

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

### `Pinecone.describe_collection(name: str) -> dict[str, Any]`

Describes a specific collection by name.

**Import:** `from pinecone import Pinecone`
**Source:** `pinecone/pinecone.py:1094-1117`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name of the collection to describe. |

**Returns:** `dict[str, Any]` — A dictionary representing the collection with properties including name, status, environment, size, dimension, and vector_count.

Structure of returned object:
```python
{
    "name": "my-collection",
    "status": "Ready",
    "environment": "us-west-2",
    "size": 1024000,
    "dimension": 1536,
    "vector_count": 100
}
```

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | Failed to retrieve the collection description, such as when the collection does not exist. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

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

**Notes**

- The returned dictionary includes metadata about the collection such as its status, environment, dimension, vector count, and size.
- The `status` field indicates the operational state of the collection (e.g., "Ready", "Initializing", "Terminating").
- The `size` field represents the approximate storage size of the collection in bytes.
- The `vector_count` field shows the number of vectors in the collection.
- The `dimension` field indicates the vector dimension configured for the collection.

---

### `PineconeAsyncio.describe_collection(name: str) -> dict[str, Any]`

Asynchronous version of `describe_collection()`. Describes a specific collection by name.

**Import:** `from pinecone import PineconeAsyncio`
**Source:** `pinecone/pinecone_asyncio.py:1173-1194`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | str | Yes | — | The name of the collection to describe. |

**Returns:** `dict[str, Any]` — A dictionary representing the collection (when awaited).

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeApiException` | Failed to retrieve the collection description, such as when the collection does not exist. |
| `UnauthorizedException` | The API key is invalid or missing. |

**Example**

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

## Data Models

### `CollectionList`

An iterable collection of collections returned by the `list_collections()` method.

**Import:** `from pinecone import CollectionList`
**Source:** `pinecone/db_control/models/collection_list.py:7-36`

**Methods**

#### `CollectionList.names() -> list[str]`

Returns a list of collection names without needing to extract the name from each collection item.

**Returns:** `list[str]` — A list of collection name strings.

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

collections = pc.list_collections()
names = collections.names()
print(names)  # Output: ['collection-1', 'collection-2']
```

---

#### `CollectionList.__iter__() -> Iterator`

Returns an iterator over the collections in the list, allowing the CollectionList to be used in `for` loops.

**Returns:** `Iterator` — An iterator that yields individual collection dictionaries.

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

collections = pc.list_collections()
for collection in collections:
    print(f"Collection: {collection['name']}")
```

---

#### `CollectionList.__len__() -> int`

Returns the number of collections in the list.

**Returns:** `int` — The count of collections in the list.

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

collections = pc.list_collections()
count = len(collections)
print(f"Total collections: {count}")
```

---

#### `CollectionList.__getitem__(index: int) -> dict[str, Any]`

Accesses a specific collection by numeric position (0-based indexing).

**Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| index | integer | Yes | — | The numeric index (0-based) of the collection to retrieve. |

**Returns:** `dict[str, Any]` — The collection dictionary at the specified position.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `IndexError` | The numeric index is out of bounds. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

collections = pc.list_collections()
first_collection = collections[0]
print(f"First collection: {first_collection['name']}")
```

---

### `CollectionDescription`

A NamedTuple representing the description of a collection (returned by the `describe_collection()` method in some contexts, though the primary return type is `dict[str, Any]`).

**Import:** `from pinecone import CollectionDescription`
**Source:** `pinecone/db_control/models/collection_description.py:4-18`

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| name | str | No | The name of the collection. |
| source | str | No | The name of the index used to create the collection. |

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

## Backward Compatibility

- **v1.0**: Initial release of collection operations.
- **No breaking changes** have been introduced since initial release.
- All method signatures and return types are stable.

---

## See Also

- **Pinecone Client:** `pinecone.Pinecone`
- **PineconeAsyncio Client:** `pinecone.PineconeAsyncio`
- **Index Operations:** `spec/interfaces/sdk/index_management_operations.md`
