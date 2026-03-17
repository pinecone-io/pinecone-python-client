# Collection Operations

Methods for managing collections via the `Pinecone` and `PineconeAsyncio` client instances. Collections are snapshots of pod-based indexes that can be used to preserve data at a point in time.

---

## Pinecone.create_collection

Creates a new collection from a pod-based index.

**Source:** `pinecone/pinecone.py:1028-1046`, `pinecone/pinecone_asyncio.py:1110-1116` (async equivalent)

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Creates a new collection; the source index remains unchanged and usable.

### Signature

```python
def create_collection(self, name: str, source: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the collection to create. Must be unique within the workspace. |
| `source` | `string` | Yes | — | v1.0 | No | The name of a pod-based index to create the collection from. The index must exist and be pod-based (serverless indexes are not supported). |

### Returns

**Type:** `None`

Returns `None` on success. The collection is created asynchronously; use `describe_collection()` to poll for status.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `name` is an empty string. |
| `UnauthorizedException` | The API key is missing or invalid. |
| `ForbiddenException` | The user lacks permission to create collections. |
| `NotFoundException` | The source index does not exist. |
| `pinecone.PineconeApiException` | The source index is serverless (not pod-based), or another API error occurs. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Create a collection from an existing pod-based index
pc.create_collection(name="my_collection", source="my_index")
```

### Notes

- Collections can only be created from pod-based indexes. Serverless indexes cannot be used as sources.
- The source index remains fully operational; collections are read-only snapshots.
- Collection creation is asynchronous. The collection may not be immediately available for use.

---

## Pinecone.describe_collection

Retrieves metadata about a single collection.

**Source:** `pinecone/pinecone.py:1094-1117`, `pinecone/pinecone_asyncio.py:1173-1195` (async equivalent)

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation).

### Signature

```python
def describe_collection(self, name: str) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the collection to describe. |

### Returns

**Type:** `dict[str, Any]`

A dictionary containing collection metadata with the following fields:

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
| `ValueError` | `name` is an empty string. |
| `UnauthorizedException` | The API key is missing or invalid. |
| `ForbiddenException` | The user lacks permission to describe collections. |
| `NotFoundException` | The collection does not exist. |
| `pinecone.PineconeApiException` | Another API error occurs. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

description = pc.describe_collection(name="my_collection")
print(f"Collection: {description['name']}")
print(f"Status: {description['status']}")
print(f"Dimension: {description['dimension']}")
print(f"Vector count: {description['vector_count']}")
print(f"Size: {description['size']} bytes")
```

### Notes

- Collections in `Initializing` status are being created from the source index and are not yet ready for use.
- The `vector_count` and `size` reflect the snapshot state at the time of collection creation.

---

## Pinecone.list_collections

Retrieves a list of all collections in the workspace.

**Source:** `pinecone/pinecone.py:1048-1068`, `pinecone/pinecone_asyncio.py:1118-1147` (async equivalent)

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Safe to retry
**Side effects:** None (read-only operation).

### Signature

```python
def list_collections(self) -> CollectionList
```

### Parameters

This method takes no parameters.

### Returns

**Type:** `CollectionList`

A `CollectionList` object containing all collections in the workspace. The object is iterable and supports the following methods and operations:

| Method/Operation | Description |
|------------------|-------------|
| `names()` | Returns a list of collection names (strings). |
| `__iter__()` | Iterate over individual collection objects. |
| `__len__()` | Get the number of collections. |
| `__getitem__(index)` | Access a collection by index. |

Each collection object has the following fields:
- `name` (string): The name of the collection
- `status` (string): The collection status (`Initializing` or `Ready`)
- `size` (integer): The size of the collection in bytes
- `vector_count` (integer): The number of vectors in the collection
- `environment` (string): The environment (region) where the collection resides
- `dimension` (integer): The number of dimensions for vectors in the collection

### Raises

| Exception | Condition |
|-----------|-----------|
| `UnauthorizedException` | The API key is missing or invalid. |
| `ForbiddenException` | The user lacks permission to list collections. |
| `pinecone.PineconeApiException` | Another API error occurs. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# List all collections
collections = pc.list_collections()

# Iterate over all collections
for collection in collections:
    print(f"Collection: {collection['name']}, Status: {collection['status']}")

# Get just the names
for name in collections.names():
    print(f"Collection name: {name}")

# Get the number of collections
print(f"Total collections: {len(collections)}")
```

### Notes

- The `CollectionList` provides convenient access to all collections via iteration, indexing, and the `names()` method.
- All collections are returned; there is no pagination or filtering.

---

## Pinecone.delete_collection

Deletes a collection.

**Source:** `pinecone/pinecone.py:1070-1092`, `pinecone/pinecone_asyncio.py:1149-1171` (async equivalent)

**Added:** v1.0
**Deprecated:** No
**Idempotency:** Non-idempotent
**Side effects:** Deletes the specified collection. This operation is irreversible.

### Signature

```python
def delete_collection(self, name: str) -> None
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|-----------|-------------|
| `name` | `string` | Yes | — | v1.0 | No | The name of the collection to delete. |

### Returns

**Type:** `None`

Returns `None` on success. The deletion request is sent to the server, and the collection may take a few moments to be fully removed.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `name` is an empty string. |
| `UnauthorizedException` | The API key is missing or invalid. |
| `ForbiddenException` | The user lacks permission to delete collections. |
| `NotFoundException` | The collection does not exist. |
| `pinecone.PineconeApiException` | Another API error occurs. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Delete a collection
pc.delete_collection(name="my_collection")

# Verify deletion by checking that describe_collection raises NotFoundException
try:
    pc.describe_collection(name="my_collection")
except NotFoundException:
    print("Collection has been deleted")
```

### Notes

- Deleting a collection is an irreversible operation. All data in the collection will be lost.
- The deletion is asynchronous; the collection may not be immediately removed.
- After sending a deletion request, use `describe_collection()` to verify the collection has been fully deleted (it will raise `NotFoundException` once deletion completes).
