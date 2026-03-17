# Index Vector Write Operations

This module documents the Index class methods for writing and deleting vectors in Pinecone indexes: `upsert()` for inserting or updating vectors with dense and sparse values and metadata, and `delete()` for removing vectors by ID, metadata filter, or bulk deletion.

Vector write operations are accessed through an Index client, obtained from the Pinecone client:

```python
from pinecone import Pinecone, Vector

pc = Pinecone(api_key="your-api-key")
index = pc.Index(host="your-index-host")

# Synchronous upsert/delete
response = index.upsert(vectors=[Vector(id="vec1", values=[0.1, 0.2, 0.3])], namespace="my-namespace")
index.delete(ids=["vec1"], namespace="my-namespace")

# Asynchronous upsert/delete
async_index = pc.IndexAsyncio(host="your-index-host")
response = await async_index.upsert(vectors=[...], namespace="my-namespace")
await async_index.delete(ids=["vec1"], namespace="my-namespace")
```

---

## `Index.upsert()`

Upserts vectors into a namespace of your index. If a vector with the same ID already exists, it will be overwritten with the new values and metadata.

**Source:** `pinecone/db_data/index.py:226-442`, `pinecone/db_data/index_asyncio.py:292-467` (async equivalent)

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — upserting the same vectors multiple times produces the same result
**Side effects:** Creates or updates vectors in the index namespace

### Signature

```python
def upsert(
    self,
    vectors: list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict],
    namespace: str | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
    **kwargs
) -> UpsertResponse | ApplyResult
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `vectors` | `list[Vector] \| list[VectorTuple] \| list[VectorTupleWithMetadata] \| list[VectorTypedDict]` | Yes | — | v8.0 | No | List of vectors to upsert. Each vector must have a unique `id` within the namespace. Vectors can be provided as: `Vector` objects, 2-tuples `(id, values)` for dense vectors without metadata, 3-tuples `(id, values, metadata)` for dense vectors with metadata, or dictionaries with `id`, `values`, and optional `metadata` and `sparse_values` keys. |
| `namespace` | `str` | No | `""` (default namespace) | v8.0 | No | The namespace to write vectors to. If not specified, vectors are upserted into the default empty-string namespace. |
| `batch_size` | `int` | No | `None` | v8.0 | No | Number of vectors to upsert in each batch. If not specified, all vectors are upserted in a single request. When specified, vectors are sent in batches with the `tqdm` progress bar shown if `show_progress=True`. Cannot be used with `async_req=True`. |
| `show_progress` | `bool` | No | `True` | v8.0 | No | Whether to display a progress bar using `tqdm` when upserting in batches. Only applied when `batch_size` is provided. Requires `tqdm` to be installed. |
| `**kwargs` | `dict` | No | — | v8.0 | No | Additional keyword arguments, including `async_req` (bool) to execute the request asynchronously. |

### Vector Format Details

Vectors can include dense values, sparse values, or both:

- **Dense vectors:** Represented as a `list[float]`. The dimension must match the index's configured dimension.
- **Sparse vectors:** Represented as a `SparseValues` object with `indices` (list[int]) and `values` (list[float])
- **Metadata:** Arbitrary JSON-serializable object (typically a dict) associated with the vector for filtering.

Each vector in the list must have a unique `id` within the namespace. Attempting to upsert vectors with duplicate IDs within a single batch will result in undefined behavior (likely only the last vector with that ID will be stored).

### Returns

**Type:** `UpsertResponse` — A response object containing:
- `upserted_count` (int) — The number of vectors successfully upserted.
- When `async_req=True`, returns `ApplyResult` which wraps the response and can be awaited with `.get()`.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `batch_size` is specified with `async_req=True`. |
| `ValueError` | `batch_size` is not a positive integer. |
| `VectorDictionaryMissingKeysError` | A vector dictionary is missing required keys (`id` and either `values` or `sparse_values`). |
| `VectorDictionaryExcessKeysError` | A vector dictionary contains unrecognized keys. |
| `VectorTupleLengthError` | A vector tuple does not have 2 or 3 elements. |
| `SparseValuesTypeError` | `sparse_values` is not a `SparseValues` object or dict. |
| `SparseValuesMissingKeysError` | `sparse_values` is missing required keys (`indices` or `values`). |
| `SparseValuesDictionaryExpectedError` | `sparse_values` is not a dict when provided as a dict format. |
| `ProtocolError` | Connection failed to the index host. |
| `PineconeApiException` | Vector dimension mismatch, invalid metadata, or other validation/server error. |

### Example

```python
from pinecone import Pinecone, Vector, SparseValues

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index(host="example-index-host")

# Upsert dense vectors with metadata
response = index.upsert(
    vectors=[
        Vector(
            id="vector-1",
            values=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "document-1", "date": "2024-01-01"}
        ),
        Vector(
            id="vector-2",
            values=[0.5, 0.6, 0.7, 0.8],
            metadata={"source": "document-2", "date": "2024-01-02"}
        ),
    ],
    namespace="documents"
)
print(f"Upserted {response.upserted_count} vectors")

# Upsert sparse vectors for hybrid search
response = index.upsert(
    vectors=[
        Vector(
            id="sparse-1",
            sparse_values=SparseValues(
                indices=[1, 5, 8],
                values=[0.2, 0.4, 0.6]
            ),
            metadata={"type": "sparse"}
        ),
    ],
    namespace="hybrid-search"
)

# Upsert in batches for large datasets
vectors = [
    Vector(id=f"batch-{i}", values=[float(i) / 100 for _ in range(10)])
    for i in range(10000)
]
response = index.upsert(vectors=vectors, batch_size=100, namespace="large-batch")

# Using tuple format (shorthand)
response = index.upsert(
    vectors=[
        ("tuple-vec-1", [0.1, 0.2, 0.3]),
        ("tuple-vec-2", [0.4, 0.5, 0.6], {"label": "example"})
    ],
    namespace="tuples"
)
```

### Notes

- The upsert operation is idempotent: upserting the same vectors multiple times produces the same result.
- Vector IDs must be unique within a namespace. If you upsert a vector with an ID that already exists, the old vector is completely replaced.
- When `batch_size` is provided, vectors are sent in multiple requests, each up to `batch_size` vectors. The progress bar (if shown) tracks progress across all batches.
- The `async_req` parameter cannot be combined with `batch_size`. To upsert in parallel, use multiple async tasks or follow the parallel upsert pattern in the Pinecone documentation.
- Metadata is optional and can be any JSON-serializable object. It is useful for filtering results in `query()` and `search()` operations.

---

## `Index.delete()`

Deletes vectors from the index within a single namespace. Supports deletion by vector ID, deletion of all vectors in a namespace, or deletion by metadata filter. Does not raise an error if a vector ID does not exist.

**Source:** `pinecone/db_data/index.py:762-824`, `pinecone/db_data/index_asyncio.py:512-598` (async equivalent)

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — deleting the same vectors multiple times is safe (no error if vectors do not exist)
**Side effects:** Removes vectors from the index namespace

### Signature

```python
def delete(
    self,
    ids: list[str] | None = None,
    delete_all: bool | None = None,
    namespace: str | None = None,
    filter: FilterTypedDict | None = None,
    **kwargs
) -> dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `ids` | `list[str]` | No | `None` | v8.0 | No | List of vector IDs to delete from the namespace. Mutually exclusive with `delete_all` and `filter`. If none of the three parameters are specified, the method raises an error. |
| `delete_all` | `bool` | No | `None` | v8.0 | No | If `True`, deletes all vectors from the namespace. Mutually exclusive with `ids` and `filter`. Must be explicitly set to `True`; the default is `None` (not `False`). |
| `namespace` | `str` | No | `""` (default namespace) | v8.0 | No | The namespace to delete vectors from. If not specified, the default empty-string namespace is used. **Note:** No error is raised if the namespace does not exist. |
| `filter` | `FilterTypedDict` | No | `None` | v8.0 | No | Metadata filter expression to select vectors for deletion. Vectors matching the filter are deleted from the namespace. Mutually exclusive with `ids` and `delete_all`. See metadata filtering documentation for filter syntax. |
| `**kwargs` | `dict` | No | — | v8.0 | No | Additional keyword arguments for the API call. |

### Deletion Modes

Delete supports three mutually exclusive deletion modes:

1. **Delete by IDs:** Specify a list in `ids` to delete specific vectors. If an ID does not exist, no error is raised.
2. **Delete all:** Set `delete_all=True` to delete all vectors in the namespace. This is an irreversible operation on that namespace.
3. **Delete by filter:** Specify a `filter` to delete all vectors matching the metadata filter condition.

At least one of `ids`, `delete_all`, or `filter` must be specified. If none are provided or if multiple are specified simultaneously (e.g., both `ids` and `delete_all`), the behavior is undefined or an error may be raised.

### Returns

**Type:** `dict[str, Any]` — An empty dictionary `{}` on success. No metadata about deleted vector count is returned.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ProtocolError` | Connection failed to the index host. |
| `PineconeApiException` | Invalid filter expression, other validation error, or unexpected server error. |

### Example

```python
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index(host="example-index-host")

# Delete specific vectors by ID
result = index.delete(ids=["vector-1", "vector-2"], namespace="documents")
print(result)  # {}

# Delete all vectors from a namespace
result = index.delete(delete_all=True, namespace="documents")
print(result)  # {}

# Delete vectors matching a metadata filter
result = index.delete(
    filter={"source": "old-documents", "archived": True},
    namespace="documents"
)
print(result)  # {}

# Delete from the default namespace
result = index.delete(ids=["default-vec-1"])  # namespace defaults to ""
```

### Notes

- The delete operation is idempotent: deleting the same vector IDs multiple times does not raise an error if the vectors no longer exist.
- If you delete from the wrong namespace by accident (e.g., if `namespace` is not specified), you may not receive an error because the operation silently succeeds when vectors are not found.
- The `delete_all=True` operation is irreversible and permanently removes all vectors from the namespace. Use with caution.
- No information is returned about the number of vectors deleted. If you need to know how many vectors were deleted, you must track this separately in your application logic.
- Deletion by filter is asynchronous on the server side: the API returns immediately, but vectors may take a moment to be fully removed from indexes.
- At least one of `ids`, `delete_all`, or `filter` should be specified. If none are provided, the server request may contain no deletion parameters, but validation is not enforced on the client side.
