# Index Vector Write Operations

This module documents the Index and IndexAsyncio class methods for writing, updating, and deleting vectors in Pinecone indexes: `upsert()` for inserting or updating vectors with dense and sparse values and metadata, `delete()` for removing vectors by ID, metadata filter, or bulk deletion, `update()` for modifying existing vectors by ID or by metadata filter, `upsert_from_dataframe()` for bulk upserting from pandas DataFrames, and `upsert_records()` for upserting records to indexes with integrated inference.

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

---

## `IndexAsyncio.upsert()`

Async variant of `Index.upsert()`. Upserts vectors into a namespace of your index. If a vector with the same ID already exists, it will be overwritten with the new values and metadata. When `batch_size` is specified, batches are dispatched concurrently using `asyncio.as_completed`.

**Source:** `pinecone/db_data/index_asyncio.py:293-467`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — upserting the same vectors multiple times produces the same result
**Side effects:** Creates or updates vectors in the index namespace

### Signature

```python
async def upsert(
    self,
    vectors: list[Vector] | list[VectorTuple] | list[VectorTupleWithMetadata] | list[VectorTypedDict],
    namespace: str | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
    **kwargs
) -> UpsertResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `vectors` | `list[Vector] \| list[VectorTuple] \| list[VectorTupleWithMetadata] \| list[VectorTypedDict]` | Yes | — | v8.0 | No | List of vectors to upsert. Accepts the same formats as the synchronous `Index.upsert()`: `Vector` objects, 2-tuples, 3-tuples, or typed dictionaries. |
| `namespace` | `str` | No | `""` (default namespace) | v8.0 | No | The namespace to write vectors to. If not specified, vectors are upserted into the default empty-string namespace. |
| `batch_size` | `int` | No | `None` | v8.0 | No | Number of vectors to upsert in each batch. When specified, batches are dispatched concurrently using `asyncio.as_completed` and a `tqdm` progress bar is shown if `show_progress=True`. |
| `show_progress` | `bool` | No | `True` | v8.0 | No | Whether to display a progress bar using `tqdm` when upserting in batches. Only applied when `batch_size` is provided. |
| `**kwargs` | `dict` | No | — | v8.0 | No | Additional keyword arguments for the API call. |

### Returns

**Type:** `UpsertResponse` — A response object containing:
- `upserted_count` (int) — The total number of vectors successfully upserted across all batches.

Note: Unlike the synchronous variant, `async_req` is not supported. The async variant never returns `ApplyResult`.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `batch_size` is not a positive integer. |
| `VectorDictionaryMissingKeysError` | A vector dictionary is missing required keys. |
| `VectorDictionaryExcessKeysError` | A vector dictionary contains unrecognized keys. |
| `VectorTupleLengthError` | A vector tuple does not have 2 or 3 elements. |
| `SparseValuesTypeError` | `sparse_values` is not a `SparseValues` object or dict. |
| `SparseValuesMissingKeysError` | `sparse_values` is missing required keys. |
| `ProtocolError` | Connection failed to the index host. |
| `PineconeApiException` | Vector dimension mismatch, invalid metadata, or other validation/server error. |

### Example

```python
import asyncio
from pinecone import Pinecone, Vector

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio(host="example-index-host") as index:
        # Single upsert
        response = await index.upsert(
            vectors=[
                Vector(id="vec-1", values=[0.1, 0.2, 0.3, 0.4]),
                Vector(id="vec-2", values=[0.5, 0.6, 0.7, 0.8]),
            ],
            namespace="documents"
        )
        print(f"Upserted {response.upserted_count} vectors")

        # Batch upsert with concurrent dispatch
        vectors = [
            Vector(id=f"batch-{i}", values=[float(i) / 100 for _ in range(4)])
            for i in range(10000)
        ]
        response = await index.upsert(vectors=vectors, batch_size=100, namespace="large-batch")

asyncio.run(main())
```

### Notes

- Unlike the synchronous variant, the `async_req` parameter is not supported. Use `asyncio` concurrency patterns instead.
- When `batch_size` is specified, batches are dispatched concurrently via `asyncio.as_completed`, which may result in out-of-order completion. The aggregated `UpsertResponse` uses `_response_info` from the last completed batch.
- Must be used within an `async with` block or with explicit `await index.close()` to properly clean up resources.

---

## `IndexAsyncio.delete()`

Async variant of `Index.delete()`. Deletes vectors from the index within a single namespace. Supports the same three deletion modes as the synchronous variant: by ID, by filter, or delete all.

**Source:** `pinecone/db_data/index_asyncio.py:512-598`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — deleting the same vectors multiple times is safe
**Side effects:** Removes vectors from the index namespace

### Signature

```python
async def delete(
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
| `ids` | `list[str]` | No | `None` | v8.0 | No | List of vector IDs to delete. Mutually exclusive with `delete_all` and `filter`. |
| `delete_all` | `bool` | No | `None` | v8.0 | No | If `True`, deletes all vectors from the namespace. Mutually exclusive with `ids` and `filter`. |
| `namespace` | `str` | No | `""` (default namespace) | v8.0 | No | The namespace to delete vectors from. |
| `filter` | `FilterTypedDict` | No | `None` | v8.0 | No | Metadata filter expression to select vectors for deletion. Mutually exclusive with `ids` and `delete_all`. |
| `**kwargs` | `dict` | No | — | v8.0 | No | Additional keyword arguments for the API call. |

### Returns

**Type:** `dict[str, Any]` — An empty dictionary `{}` on success.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ProtocolError` | Connection failed to the index host. |
| `PineconeApiException` | Invalid filter expression or other server error. |

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio(host="example-index-host") as index:
        # Delete specific vectors by ID
        await index.delete(ids=["vec-1", "vec-2"], namespace="documents")

        # Delete all vectors from a namespace
        await index.delete(delete_all=True, namespace="documents")

        # Delete by metadata filter
        await index.delete(
            filter={"source": "old-documents"},
            namespace="documents"
        )

asyncio.run(main())
```

### Notes

- Identical behavior to the synchronous `Index.delete()`. See that method's notes for details on deletion modes and idempotency.
- Must be used within an `async with` block or with explicit `await index.close()` to properly clean up resources.

---

## `Index.update()`

Updates vectors in a namespace. Supports two modes: single vector update by ID (updating values, sparse values, and/or metadata), and bulk update by metadata filter (updating metadata on all matching vectors). Metadata updates are merged, not replaced -- only specified fields are overwritten.

**Source:** `pinecone/db_data/index.py:1222-1371`, `pinecone/db_data/index_asyncio.py:1018-1177` (async equivalent)

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent for single-vector updates by ID; bulk filter updates are also idempotent if the same metadata values are applied
**Side effects:** Modifies vector values and/or metadata in the index namespace

### Signature

```python
def update(
    self,
    id: str | None = None,
    values: list[float] | None = None,
    set_metadata: VectorMetadataTypedDict | None = None,
    namespace: str | None = None,
    sparse_values: SparseValues | SparseVectorTypedDict | None = None,
    filter: FilterTypedDict | None = None,
    dry_run: bool | None = None,
    **kwargs
) -> UpdateResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `id` | `str` | Conditionally | `None` | v8.0 | No | The unique ID of the vector to update. Required for single vector updates. Mutually exclusive with `filter`. |
| `values` | `list[float]` | No | `None` | v8.0 | No | New dense vector values. Overwrites the existing values entirely. Only valid with `id` mode. |
| `set_metadata` | `VectorMetadataTypedDict` | No | `None` | v8.0 | No | Metadata to merge with existing metadata. Specified fields overwrite existing fields with the same key; unspecified fields remain unchanged. |
| `namespace` | `str` | No | `""` (default namespace) | v8.0 | No | The namespace containing the vector(s) to update. |
| `sparse_values` | `SparseValues \| SparseVectorTypedDict` | No | `None` | v8.0 | No | Sparse values to set on the vector. Accepts a `SparseValues` object or a dict `{'indices': list[int], 'values': list[float]}`. Only valid with `id` mode. |
| `filter` | `FilterTypedDict` | Conditionally | `None` | v8.0 | No | Metadata filter expression to select vectors for bulk update. Mutually exclusive with `id`. When provided, all matching vectors have `set_metadata` applied. |
| `dry_run` | `bool` | No | `None` | v8.0 | No | If `True`, returns the count of matching records without executing the update. Only meaningful with `filter` mode. |
| `**kwargs` | `dict` | No | — | v8.0 | No | Additional keyword arguments for the API call. |

### Update Modes

Update supports two mutually exclusive modes. Exactly one of `id` or `filter` must be provided:

1. **Single vector update by ID:** Provide `id` to update a specific vector's values, sparse values, and/or metadata.
2. **Bulk update by filter:** Provide `filter` to update metadata on all matching vectors. The `values` and `sparse_values` parameters are not applicable in this mode.

### Returns

**Type:** `UpdateResponse` — A response object containing:
- `matched_records` (`int | None`) — The number of vectors that matched the filter. Present when using `filter` mode; `None` for single-vector `id` updates.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Neither `id` nor `filter` is provided. |
| `ValueError` | Both `id` and `filter` are provided in the same call. |
| `ProtocolError` | Connection failed to the index host. |
| `PineconeApiException` | Vector ID not found, invalid metadata, dimension mismatch, or other server error. |

### Example

```python
from pinecone import Pinecone, SparseValues

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index(host="example-index-host")

# Update vector values by ID
index.update(id="vec-1", values=[0.1, 0.2, 0.3], namespace="documents")

# Update metadata only (merge with existing)
index.update(id="vec-1", set_metadata={"status": "reviewed"}, namespace="documents")

# Update values and sparse values together
index.update(
    id="vec-1",
    values=[0.1, 0.2, 0.3],
    sparse_values=SparseValues(indices=[1, 5], values=[0.2, 0.4]),
    namespace="documents"
)

# Bulk update metadata by filter
response = index.update(
    set_metadata={"status": "active"},
    filter={"genre": {"$eq": "drama"}},
    namespace="documents"
)
print(f"Updated {response.matched_records} vectors")

# Dry run to preview how many vectors would be affected
response = index.update(
    set_metadata={"status": "active"},
    filter={"genre": {"$eq": "drama"}},
    namespace="documents",
    dry_run=True
)
print(f"Would update {response.matched_records} vectors")
```

### Notes

- Metadata updates are **merged**, not replaced. Only the fields specified in `set_metadata` are overwritten; all other existing metadata fields remain unchanged.
- The `id` and `filter` parameters are mutually exclusive. Providing both raises a `ValueError` on the client side before any API call is made.
- For single-vector updates, if the vector ID does not exist, the server may return an error (unlike delete, which is silent).
- The `dry_run` parameter is only meaningful with `filter` mode. Using it with `id` has no effect.

---

## `IndexAsyncio.update()`

Async variant of `Index.update()`. Updates vectors in a namespace with the same two modes (by ID or by filter) and identical parameters.

**Source:** `pinecone/db_data/index_asyncio.py:1018-1177`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Same as `Index.update()`
**Side effects:** Modifies vector values and/or metadata in the index namespace

### Signature

```python
async def update(
    self,
    id: str | None = None,
    values: list[float] | None = None,
    set_metadata: VectorMetadataTypedDict | None = None,
    namespace: str | None = None,
    sparse_values: SparseValues | SparseVectorTypedDict | None = None,
    filter: FilterTypedDict | None = None,
    dry_run: bool | None = None,
    **kwargs
) -> UpdateResponse
```

### Parameters

See [`Index.update()` Parameters](#parameters-2) — all parameters are identical.

### Returns

**Type:** `UpdateResponse` — Same as `Index.update()`. Contains `matched_records` when using filter mode.

### Raises

Same as `Index.update()`.

### Example

```python
import asyncio
from pinecone import Pinecone, SparseValues

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio(host="example-index-host") as index:
        # Update vector values by ID
        await index.update(id="vec-1", values=[0.1, 0.2, 0.3], namespace="documents")

        # Bulk update metadata by filter
        response = await index.update(
            set_metadata={"status": "active"},
            filter={"genre": {"$eq": "drama"}},
            namespace="documents"
        )
        print(f"Updated {response.matched_records} vectors")

        # Dry run
        response = await index.update(
            set_metadata={"status": "active"},
            filter={"genre": {"$eq": "drama"}},
            namespace="documents",
            dry_run=True
        )
        print(f"Would update {response.matched_records} vectors")

asyncio.run(main())
```

### Notes

- Identical behavior to `Index.update()`. See that method's notes for details.
- Must be used within an `async with` block or with explicit `await index.close()`.

---

## `Index.upsert_from_dataframe()`

Upserts vectors from a pandas DataFrame into the index. The DataFrame must contain an `id` column and a `values` column, and may optionally contain `sparse_values` and `metadata` columns. Vectors are sent in batches with a progress bar.

**Source:** `pinecone/db_data/index.py:483-567`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — upserting the same DataFrame multiple times produces the same result
**Side effects:** Creates or updates vectors in the index namespace

### Signature

```python
def upsert_from_dataframe(
    self,
    df,
    namespace: str | None = None,
    batch_size: int = 500,
    show_progress: bool = True
) -> UpsertResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `df` | `pandas.DataFrame` | Yes | — | v8.0 | No | A pandas DataFrame with columns: `id` (str), `values` (list[float]), and optionally `sparse_values` and `metadata`. Each row represents one vector. |
| `namespace` | `str` | No | `""` (default namespace) | v8.0 | No | The namespace to upsert into. |
| `batch_size` | `int` | No | `500` | v8.0 | No | Number of rows to upsert per batch. |
| `show_progress` | `bool` | No | `True` | v8.0 | No | Whether to display a `tqdm` progress bar during the upsert. |

### Returns

**Type:** `UpsertResponse` — A response object containing:
- `upserted_count` (int) — The total number of vectors successfully upserted across all batches.

### Raises

| Exception | Condition |
|-----------|-----------|
| `RuntimeError` | The `pandas` package is not installed. |
| `ValueError` | The `df` argument is not a `pandas.DataFrame`. |
| `ProtocolError` | Connection failed to the index host. |
| `PineconeApiException` | Vector dimension mismatch, invalid metadata, or other server error. |

### Example

```python
import pandas as pd
from pinecone import Pinecone

pc = Pinecone(api_key="sk-example-key-do-not-use")
index = pc.Index(host="example-index-host")

df = pd.DataFrame({
    "id": ["vec-1", "vec-2", "vec-3"],
    "values": [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ],
    "metadata": [
        {"source": "doc-1"},
        {"source": "doc-2"},
        {"source": "doc-3"}
    ]
})

response = index.upsert_from_dataframe(
    df=df,
    namespace="documents",
    batch_size=100,
    show_progress=True
)
print(f"Upserted {response.upserted_count} vectors")
```

### Notes

- Requires `pandas` to be installed separately. If not installed, a `RuntimeError` is raised with an installation hint.
- Batches are sent sequentially (not concurrently). Each batch calls `self.upsert()` internally.
- The `async_req` parameter is not used internally, so all batch calls are synchronous.
- The aggregated `UpsertResponse` uses `_response_info` from the final batch.

---

## `IndexAsyncio.upsert_from_dataframe()`

Async stub that raises `NotImplementedError`. This method has not been implemented for the `IndexAsyncio` class.

**Source:** `pinecone/db_data/index_asyncio.py:506-510`

**Added:** v8.0
**Deprecated:** No

### Signature

```python
async def upsert_from_dataframe(
    self,
    df,
    namespace: str | None = None,
    batch_size: int = 500,
    show_progress: bool = True
)
```

### Raises

| Exception | Condition |
|-----------|-----------|
| `NotImplementedError` | Always raised. This method is not implemented for `IndexAsyncio`. |

### Notes

- Use the synchronous `Index.upsert_from_dataframe()` or convert data manually and call `IndexAsyncio.upsert()` instead.

---

## `Index.upsert_records()`

Upserts records into a namespace of an index configured with integrated inference (embedding). Each record is a dictionary containing an `id` or `_id` field and additional fields that map to the index's embed configuration. Pinecone converts the mapped fields into embeddings server-side.

**Source:** `pinecone/db_data/index.py:569-658`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — upserting the same records multiple times produces the same result
**Side effects:** Creates or updates records and their embeddings in the index namespace

### Signature

```python
def upsert_records(
    self,
    namespace: str,
    records: list[dict]
) -> UpsertResponse
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `namespace` | `str` | Yes | — | v8.0 | No | The namespace to upsert records into. |
| `records` | `list[dict]` | Yes | — | v8.0 | No | List of record dictionaries. Each must contain an `id` or `_id` field as the unique identifier. Other fields should correspond to the field mappings in the index's embed configuration. |

### Returns

**Type:** `UpsertResponse` — A response object containing:
- `upserted_count` (int) — The number of records upserted (set to `len(records)` on the client side, assuming all succeed).

### Raises

| Exception | Condition |
|-----------|-----------|
| `ProtocolError` | Connection failed to the index host. |
| `PineconeApiException` | Missing required fields, index not configured for integrated inference, or other server error. |

### Example

```python
from pinecone import Pinecone, CloudProvider, AwsRegion, EmbedModel, IndexEmbed

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Create an index with integrated inference
index_model = pc.create_index_for_model(
    name="my-model-index",
    cloud=CloudProvider.AWS,
    region=AwsRegion.US_WEST_2,
    embed=IndexEmbed(
        model=EmbedModel.Multilingual_E5_Large,
        field_map={"text": "my_text_field"}
    )
)

index = pc.Index(host=index_model.host)

# Upsert records -- Pinecone generates embeddings server-side
response = index.upsert_records(
    namespace="articles",
    records=[
        {"_id": "rec-1", "my_text_field": "Apple is a popular fruit."},
        {"_id": "rec-2", "my_text_field": "The tech company Apple makes iPhones."},
        {"_id": "rec-3", "my_text_field": "Many people enjoy eating apples."},
    ]
)
print(f"Upserted {response.upserted_count} records")
```

### Notes

- This method is designed for indexes created with `create_index_for_model()` that have an embed configuration with field mappings. Using it on a standard index (without integrated inference) will result in a server error.
- Each record must contain either an `id` or `_id` field. At least one other field must match a field mapping in the index's embed configuration.
- The `upserted_count` in the response is set to `len(records)` on the client side. The server does not return a count of individually successful upserts.
- Unlike `upsert()`, this method does not support `batch_size` or `show_progress` parameters. All records are sent in a single request.

---

## `IndexAsyncio.upsert_records()`

Async variant of `Index.upsert_records()`. Upserts records into a namespace of an index configured with integrated inference. Identical parameters and behavior to the synchronous version.

**Source:** `pinecone/db_data/index_asyncio.py:1294-1398`

**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent
**Side effects:** Creates or updates records and their embeddings in the index namespace

### Signature

```python
async def upsert_records(
    self,
    namespace: str,
    records: list[dict]
) -> UpsertResponse
```

### Parameters

See [`Index.upsert_records()` Parameters](#parameters-5) — all parameters are identical.

### Returns

**Type:** `UpsertResponse` — Same as `Index.upsert_records()`.

### Raises

Same as `Index.upsert_records()`.

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")
    async with pc.IndexAsyncio(host="example-index-host") as index:
        response = await index.upsert_records(
            namespace="articles",
            records=[
                {"_id": "rec-1", "my_text_field": "Apple is a popular fruit."},
                {"_id": "rec-2", "my_text_field": "The tech company Apple makes iPhones."},
            ]
        )
        print(f"Upserted {response.upserted_count} records")

asyncio.run(main())
```

### Notes

- Identical behavior to `Index.upsert_records()`. See that method's notes for details.
- Must be used within an `async with` block or with explicit `await index.close()`.
