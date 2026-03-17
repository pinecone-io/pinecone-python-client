# Index Read Operations

This module documents the Index class methods for reading vectors from a Pinecone index: `fetch()` for retrieving vectors by ID, and `fetch_by_metadata()` for retrieving vectors matching a metadata filter.

## Overview

**Language / runtime:** Python 3.9+
**Package:** `pinecone`
**Module:** `pinecone.db_data.index`
**Class:** `Index` (sync) and `AsyncIndex` (async)
**Version:** v8.1.0
**Breaking change definition:** Changing the return type or return value structure of any method, removing a method, or renaming a parameter.

## Access Pattern

Vector read operations are accessed through an Index client, obtained from the Pinecone client:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index(host="your-index-host")

# Synchronous read operations
response = index.fetch(ids=["vec1", "vec2"], namespace="my-namespace")
for vector_id, vector in response.vectors.items():
    print(vector.values)

# Fetch by metadata filter
metadata_response = index.fetch_by_metadata(
    filter={"genre": {"$in": ["comedy", "drama"]}},
    namespace="my-namespace"
)

# Asynchronous read operations
async_index = pc.IndexAsyncio(host="your-index-host")
async_response = await async_index.fetch(ids=["vec1", "vec2"])
```

## Methods

### `Index.fetch(ids: list[str], namespace: str | None = None, **kwargs) -> FetchResponse`

Retrieves vectors by their IDs from a single namespace.

**Source:** `pinecone/db_data/index.py:827-856`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — fetching the same IDs multiple times returns the same vectors
**Side effects:** None (read-only operation)
**Async variant:** `async def fetch(...)` in `pinecone/db_data/index_asyncio.py:601-637`

**Parameters**

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| ids | array of string | Yes | — | v8.0 | No | The vector IDs to fetch. |
| namespace | string \| None | No | None | v8.0 | No | The namespace to fetch vectors from. When `None`, the default namespace is used. |

**Returns:** `FetchResponse` — An object containing the fetched vectors and the namespace name. The `vectors` field is a dictionary mapping vector IDs to `Vector` objects. If a requested ID does not exist in the index, it is not included in the response.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeException` | The API call fails due to network issues, authentication failure, or server errors. |
| `ValueError` | The `ids` parameter is empty or not a list of strings. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index(host="your-index-host")

# Fetch specific vectors
response = index.fetch(ids=["doc-001", "doc-002"], namespace="production-embeddings")
print(f"Fetched {len(response.vectors)} vectors from namespace '{response.namespace}'")

for vector_id, vector in response.vectors.items():
    print(f"Vector {vector_id}: {vector.values[:3]}...")  # Print first 3 dimensions
    if vector.metadata:
        print(f"  Metadata: {vector.metadata}")
```

**Notes**

- Non-existent vector IDs are silently ignored; the response contains only vectors that exist in the index.
- The returned `Vector` objects include both dense `values` and any `metadata` stored with the vector.
- If no vectors are found for any of the requested IDs, an empty `vectors` dictionary is returned, not an error.
- The `namespace` field in the response reflects the actual namespace queried (defaults to the index's default namespace if not specified).

---

### `Index.fetch_by_metadata(filter: FilterTypedDict, namespace: str | None = None, limit: int | None = None, pagination_token: str | None = None, **kwargs) -> FetchByMetadataResponse`

Retrieves vectors matching a metadata filter expression from a single namespace.

**Source:** `pinecone/db_data/index.py:859-934`
**Added:** v8.1
**Deprecated:** No
**Idempotency:** Idempotent — executing the same filter multiple times returns the same vectors (assuming no concurrent writes)
**Side effects:** None (read-only operation)
**Async variant:** `async def fetch_by_metadata(...)` in `pinecone/db_data/index_asyncio.py:639-726`

**Parameters**

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| filter | FilterTypedDict | Yes | — | v8.1 | No | A metadata filter expression to select vectors. Supports MongoDB-style query operators such as `$eq`, `$in`, `$gt`, `$lt`, `$ne`, `$and`, `$or`. See [metadata filtering documentation](https://docs.pinecone.io/docs/metadata-filtering). |
| namespace | string \| None | No | None | v8.1 | No | The namespace to fetch vectors from. When `None`, the default namespace is used. |
| limit | integer (int32) \| None | No | None | v8.1 | No | The maximum number of vectors to return. Must be a positive integer. When `None`, the server defaults to 100. |
| pagination_token | string \| None | No | None | v8.1 | No | A pagination token from a previous response's `pagination.next` field. When provided, returns the next page of results matching the filter. |

**Returns:** `FetchByMetadataResponse` — An object containing the fetched vectors, namespace name, usage statistics, and an optional pagination token for retrieving additional results. The `vectors` field is a dictionary mapping vector IDs to `Vector` objects.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeException` | The API call fails due to network issues, authentication failure, or server errors. |
| `ValueError` | The `filter` is invalid, malformed, or references non-existent metadata fields. |

**Example**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index(host="your-index-host")

# Fetch vectors with a complex metadata filter
response = index.fetch_by_metadata(
    filter={
        "genre": {"$in": ["comedy", "drama"]},
        "year": {"$gte": 2018},
        "rating": {"$gt": 7.0}
    },
    namespace="movie-embeddings",
    limit=50
)

print(f"Found {len(response.vectors)} matching vectors")
for vector_id, vector in response.vectors.items():
    print(f"{vector_id}: {vector.metadata}")

# Paginate through results if more are available
if response.pagination:
    next_response = index.fetch_by_metadata(
        filter={"genre": {"$in": ["comedy", "drama"]}, "year": {"$gte": 2018}},
        limit=50,
        pagination_token=response.pagination.next
    )
    print(f"Next page has {len(next_response.vectors)} vectors")
```

**Notes**

- The `limit` parameter controls how many vectors are returned per request, not the total number of matches. Use pagination to retrieve all matching vectors.
- The filter expression uses MongoDB query syntax. Invalid filters result in a `ValueError`.
- When paginating, the `pagination_token` from the previous response's `pagination.next` field is used to retrieve the next batch of results. The pagination token is `None` when all results have been retrieved.
- Results are returned as a dictionary of vectors. The order is not guaranteed across pagination calls.
- The `usage` field in the response contains read unit statistics for billing purposes.
- If no vectors match the filter, an empty `vectors` dictionary is returned with an empty `pagination` field.

---

### `AsyncIndex.fetch(ids: list[str], namespace: str | None = None, **kwargs) -> FetchResponse`

Asynchronous version of `fetch()`. Retrieves vectors by their IDs from a single namespace.

**Source:** `pinecone/db_data/index_asyncio.py:601-637`
**Added:** v8.0
**Deprecated:** No
**Idempotency:** Idempotent — fetching the same IDs multiple times returns the same vectors
**Side effects:** None (read-only operation)

**Parameters**

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| ids | array of string | Yes | — | v8.0 | No | The vector IDs to fetch. |
| namespace | string \| None | No | None | v8.0 | No | The namespace to fetch vectors from. When `None`, the default namespace is used. |

**Returns:** `FetchResponse` — An object containing the fetched vectors and the namespace name. The `vectors` field is a dictionary mapping vector IDs to `Vector` objects. If a requested ID does not exist in the index, it is not included in the response. This method is asynchronous and must be awaited.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeException` | The API call fails due to network issues, authentication failure, or server errors. |
| `ValueError` | The `ids` parameter is empty or not a list of strings. |

**Example**

```python
from pinecone import Pinecone
import asyncio

async def main():
    pc = Pinecone(api_key="your-api-key")
    async_index = pc.IndexAsyncio(host="your-index-host")

    # Fetch specific vectors asynchronously
    response = await async_index.fetch(ids=["doc-001", "doc-002"], namespace="production-embeddings")
    print(f"Fetched {len(response.vectors)} vectors from namespace '{response.namespace}'")

    for vector_id, vector in response.vectors.items():
        print(f"Vector {vector_id}: {vector.values[:3]}...")
        if vector.metadata:
            print(f"  Metadata: {vector.metadata}")

asyncio.run(main())
```

**Notes**

- Non-existent vector IDs are silently ignored; the response contains only vectors that exist in the index.
- The returned `Vector` objects include both dense `values` and any `metadata` stored with the vector.
- If no vectors are found for any of the requested IDs, an empty `vectors` dictionary is returned, not an error.
- Must be called with `await` since it returns an awaitable.

---

### `AsyncIndex.fetch_by_metadata(filter: FilterTypedDict, namespace: str | None = None, limit: int | None = None, pagination_token: str | None = None, **kwargs) -> FetchByMetadataResponse`

Asynchronous version of `fetch_by_metadata()`. Retrieves vectors matching a metadata filter expression from a single namespace.

**Source:** `pinecone/db_data/index_asyncio.py:639-717`
**Added:** v8.1
**Deprecated:** No
**Idempotency:** Idempotent — executing the same filter multiple times returns the same vectors (assuming no concurrent writes)
**Side effects:** None (read-only operation)

**Parameters**

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| filter | FilterTypedDict | Yes | — | v8.1 | No | A metadata filter expression to select vectors. Supports MongoDB-style query operators such as `$eq`, `$in`, `$gt`, `$lt`, `$ne`, `$and`, `$or`. See [metadata filtering documentation](https://docs.pinecone.io/docs/metadata-filtering). |
| namespace | string \| None | No | None | v8.1 | No | The namespace to fetch vectors from. When `None`, the default namespace is used. |
| limit | integer (int32) \| None | No | None | v8.1 | No | The maximum number of vectors to return. Must be a positive integer. When `None`, the server defaults to 100. |
| pagination_token | string \| None | No | None | v8.1 | No | A pagination token from a previous response's `pagination.next` field. When provided, returns the next page of results matching the filter. |

**Returns:** `FetchByMetadataResponse` — An object containing the fetched vectors, namespace name, usage statistics, and an optional pagination token for retrieving additional results. The `vectors` field is a dictionary mapping vector IDs to `Vector` objects. This method is asynchronous and must be awaited.

**Raises / Throws**

| Exception / Error | Condition |
|-------------------|-----------|
| `PineconeException` | The API call fails due to network issues, authentication failure, or server errors. |
| `ValueError` | The `filter` is invalid, malformed, or references non-existent metadata fields. |

**Example**

```python
from pinecone import Pinecone
import asyncio

async def main():
    pc = Pinecone(api_key="your-api-key")
    async_index = pc.IndexAsyncio(host="your-index-host")

    # Fetch vectors with a complex metadata filter asynchronously
    response = await async_index.fetch_by_metadata(
        filter={
            "genre": {"$in": ["comedy", "drama"]},
            "year": {"$gte": 2018},
            "rating": {"$gt": 7.0}
        },
        namespace="movie-embeddings",
        limit=50
    )

    print(f"Found {len(response.vectors)} matching vectors")
    for vector_id, vector in response.vectors.items():
        print(f"{vector_id}: {vector.metadata}")

    # Paginate through results if more are available
    if response.pagination:
        next_response = await async_index.fetch_by_metadata(
            filter={"genre": {"$in": ["comedy", "drama"]}, "year": {"$gte": 2018}},
            limit=50,
            pagination_token=response.pagination.next
        )
        print(f"Next page has {len(next_response.vectors)} vectors")

asyncio.run(main())
```

**Notes**

- The `limit` parameter controls how many vectors are returned per request, not the total number of matches. Use pagination to retrieve all matching vectors.
- The filter expression uses MongoDB query syntax. Invalid filters result in a `ValueError`.
- When paginating, the `pagination_token` from the previous response's `pagination.next` field is used to retrieve the next batch of results. The pagination token is `None` when all results have been retrieved.
- Results are returned as a dictionary of vectors. The order is not guaranteed across pagination calls.
- The `usage` field in the response contains read unit statistics for billing purposes.
- If no vectors match the filter, an empty `vectors` dictionary is returned with an empty `pagination` field.
- Must be called with `await` since it returns an awaitable.

---

## Response Types

### `FetchResponse`

The response object returned by `fetch()`.

**Import:** `from pinecone import FetchResponse`
**Source:** `pinecone/db_data/dataclasses/fetch_response.py:10-17`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| namespace | string | No | v8.0 | No | The namespace from which vectors were fetched. |
| vectors | dict[str, Vector] | No | v8.0 | No | A dictionary mapping vector IDs to their corresponding Vector objects. Empty if no matching vectors are found. |
| usage | Usage \| None | Yes | v8.0 | No | Token usage information for the operation, including read units consumed. |

---

### `FetchByMetadataResponse`

The response object returned by `fetch_by_metadata()`.

**Import:** `from pinecone import FetchByMetadataResponse`
**Source:** `pinecone/db_data/dataclasses/fetch_by_metadata_response.py:15-23`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| namespace | string | No | v8.1 | No | The namespace from which vectors were fetched. |
| vectors | dict[str, Vector] | No | v8.1 | No | A dictionary mapping vector IDs to their corresponding Vector objects. Empty if no vectors match the filter. |
| usage | Usage \| None | Yes | v8.1 | No | Token usage information for the operation, including read units consumed. |
| pagination | Pagination \| None | Yes | v8.1 | No | Pagination metadata for retrieving additional results. Contains a `next` field with the token to use for the next request. `None` when all results have been retrieved or when no results match the filter. |

---

### `FetchByMetadataResponse.Pagination`

Pagination metadata included in `FetchByMetadataResponse`.

**Import:** `from pinecone import FetchByMetadataResponse` (access via the Pagination field)
**Source:** `pinecone/db_data/dataclasses/fetch_by_metadata_response.py:11-12`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| next | string | No | v8.1 | No | The pagination token to pass to the next `fetch_by_metadata()` call to retrieve the next page of results. |

---

### `Vector`

A vector object returned in fetch responses.

**Import:** `from pinecone import Vector`
**Source:** `pinecone/db_data/dataclasses/vector.py:9-90`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| id | string | No | v8.0 | No | The unique identifier of the vector. |
| values | array of number (double) | No | v8.0 | No | The dense vector values. Each element is a floating-point number. |
| metadata | VectorMetadataTypedDict \| None | Yes | v8.0 | No | Arbitrary metadata associated with the vector. Omitted from the response when `None`. |
| sparse_values | SparseValues \| None | Yes | v8.0 | No | Sparse vector values, if present. Contains `indices` (array of int) and `values` (array of float). Omitted from the response when `None`. |

---

### `SparseValues`

Sparse vector representation included in `Vector` objects when sparse values are present.

**Import:** `from pinecone import SparseValues`
**Source:** `pinecone/db_data/dataclasses/sparse_values.py:8-20`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| indices | array of integer (int32) | No | v8.0 | No | The indices of non-zero elements in the sparse vector. Array of non-negative integers in ascending order. |
| values | array of number (double) | No | v8.0 | No | The values corresponding to the indices. Each element is a floating-point number. Arrays `indices` and `values` must have the same length. |

---

### `Usage`

Token usage information included in fetch responses.

**Import:** `from pinecone.core.openapi.db_data.models import Usage`
**Source:** `pinecone/core/openapi/db_data/model/usage.py:36-41`

| Field | Type | Nullable | Since | Deprecated | Description |
|-------|------|----------|-------|------------|-------------|
| read_units | integer (int64) | No | v8.0 | No | The number of read units consumed by the operation, used for billing purposes. |
