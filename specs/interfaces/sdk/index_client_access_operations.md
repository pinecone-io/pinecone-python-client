# Index Client Access Operations

Provides factory methods on the Pinecone and PineconeAsyncio clients to obtain Index data plane clients. These methods are the primary entry point for accessing vector data operations (upsert, query, delete, fetch) on a Pinecone index.

---

## `Pinecone.Index()`

Returns a client for performing data operations on a Pinecone index. Targets an index by name (control plane lookup) or by host URL (direct access).

**Source:** `pinecone/pinecone.py:1286-1396`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** N/A (no side effects)
**Side effects:** None (factory method)

### Signature

```python
def Index(
    self,
    name: str = "",
    host: str = "",
    **kwargs
) -> Index
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `name` | `string (1-45 chars)` | No | `""` | v1.0 | No | The name of the index to target. If specified, the client performs a control plane lookup to obtain the index host URL. If both `name` and `host` are omitted, raises `ValueError`. Not recommended for production due to added control plane dependency. |
| `host` | `string (uri)` | No | `""` | v1.0 | No | The host URL of the index to target (e.g., `index-abc123.svc.pinecone.io`). If specified, the client uses the host directly without making any additional control plane calls. Recommended for production. Raises `ValueError` if the host appears invalid (contains no `.` and is not `localhost`). |
| `pool_threads` | `int` | No | — | v1.0 | No | The number of threads to use when making parallel requests by calling index methods with `async_req=True`, or using methods that make use of thread-based parallelism automatically such as `query_namespaces()`. Passed via `**kwargs`. |
| `connection_pool_maxsize` | `int` | No | — | v1.0 | No | The maximum number of connections to keep in the connection pool. Passed via `**kwargs`. |

### Returns

**Type:** `Index` — A client instance configured to perform vector operations (query, upsert, delete, fetch) on the target index.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Both `name` and `host` are empty strings (required: must provide at least one). |
| `ValueError` | The `host` does not appear valid (no `.` and is not `localhost`). |
| `PineconeException` | The specified index `name` does not exist or cannot be resolved to a host. Only raised when using `name` parameter. |

### Example

```python
from pinecone import Pinecone, Vector

pc = Pinecone(api_key="sk-example-key-do-not-use")

# Option 1: Target by host (recommended for production)
index = pc.Index(host="index-abc123.svc.pinecone.io")
response = index.upsert(vectors=[Vector(id="vec1", values=[0.1, 0.2, 0.3])])

# Option 2: Target by name (control plane lookup)
index = pc.Index(name="my-index")
response = index.query(vector=[0.1, 0.2, 0.3], top_k=10)
```

### Notes

- **Host lookup performance:** When using the `name` parameter, the client caches the resolved host for future use, so the control plane call is incurred only once per index name. However, this still introduces a runtime dependency on api.pinecone.io and is not recommended for production.
- **Host validation:** The host validation is client-side and basic; it checks for a dot (or `localhost`) to prevent accidental use of index names as hosts. The actual connection attempt will occur during the first data operation.
- **Index client lifetime:** The returned Index client is independent of the Pinecone client; closing the Pinecone client does not affect previously created Index instances.
- **Pool threads:** The Index client inherits the `pool_threads` configuration from the parent Pinecone client. Override via the `pool_threads` kwarg if needed.

---

## `PineconeAsyncio.IndexAsyncio()`

Returns an asyncio-compatible client from an async PineconeAsyncio client. Requires explicit `host` parameter (no name-based lookup for async clients).

**Source:** `pinecone/pinecone_asyncio.py:1271-1353`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** N/A (no side effects)
**Side effects:** None (factory method)

### Signature

```python
def IndexAsyncio(
    self,
    host: str
) -> IndexAsyncio
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `host` | `string (uri)` | Yes | — | v1.0 | No | The host URL of the index to target. Required; name-based lookup is not supported via this method. Raises `ValueError` if empty or invalid. |

### Returns

**Type:** `IndexAsyncio` — An asyncio-compatible client instance.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `host` is `None` or empty string. |
| `ValueError` | The `host` does not appear valid. |

### Example

```python
import asyncio
from pinecone import PineconeAsyncio

async def main():
    async with PineconeAsyncio(api_key="sk-example-key-do-not-use") as pc:
        async with pc.IndexAsyncio(host="index-abc123.svc.pinecone.io") as index:
            results = await index.query(vector=[0.1, 0.2, 0.3], top_k=10)

asyncio.run(main())
```

### Notes

- **Context manager:** The returned IndexAsyncio instance should be used as an async context manager (via `async with`) for proper cleanup. Manual `close()` can be called if a context manager is not used.
- **Concurrency:** All async methods on the Index client are safe for concurrent use.
- **Client context:** This is the recommended way to get an IndexAsyncio client when using the PineconeAsyncio client. It is also possible to get an IndexAsyncio from a sync Pinecone client via `Pinecone.IndexAsyncio()`, but that should only be used if you already have a sync client and need async index access.

---

## `Pinecone.IndexAsyncio()`

Returns an asyncio-compatible client from a synchronous Pinecone client. Requires explicit `host` parameter (no name-based lookup for async clients).

**Source:** `pinecone/pinecone.py:1398-1440`
**Added:** v1.0
**Deprecated:** No
**Idempotency:** N/A (no side effects)
**Side effects:** None (factory method)

### Signature

```python
def IndexAsyncio(
    self,
    host: str
) -> IndexAsyncio
```

### Parameters

| Parameter | Type | Required | Default | Since | Deprecated | Description |
|-----------|------|----------|---------|-------|------------|-------------|
| `host` | `string (uri)` | Yes | — | v1.0 | No | The host URL of the index to target. Required; name-based lookup is not supported via this method. Raises `ValueError` if empty or invalid. |

### Returns

**Type:** `IndexAsyncio` — An asyncio-compatible client instance.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | `host` is `None` or empty string. |
| `ValueError` | The `host` does not appear valid. |

### Example

```python
import asyncio
from pinecone import Pinecone

async def main():
    pc = Pinecone(api_key="sk-example-key-do-not-use")

    # Create async index client from sync Pinecone client
    async with pc.IndexAsyncio(host="index-abc123.svc.pinecone.io") as index:
        results = await index.query(vector=[0.1, 0.2, 0.3], top_k=10)

asyncio.run(main())
```

### Notes

- **Sync vs Async factories:** `Pinecone.Index()` returns a sync client and `Pinecone.IndexAsyncio()` returns an async client. Both can be obtained from the same `Pinecone` instance, but they maintain separate connections.
- **Alternative:** `PineconeAsyncio` is the recommended entry point for async workloads. Use `Pinecone.IndexAsyncio()` only when you already have a sync Pinecone client and need async index access.
