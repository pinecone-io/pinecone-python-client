# Migrating from v8.x to v9.x

v9 is a ground-up rewrite focused on simplicity, performance, and type safety. This guide
covers the breaking changes and shows you the v9 equivalent for each v8 pattern.

## Key changes

### 1. Namespace pattern for control-plane operations

In v8, control-plane methods lived directly on the `Pinecone` client:

```python
# v8
pc.create_index(name="my-index", dimension=1536, metric="cosine", spec=...)
indexes = pc.list_indexes()
pc.delete_index("my-index")
```

In v9, they are grouped under namespace properties:

```python
# v9
pc.indexes.create(name="my-index", dimension=1536, metric="cosine", spec=...)
indexes = pc.indexes.list()
pc.indexes.delete("my-index")
```

The same pattern applies to collections, backups, and inference:

```python
pc.collections.create(...)
pc.backups.list()
pc.inference.embed(...)
```

### 2. Async client rename

`PineconeAsyncio` is renamed to `AsyncPinecone`. The old name still works but is
deprecated and will be removed in a future release.

```python
# v8
from pinecone import PineconeAsyncio
async with PineconeAsyncio(api_key="...") as pc:
    ...

# v9
from pinecone import AsyncPinecone
async with AsyncPinecone(api_key="...") as pc:
    ...
```

### 3. Response models

v8 returned a mix of plain dicts and Pydantic models. v9 returns `msgspec.Struct` instances.
Field access is identical—`idx.name`, `idx.dimension`—but the objects are immutable.
`dict()` no longer works; use `msgspec.structs.asdict(idx)` if you need a dict.

```python
# v9 — field access is unchanged
idx = pc.indexes.describe("my-index")
print(idx.name)        # works
print(idx.dimension)   # works
print(dict(idx))       # TypeError — structs are not dict-convertible
```

### 4. HTTP transport: httpx replaces urllib3

The SDK uses `httpx` with HTTP/2 instead of `urllib3`. Retry behavior is now configured
with `RetryConfig` passed at client construction:

```python
# v8 — retry parameters were keyword args on the client
pc = Pinecone(api_key="...", retries=3)

# v9
from pinecone import Pinecone, RetryConfig
pc = Pinecone(
    api_key="...",
    retry_config=RetryConfig(max_retries=3, backoff_factor=1.5),
)
```

### 5. gRPC: Rust extension replaces grpcio

`GrpcIndex` is now backed by a compiled Rust extension instead of the Python `grpcio`
package. You do not need to install `grpcio` or `grpcio-tools`. The interface—`upsert`,
`query`, `fetch`, `delete`—is unchanged.

```python
# v9 — interface is the same; no grpcio dependency required
index = pc.index("my-index", grpc=True)
index.upsert(vectors=[...])
```

### 6. Import paths

Most public classes are still importable directly from `pinecone`:

```python
from pinecone import Pinecone, AsyncPinecone, Index, GrpcIndex
from pinecone import ServerlessSpec, PodSpec
from pinecone import ConflictError, NotFoundException, ForbiddenException
```

Deep imports (`from pinecone.core.client.api...`) are no longer supported. Use the
top-level package instead.

### 7. Python version requirement

Python 3.9 support is dropped. The minimum supported version is Python 3.10.

---

## v8 → v9 migration table

| Operation | v8 | v9 |
|---|---|---|
| Create index | `pc.create_index(name=..., dimension=..., spec=...)` | `pc.indexes.create(name=..., dimension=..., spec=...)` |
| List indexes | `pc.list_indexes()` | `pc.indexes.list()` |
| Describe index | `pc.describe_index("name")` | `pc.indexes.describe("name")` |
| Delete index | `pc.delete_index("name")` | `pc.indexes.delete("name")` |
| Configure index | `pc.configure_index("name", ...)` | `pc.indexes.configure("name", ...)` |
| Check index exists | `pc.describe_index("name")` + try/except | `pc.indexes.exists("name")` |
| Get data-plane index | `pc.Index("name")` | `pc.Index("name")` *(unchanged)* |
| Get gRPC index | `Pinecone(...).GrpcIndex("name")` | `pc.index("name", grpc=True)` |
| Create collection | `pc.create_collection(name=..., source=...)` | `pc.collections.create(name=..., source=...)` |
| List collections | `pc.list_collections()` | `pc.collections.list()` |
| Delete collection | `pc.delete_collection("name")` | `pc.collections.delete("name")` |
| Upsert vectors | `index.upsert(vectors=[...])` | `index.upsert(vectors=[...])` *(unchanged)* |
| Query vectors | `index.query(vector=[...], top_k=10)` | `index.query(vector=[...], top_k=10)` *(unchanged)* |
| Fetch vectors | `index.fetch(ids=[...])` | `index.fetch(ids=[...])` *(unchanged)* |
| Delete vectors | `index.delete(ids=[...])` | `index.delete(ids=[...])` *(unchanged)* |
| Async client | `PineconeAsyncio(api_key=...)` | `AsyncPinecone(api_key=...)` |
| Retry config | `Pinecone(retries=3)` | `Pinecone(retry_config=RetryConfig(max_retries=3))` |
| Convert response to dict | `dict(idx)` or `idx.to_dict()` | `msgspec.structs.asdict(idx)` |
| Embed text | `pc.inference.embed(...)` | `pc.inference.embed(...)` *(unchanged)* |

---

## Legacy aliases

The following aliases remain importable from `pinecone` but are deprecated:

| Deprecated name | Canonical name |
|---|---|
| `PineconeAsyncio` | `AsyncPinecone` |
| `ForbiddenException` | `ForbiddenException` *(still valid — error class name unchanged)* |
| `NotFoundException` | `NotFoundException` *(still valid — error class name unchanged)* |

These aliases will be removed in a future major release. Update your code to use the
canonical names.
