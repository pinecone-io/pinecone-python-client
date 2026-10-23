# FAQ

### Why is my import slow?

`import pinecone` on its own is nearly free: the package binds its exports lazily, so
a module loads only when you name something from it. Import just the names you use:

```python
from pinecone import Pinecone
```

`from pinecone import *` loads every export and is by far the slowest form. Avoid it in
startup paths you care about. The HTTP stack (`httpx`, `orjson`) loads when you
construct a client, not at import.

### Why does `pc.indexes.list()` yield only a single page?

The server returns every index in one page today. The returned `Paginator` (or
`AsyncPaginator`) still exposes the paginator interface, so code that iterates it keeps
working if that changes:

```python
names = [index.name for index in pc.indexes.list()]
```

`pc.list_indexes()`, kept for 9.x compatibility, is not a paginator: it returns an
`IndexList`, which has `.names()`.

### Can I use the async client with FastAPI?

Yes. Open `AsyncPinecone` in a lifespan context manager so the app shares one client:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pinecone import AsyncPinecone

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPinecone(api_key="your-api-key") as pc:
        app.state.pc = pc
        yield

app = FastAPI(lifespan=lifespan)
```

Control-plane calls on that client reuse one connection pool. Each handle from
`await pc.index(...)` has its own pool, so create it once and reuse it rather than
opening one per request.

### What is the difference between `Index` and `GrpcIndex`?

The transport. `Index` speaks REST over HTTP; `GrpcIndex` speaks gRPC, which pays off on
high-throughput upserts. Both ship with the base `pinecone` package:

```python
# REST: general purpose
index = pc.index("article-search")

# gRPC: high-throughput upserts
index = pc.index("article-search", grpc=True)
```

`GrpcIndex` serves indexes created with top-level `dimension=`/`metric=`. It has no
document operations, so a document index (one created with `schema=`) uses `Index`. See
[Using the gRPC client](guides/grpc.md) for when the switch is worth making.

### How do I handle a `ConflictError` when creating an index that already exists?

Catch `ConflictError` from the top-level `pinecone` package:

```python
from pinecone import ConflictError, Pinecone

pc = Pinecone(api_key="your-api-key")
try:
    pc.indexes.create(
        name="article-search",
        schema={"fields": {"your_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
        deployment={"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
    )
except ConflictError:
    pass  # the index already exists
```

### Can I modify a response object?

Response objects are `msgspec.Struct` instances. Most are not frozen, but each is a
snapshot of one response: the SDK never writes your changes back, and the next call
returns a new object. For a plain dict you can modify, call `.to_dict()`, which most
response models provide and which converts nested models too:

```python
d = pc.indexes.describe("article-search").to_dict()
```

`msgspec.structs.asdict()` works on any response but converts only the top level.
