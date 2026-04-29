# FAQ

### Why is my import slow?

Cold imports of large packages can take tens of milliseconds. The SDK uses lazy imports
so the heavy modules (`httpx`, `msgspec`, `orjson`) load only when you first use them.
The fastest way to initialize is to import just what you need:

```python
from pinecone import Pinecone   # imports only the Pinecone class
```

Avoid wildcard imports (`from pinecone import *`) in performance-sensitive startup paths.

### Why does `pc.indexes.list()` not support pagination?

Serverless index listings return at most a few hundred entries, which fits comfortably
in a single response. A paginated API would add complexity for no practical benefit at
this scale.

### Can I use the async client with FastAPI?

Yes. Use `AsyncPinecone` as a FastAPI dependency or inside a lifespan context manager:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pinecone import AsyncPinecone

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPinecone(api_key="...") as pc:
        app.state.pc = pc
        yield

app = FastAPI(lifespan=lifespan)
```

`AsyncPinecone` shares an `httpx.AsyncClient` connection pool across the full lifetime
of the context, so you get efficient connection reuse without re-establishing the pool
on every request.

### Does the SDK support HTTP/2?

Yes. The SDK uses `httpx`, which multiplexes requests over a single HTTP/2 connection
by default when the server supports it. No configuration is required.

### What is the difference between `Index` and `GrpcIndex`?

`Index` uses the REST/HTTP API. `GrpcIndex` uses gRPC, which has lower per-request
overhead and is better suited to high-throughput bulk operations such as large upsert
batches. For typical read-heavy or mixed workloads, `Index` is simpler to operate.

```python
# REST — general purpose
index = pc.Index("my-index")

# gRPC — high-throughput upserts
index = pc.GrpcIndex("my-index")
```

### How do I handle a `ConflictError` when creating an index that already exists?

Catch `ConflictError` from the top-level `pinecone` package:

```python
from pinecone import Pinecone, ConflictError, ServerlessSpec

pc = Pinecone(api_key="...")
try:
    pc.indexes.create(
        name="my-index",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
except ConflictError:
    pass  # index already exists — continue
```

### Why are responses immutable?

Response objects are `msgspec.Struct` instances, which are frozen by default. Immutability
eliminates a class of bugs where code accidentally modifies a shared response object and
enables safe use across threads without locks. If you need a mutable copy, convert it:

```python
import msgspec
idx = pc.indexes.describe("my-index")
d = msgspec.structs.asdict(idx)   # returns a plain dict you can modify
```
