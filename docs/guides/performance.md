# Performance

The SDK is designed for low overhead. This page describes the key design choices and
the patterns that keep your application fast.

## HTTP/2 Multiplexing

The SDK uses [httpx](https://www.python-httpx.org/) with the `httpx[http2]` transport.
HTTP/2 multiplexes multiple requests over a single TCP connection, eliminating the
per-request connection setup cost that HTTP/1.1 incurs under concurrency.

This is enabled by default — no configuration required.

## Connection Pooling

Both `Pinecone` and `Index` maintain a persistent `httpx.Client` (or `httpx.AsyncClient`
for the async variants). Creating a new client for every request wastes time on TLS
handshakes and connection setup.

**Reuse the same `Index` instance** across calls rather than constructing a new one each
time:

```python
# Good — one client, many calls
from pinecone import Pinecone

pc = Pinecone()
desc = pc.indexes.describe("product-search")
index = pc.index(host=desc.host)

for batch in batches:
    index.upsert(vectors=batch)

# Bad — a new HTTP client for every upsert
for batch in batches:
    index = pc.index(host=desc.host)  # new client every time
    index.upsert(vectors=batch)
```

Use the context manager protocol to ensure connections are released when you are done:

```python
with pc.index(host=desc.host) as index:
    index.upsert(vectors=large_batch)
```

## Fast Serialization with msgspec and orjson

Response models are `msgspec.Struct` instances. `msgspec` uses zero-copy deserialization
and avoids Python object allocation overhead that Pydantic-based models incur. Request
bodies are serialized with `orjson`, which is typically 5–10× faster than the standard
library `json` module.

These libraries are always active — no configuration is needed.

## Cold Import Cost

The SDK uses lazy imports to keep its cold-start time under 10 ms. Top-level SDK symbols
(`Pinecone`, `AsyncPinecone`, etc.) are available as soon as you import `pinecone`, but
heavy optional dependencies (gRPC, pandas, tqdm) are only imported when you actually use
them.

If your application is latency-sensitive at startup, avoid importing `pinecone` in
module-level code that runs before it is needed:

```python
# Fine — deferred to first use
def get_index() -> Index:
    from pinecone import Pinecone
    pc = Pinecone()
    return pc.index(host="...")
```

## Batching

`Index.upsert()` sends all provided vectors in a single request. For very large datasets,
split vectors into chunks of 100–1000 and upsert each chunk separately:

```python
BATCH_SIZE = 500

all_vectors = [...]  # your full list

for i in range(0, len(all_vectors), BATCH_SIZE):
    batch = all_vectors[i : i + BATCH_SIZE]
    index.upsert(vectors=batch)
```

`Index.upsert_records()` works the same way for integrated-inference indexes.

For the highest throughput, use the `GrpcIndex` with `upsert_async()` so batches are
submitted concurrently:

```python
from concurrent.futures import as_completed
from pinecone.grpc import GrpcIndex

with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    futures = [
        index.upsert_async(vectors=batch)
        for batch in batches
    ]
    total = sum(f.result().upserted_count for f in as_completed(futures))
```

Or use `upsert_from_dataframe()` which handles batching and concurrency automatically:

```python
response = index.upsert_from_dataframe(df, batch_size=500)
```

## Async Concurrency

The async client (`AsyncPinecone` / `AsyncIndex`) allows many concurrent requests over
the same connection without threads:

```python
import asyncio
from pinecone import AsyncPinecone

async def upsert_batch(index: AsyncIndex, batch: list) -> int:
    response = await index.upsert(vectors=batch)
    return response.upserted_count

async def main() -> None:
    async with AsyncPinecone() as pc:
        desc = await pc.indexes.describe("product-search")
        async with pc.index(host=desc.host) as index:
            results = await asyncio.gather(
                *[upsert_batch(index, batch) for batch in batches]
            )
    print(f"Total upserted: {sum(results)}")

asyncio.run(main())
```

## Summary

| Technique | Where it helps |
|-----------|---------------|
| HTTP/2 + httpx | All transports, always active |
| Reuse `Index` instance | Eliminate per-call TLS/connection overhead |
| msgspec structs | Response deserialization — faster than Pydantic |
| orjson | Request serialization — faster than stdlib `json` |
| Lazy imports | Reduce cold-start time |
| Vector batching | Upsert throughput |
| `GrpcIndex` + `upsert_async()` | Highest-throughput bulk upsert |
| `AsyncIndex` + `asyncio.gather()` | High-concurrency async workloads |
