# Performance

This page covers the choices that most affect throughput and latency: which write
path you use, how you reuse clients, how much concurrency you ask for, and which
transport you pick.

## Transport

- `Index` and `AsyncIndex` (REST) use [httpx](https://www.python-httpx.org/) over
  HTTP/1.1 with connection keepalive.
- `GrpcIndex` uses a native (Rust-backed) gRPC channel over HTTP/2 with binary
  protobuf framing.

For the REST clients, parallel batched upsert (below) is how you drive
concurrency. gRPC pulls ahead only at high concurrency; see
[When to Use gRPC](#when-to-use-grpc).

## Choosing How to Load Data

Throughput depends more on picking the right write path than on tuning any
knob on the wrong one.

| Path | Reach for it when | Batching |
|---|---|---|
| `index.documents.upsert(namespace=..., documents=[...])` | The index declares a `schema=` and one request will hold the documents | None: one request |
| `index.documents.batch_upsert(namespace=..., documents=[...])` | The index declares a `schema=` and you have more documents than one request carries | Client-side, in parallel, with partial-failure reporting |
| `index.upsert(vectors=[...])` | The index was created with top-level `dimension=`/`metric=` and one request will hold the vectors | None: one request |
| `index.upsert(vectors=[...], batch_size=N)` | Same index type, more vectors than one request carries | Client-side, in parallel, with partial-failure reporting |
| `index.upsert_from_dataframe(df)` | Same index type, vectors already in a pandas DataFrame | Client-side, `batch_size=500` by default |
| `index.upsert_records(records=[...])` | The index is a legacy integrated index (created with `create_for_model`) | None: one request per call |
| `index.start_import(uri="s3://...")` | Millions of records already sit in Parquet in cloud storage | Server-side and asynchronous; no client bandwidth spent |

An unbatched call sends everything in one request, so a large enough input is
rejected for exceeding the request size limit. Switch to the batched call
instead of writing your own loop. `upsert_records()` has no batched form, so
chunk the record list yourself and call it once per chunk.

`upsert_from_dataframe()` needs pandas, which is not an SDK dependency. Install
it yourself; without it the call raises and says so.

## Connection Pooling

`Pinecone` and `Index` each hold a persistent `httpx.Client` (`httpx.AsyncClient`
for the async variants). A new client per request pays for a new TLS handshake
and connection every time.

**Reuse the same `Index` instance** across calls rather than constructing a new one each
time:

```python
# Good: one client, many calls
from pinecone import Pinecone

pc = Pinecone()
index = pc.index(name="product-search")

for batch in batches:
    index.upsert(vectors=batch)

# Bad: a new HTTP client for every upsert
for batch in batches:
    index = pc.index(name="product-search")  # new client every time
    index.upsert(vectors=batch)
```

After the first call the host comes from the client's cache, so the second
loop sends no extra describe requests, but it still builds a fresh
`httpx.Client` and does a fresh TLS handshake on every pass.

Use the context manager protocol to release connections when you are done:

```python
with pc.index(name="product-search") as index:
    index.upsert(vectors=batch)
```

## Cold Import Cost

`import pinecone` is cheap: top-level names such as `Pinecone` and
`AsyncPinecone` load on first access, and heavy dependencies (the gRPC
extension, pandas, tqdm) load only when a call needs them. If startup latency
matters, you can still defer the import to first use:

```python
def get_index():
    from pinecone import Pinecone

    pc = Pinecone()
    return pc.index(name="product-search")
```

## Batching Large Upserts

For more data than one request holds, pass `batch_size` to `Index.upsert()`.
The SDK splits the input into batches and sends them in parallel, up to
`max_concurrency` at a time. A per-host adaptive gate can hold concurrency
below that cap when the server pushes back, and each batch is retried on
transient errors.

```python
response = index.upsert(
    vectors=large_list,
    batch_size=100,     # vectors per request
    max_concurrency=8,  # batches in flight at once (the default)
)
print(response.upserted_count)     # successful items
print(response.failed_item_count)  # 0 if everything succeeded
```

`AsyncIndex.upsert()` and `GrpcIndex.upsert()` accept the same `batch_size` and
`max_concurrency` arguments, and `upsert_from_dataframe()` accepts
`batch_size`, `max_concurrency` and `total_timeout` on all three clients. For a
document index, `index.documents.batch_upsert()` takes the same
`max_concurrency` and `total_timeout` and returns a `BatchResult`.

With `batch_size` set, `upsert()` reports failed batches in the returned
`UpsertResponse` instead of raising on the first one; see
[Handling partial failures](../how-to/vectors/upsert-and-query.md#handling-partial-failures).
For what each knob controls beyond throughput, see
[How Bulk Ingest Behaves](bulk-ingest.md).

Parallel batching is many times faster than a sequential loop of single
requests, and the default `max_concurrency` gets most of that gain.

### Tuning `max_concurrency`

The default is `8` on every method and transport, so throughput does not
change from machine to machine. Raise it only when you can measure the result
on your own workload:

| `max_concurrency` | When to use it |
|---|---|
| `1` | A strict per-second quota, or you want batches sent in order |
| `8` *(default)* | General use, including large bulk loads |
| Higher | Diminishing returns: the server's ingest capacity, not the SDK, becomes the bottleneck |

To go past that point, split the work across several processes, each with its
own `Index`, rather than raising `max_concurrency` further on one client. For
multi-million-vector loads from cloud storage, use `index.start_import()`,
which skips per-batch HTTP overhead entirely.

## Query Latency

A query is one round trip, so parallel batching does not apply. On REST,
decoding the response is a large share of the latency for big results: a
`top_k` in the hundreds with `include_values=True` costs far more than the same
query returning only IDs and scores. Request values and metadata only when you
use them. gRPC responses are protobuf, so large results decode faster there.

## Async Concurrency

Pick the async client (`AsyncPinecone` / `AsyncIndex`) when your code already
runs inside `async def`, most often in an async web framework such as
**FastAPI**, **Starlette** or **Litestar**. There, a blocking sync call stalls
the event loop for every concurrent request, or forces you to offload it to a
thread; the async client lets you `await` Pinecone calls inline.

For **pure bulk upsert**, use native batched upsert rather than a hand-rolled
`asyncio.gather`: the same parallelism, with retries and partial-failure
reporting built in:

```python
# Preferred
from pinecone import AsyncPinecone

async with AsyncPinecone() as pc:
    index = await pc.index(name="product-search")
    async with index:
        response = await index.upsert(
            vectors=large_list,
            batch_size=100,
            max_concurrency=8,
        )
```

For **mixed workloads** (concurrent upserts and queries, or query fan-out across
many namespaces), `asyncio.gather` over `AsyncIndex` calls is the natural
pattern:

```python
import asyncio
from pinecone import AsyncPinecone

async with AsyncPinecone() as pc:
    index = await pc.index(name="product-search")
    async with index:
        results = await asyncio.gather(
            index.upsert(vectors=writes_batch, batch_size=100),
            index.query(vector=q1, top_k=10),
            index.query(vector=q2, top_k=10),
        )
```

For bulk upsert alone, sync and async reach similar throughput. Pick the
client that matches your application style.

## When to Use gRPC

`GrpcIndex` accepts the same `batch_size=` and `max_concurrency=` arguments as
the REST `Index`, so the call site looks identical. For how to open one, what
it supports, and how its errors and retries differ, see {doc}`/guides/grpc`.

At the default `max_concurrency`, the three transports deliver about the same
upsert throughput, so the choice is about API style. gRPC pulls ahead as
concurrency rises.

Pick gRPC when:

- You run **sustained bulk upserts at a raised `max_concurrency`**.
- Your queries return **large results** (high `top_k` with values or metadata).

Stay on REST when:

- You run at default settings or low concurrency.
- You need async: `GrpcIndex` is sync-only, so use `AsyncIndex` over REST.

```python
from pinecone import Pinecone

pc = Pinecone()
with pc.index(name="product-search", grpc=True) as index:
    response = index.upsert(
        vectors=large_list,
        batch_size=100,
        max_concurrency=16,
    )
```

## Summary

| Technique | Where it helps |
|-----------|---------------|
| [Right write path](#choosing-how-to-load-data) | Before any tuning: batched upsert, DataFrame, or server-side import |
| Reuse the `Index` instance | Avoids per-call TLS and connection setup |
| `batch_size=` with `max_concurrency=` | Bulk upsert, many times faster than a sequential loop |
| Request only the fields you use | Query latency on REST with large `top_k` |
| `AsyncIndex` + `asyncio.gather()` | Mixed concurrent read and write workloads |
| `GrpcIndex` (sync only) | Sustained bulk upserts at raised concurrency; large query results |
| `index.start_import()` | Multi-million-vector loads from cloud storage |
