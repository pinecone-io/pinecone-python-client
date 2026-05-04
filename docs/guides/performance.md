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

## Batching Large Upserts

For datasets larger than a single request payload, pass `batch_size` to
`Index.upsert()`. The SDK splits the input into batches and sends them in parallel —
sync via a cached `ThreadPoolExecutor`, async via an `asyncio.Semaphore`. HTTP-level
retries happen automatically per batch.

```python
response = index.upsert(
    vectors=large_list,    # any length
    batch_size=100,        # vectors per request
    max_concurrency=4,     # parallel in-flight requests (default 4, range 1–64)
)
print(response.upserted_count)         # successful items
print(response.failed_item_count)      # 0 if everything succeeded
```

The same kwargs are accepted on `AsyncIndex.upsert()`, `Index.upsert_from_dataframe()`,
and `Index.upsert_records()`.

When `batch_size` is set, `upsert()` returns an `UpsertResponse` with partial-failure
information instead of raising on the first failed batch — see
[Handling partial failures](../how-to/vectors/upsert-and-query.md#handling-partial-failures).

### How much faster is parallel batching?

Measured on 10k vectors / 1536-d / batch=100 against an aws-us-east-1 serverless
index, sync REST `Index`:

| Path | Wall time | Speedup |
|---|---:|---:|
| `pinecone` v8 sequential loop (baseline) | 112 s | 1.0× |
| `max_concurrency=4` (default) | 9.6 s | **11.7×** |
| `max_concurrency=8` | 5.7 s | 19.7× |
| `max_concurrency=16` | 5.0 s | 22.3× |
| `max_concurrency=32` | 4.4 s | 25.5× |

Async REST follows the same shape with somewhat smaller speedups, since v8 async
sequential was already faster than v8 sync sequential. gRPC is faster than REST
at high concurrency — see [When to Use gRPC](#when-to-use-grpc). Numbers are
from a controlled benchmark; see [Methodology](#methodology).

### Tuning `max_concurrency`

The default of `4` is calibrated to capture ~70% of the achievable speedup with
modest pressure on the cluster — safe to use without tuning. Push higher only when
you have a reason and can measure the result on your workload:

| `max_concurrency` | When to use it |
|---:|---|
| `1` | Strict per-second quota, or you want sequential semantics for ordering |
| `4` *(default)* | General use; ~70% of the win, no tuning required |
| `8` | Large bulk loads on a well-provisioned index — typically the sweet spot |
| `16–32` | Diminishing returns; the cluster (not the SDK) is usually the bottleneck above ~16 |
| `>32` | Rarely worth it for a single client; consider sharding the work across multiple clients instead |

Throughput saturates around `c≈16` for most workloads because cluster-side
ingress capacity becomes the bottleneck, not the SDK. If you do need to push past
that ceiling, run multiple `Index` instances from separate processes rather than
raising `max_concurrency` further on one client.

For multi-million-vector loads from cloud storage, prefer `index.start_import()`
over batched upsert — it avoids per-batch HTTP overhead entirely.

## Async Concurrency

The async client (`AsyncPinecone` / `AsyncIndex`) is the right choice when your
application is already async (FastAPI, Starlette, aiohttp) or when you are mixing
reads and writes that should overlap.

For **pure bulk upsert**, prefer native batched upsert over a hand-rolled
`asyncio.gather` — same parallelism, less code, automatic retries, and partial-failure
reporting:

```python
# Preferred
async with pc.index(host=desc.host) as index:
    response = await index.upsert(
        vectors=large_list,
        batch_size=100,
        max_concurrency=8,
    )
```

For **mixed workloads** — concurrent upserts and queries, or query fan-out across
many namespaces — `asyncio.gather` over `AsyncIndex` calls is still the natural
pattern:

```python
async with pc.index(host=desc.host) as index:
    results = await asyncio.gather(
        index.upsert(vectors=writes_batch, batch_size=100),
        index.query(vector=q1, top_k=10),
        index.query(vector=q2, top_k=10),
    )
```

Sync vs async at high concurrency: with native batched upsert at `max_concurrency=32`,
sync (~4.4 s) edges out async (~5.0 s) on the 10k-vector benchmark — the cached
`ThreadPoolExecutor` is competitive with `asyncio.Semaphore` once cluster-side
ingress dominates. Pick the client that matches your application style; throughput
is similar at the saturation point.

## When to Use gRPC

`pinecone.grpc.GrpcIndex` accepts the same `batch_size=` and `max_concurrency=`
kwargs as the REST `Index`, so the call site looks identical. The wire-level
difference is binary protobuf and HTTP/2 framing optimised for streaming.

The numbers, on the same 10k-vector / 1536-d / batch=100 benchmark
([Methodology](#methodology)) — wall time, p50:

| `max_concurrency` | REST sync | REST async | gRPC |
|---:|---:|---:|---:|
| `pinecone` v8 sequential | 112 s | 67 s | 34 s |
| 1 | 31.5 s | 32.7 s | 35.0 s |
| 4 (default) | 9.6 s | 10.2 s | 10.0 s |
| 8 | 5.7 s | 5.9 s | 5.7 s |
| 16 | 5.0 s | 6.6 s | 4.0 s |
| 32 | 4.4 s | 5.0 s | 2.7 s |

A few things stand out:

- **Even sequential, gRPC was already ~3× faster than REST sync** — `pinecone` v8
  baseline: 34 s gRPC vs 112 s REST. Protobuf encoding and HTTP/2 framing buy a
  lot before any parallelism enters the picture.
- **At default settings, the three transports are essentially tied** (~10 s).
  For typical workloads, the choice is about API style and dependencies, not
  throughput.
- **gRPC pulls ahead as concurrency rises** — at `max_concurrency=32`, gRPC
  finishes the same work 1.5–1.9× faster than REST.
- **`max_concurrency=1` doesn't help gRPC** — `pinecone` v8's gRPC was already
  pipelining requests internally, so the new code path's win comes from explicit
  fan-out at higher concurrency, not from the protocol switch.

Pick gRPC when:

- You're doing **sustained bulk upserts at `max_concurrency` ≥ 16** and the
  extra throughput is worth the extra dependency (`pinecone[grpc]`).
- You want the **lowest absolute write latency floor** on a single client
  (~2.7 s for 10k vectors at `c=32` on the reference workload).

Stay on REST when:

- You're at default settings or low concurrency — there is no measurable
  benefit, and REST has fewer transitive dependencies.
- You need async — `GrpcIndex` is sync-only; for async workloads use
  `AsyncIndex` over REST.

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
| HTTP/2 + httpx | All transports, always active |
| Reuse `Index` instance | Eliminate per-call TLS/connection overhead |
| msgspec structs | Response deserialization — faster than Pydantic |
| orjson | Request serialization — faster than stdlib `json` |
| Lazy imports | Reduce cold-start time |
| `Index.upsert(batch_size=…, max_concurrency=…)` | Bulk upsert — typical 10–25× over a sequential loop |
| `AsyncIndex` + `asyncio.gather()` | Mixed concurrent read/write workloads |
| `GrpcIndex` (sync only) | Sustained bulk upserts at `max_concurrency` ≥ 16 — ~1.5–1.9× over REST |
| `index.start_import()` | Multi-million-vector loads from cloud storage |

## Methodology

The numbers in this guide come from a controlled benchmark — 10,000 random
1536-dimensional vectors, `batch_size=100`, single client, fresh namespace per
run, against an aws-us-east-1 serverless index. The "v8 sequential" rows use
`pinecone==8.1.2` from PyPI (sequential `batch_size=` loop, fail-fast on first
batch error). The `max_concurrency=N` rows use this version of the SDK with
native parallel batched upsert. Each row is the p50 of 3 measured iterations
after 1 warmup.

Your numbers will vary with region, RTT, vector dimension, batch size, payload
metadata, and concurrent traffic from other clients. When in doubt, measure
on your own workload.
