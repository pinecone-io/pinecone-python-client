# Using the gRPC Client

`GrpcIndex` sends data-plane requests over gRPC instead of REST. Binary serialization
and HTTP/2 multiplexing give it lower per-call overhead, which shows most on bulk upserts
and high-throughput query loops. gRPC support ships in the base `pinecone` package; there
is no extra to install.

`GrpcIndex` serves indexes created with top-level `dimension=`/`metric=`. It has no
`documents` namespace, so a schema-based (document) index needs the HTTP `Index` (see
[Limitations](#limitations)).

## Creating a GrpcIndex

`pc.index()` with `grpc=True` resolves the host for you:

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index(name="product-search", grpc=True)
```

If you already know the host, construct it directly. Every argument is keyword-only:

```python
from pinecone.grpc import GrpcIndex

index = GrpcIndex(
    host="product-search-abc123.svc.pinecone.io",
    api_key="YOUR_API_KEY",  # or set PINECONE_API_KEY
)
```

`GRPCIndex` is an alias for the same class. `GrpcIndex` is a context manager, so use
`with` or call `close()` when you are done:

```python
with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    index.upsert(vectors=[("product-42", [0.012, -0.087, 0.153])])
```

## Endpoint scheme

The channel dials `https`. To reach a data plane over plaintext (a plaintext gateway, an
egress proxy in front of a private endpoint, or a local simulator), set `grpc_scheme` on
the client or the index:

```python
pc = Pinecone(grpc_scheme="http")
index = pc.index(name="product-search", grpc=True)

index = GrpcIndex(host="http://10.0.0.7:50051", grpc_scheme="http")
```

`PINECONE_GRPC_SCHEME` sets the same thing from the environment; the keyword argument
wins. The scheme decides whether the connection uses TLS, and `secure` supplies the TLS
material, so `grpc_scheme="http"` is plaintext whatever `secure` says.
`grpc_scheme="https"` with `secure=False` cannot connect, and `GrpcIndex` raises
`PineconeValueError` when you build it. With `grpc_scheme` unset, the scheme follows
`secure`.

## Basic operations

`GrpcIndex` has the same vector methods as the HTTP `Index`:

```python
response = index.upsert(
    vectors=[
        ("product-42", [0.012, -0.087, 0.153]),  # 1536-dim vector
        ("product-99", [0.045, 0.021, -0.064]),
    ],
    namespace="products-en",
)
print(response.upserted_count)

results = index.query(
    top_k=10,
    vector=[0.012, -0.087, 0.153],
    namespace="products-en",
)
for match in results.matches:
    print(match.id, match.score)
```

`upsert_records`, `search`, `search_records` and the bulk-import methods (`start_import`,
`describe_import`, `cancel_import`, `list_imports`, `list_imports_paginated`) are also on
`GrpcIndex`, but they go over REST because the gRPC API has no equivalent.

## Retries and timeouts

[Retries and Resilience](retries.md#transport-differences) compares the two transports.
Where gRPC differs:

**What it retries.** Three status codes: `UNAVAILABLE`, `RESOURCE_EXHAUSTED` and
`ABORTED`. The set is fixed; `RetryConfig.retryable_status_codes` holds HTTP statuses and
does not apply here. The defaults are `max_retries=5` and `backoff_factor=0.1`. A
`Pinecone()` created without `retry_config` does not pass REST's defaults to
`GrpcIndex`; pass a `RetryConfig` to `Pinecone()` or `GrpcIndex()` and it applies.

**`DEADLINE_EXCEEDED` is not retried.** A call whose deadline expires raises
`PineconeTimeoutError` after one attempt, so raising `max_retries` does nothing for it.
Set a longer deadline:

```python
from pinecone import RetryConfig
from pinecone.grpc import GrpcIndex

host = "product-search-abc123.svc.pinecone.io"

# No effect on deadlines: a timed-out call is never retried.
index = GrpcIndex(host=host, retry_config=RetryConfig(max_retries=20))

# `timeout` is the per-attempt deadline in seconds (default 20).
index = GrpcIndex(host=host, timeout=120.0)
```

A per-call `timeout=` applies alongside the index-level one, and the shorter fires. For
`upsert_from_dataframe`, `total_timeout` bounds the whole ingest.

**Backoff** is decorrelated jitter capped at `max_wait`, as on REST, but the first retry
draws from a window ten times wider so clients that fail together do not retry together.
When the server sends a pushback hint (`grpc-retry-pushback-ms` or `retry-after`), the
client waits that long instead, capped at `max_wait`, plus a small random spread so
concurrent clients do not retry in lockstep. A negative hint means do not retry, and the
call fails at once.

**Each `GrpcIndex` has its own retry budget.** Sustained retrying drains it, after which
retryable errors fail immediately instead of adding load to a struggling server.

**`retry_config` reaches only the gRPC channel.** The REST methods listed above keep
REST's default retry settings.

## Non-blocking calls with PineconeFuture

`upsert`, `query`, `fetch`, `delete`, `update` and `query_namespaces` each have an
`_async` variant that runs the call on a background thread and returns a
`PineconeFuture` at once. Issue several to overlap requests instead of queueing them:

```python
from concurrent.futures import as_completed

futures = [
    index.upsert_async(vectors=[("product-42", [0.012, -0.087, 0.153])]),
    index.upsert_async(vectors=[("product-99", [0.045, 0.021, -0.064])]),
]

for future in as_completed(futures):
    result = future.result()  # waits up to 5 seconds by default
    print(result.upserted_count)
```

A request still reaches the server if you never collect its future.

These are threads, not `await`. A `PineconeFuture` is not awaitable, and there is no
async gRPC client: `AsyncPinecone.index()` takes no `grpc` argument. Code running under
asyncio should use `AsyncIndex`; see [Sync vs Async](sync-vs-async.md).

`PineconeFuture` subclasses `concurrent.futures.Future`, so it works with
`as_completed()`, `wait()` and the usual `done()`, `cancel()` and `add_done_callback()`
methods. `result()` and `exception()` default to a 5-second timeout and raise
`PineconeTimeoutError` when it elapses; pass `timeout=None` to wait indefinitely:

```python
result = future.result(timeout=None)
```

`PineconeGrpcFuture` is an alias for `PineconeFuture`.

## Bulk upsert from a DataFrame

`upsert_from_dataframe()` splits a pandas `DataFrame` into batches and sends them
concurrently. pandas is not an SDK dependency; install it with `pip install pandas`.

```python
import pandas as pd
from pinecone.grpc import GrpcIndex

df = pd.DataFrame([
    {"id": "product-42", "values": [0.012, -0.087, 0.153]},
    {"id": "product-99", "values": [0.045, 0.021, -0.064]},
])

with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    response = index.upsert_from_dataframe(df, namespace="products-en", batch_size=500)
    print(response.upserted_count)
```

`timeout` is the deadline for one attempt of one batch; `None` uses the index's
`timeout`. `total_timeout` bounds the whole ingest. Raise `timeout` for large or slow
batches:

```python
with GrpcIndex(host="product-search-abc123.svc.pinecone.io") as index:
    response = index.upsert_from_dataframe(
        df, namespace="products-en", batch_size=200, timeout=120.0
    )
    print(response.upserted_count)
```

A failed batch does not stop the ingest; the returned `UpsertResponse` reports what
landed and what did not. See
[gRPC `upsert_from_dataframe` reports partial failures](../migration/v10-grpc-partial-failures.md)
for the response fields and a bounded retry loop.

## When to prefer gRPC

| Scenario | Recommendation |
|----------|---------------|
| Bulk upsert (thousands of vectors) | gRPC, for lower per-call overhead |
| High-throughput query loops | gRPC with the `*_async()` methods |
| asyncio code (FastAPI and similar) | `AsyncIndex`; `GrpcIndex` has no `await` support |
| Schema-based (document) indexes | HTTP `Index`; `GrpcIndex` has no `documents` namespace |
| Simple scripts and CLI tools | Either; HTTP `Index` needs no compiled extension |

## Limitations

- **No `documents` namespace.** `index.documents`, the entry point for schema-based
  (document) indexes, exists only on the HTTP `Index`.
- **Sync only.** For overlapping requests use the `*_async()` methods; under asyncio use
  `AsyncIndex`.
- **No `async_req=True` or `pool_threads`.** Those legacy options exist only on the HTTP
  `Index`, and `pc.index(..., grpc=True)` ignores `pool_threads`.
- **Some methods travel over REST**, as listed under [Basic operations](#basic-operations).
- **Retries are less configurable than on REST.** See
  [Retries and timeouts](#retries-and-timeouts).
- **Platform-specific wheels.** gRPC runs on a compiled extension bundled in the base
  package. If installation fails on your platform, check the published wheels for it.
