# How Bulk Ingest Behaves

This page explains what happens when you hand the SDK a large amount of data
in one call, and what each knob controls. The batched calls are
`documents.batch_upsert()` for a document index, and `upsert(vectors=...,
batch_size=...)` or `upsert_from_dataframe()` for an index created with
top-level `dimension=`/`metric=`. For transport and throughput tuning, see
[Performance](performance.md); for per-request retry settings, see
[Retries](retries.md); for DataFrame recipes, see
[Reliable Large Ingests with upsert_from_dataframe](../how-to/vectors/upsert-from-dataframe.md).

The REST sync, asyncio and gRPC clients share one bulk engine, so everything
here applies to all three. Vectors and documents name things differently:

| | Vectors | Documents |
|---|---|---|
| Batched call | `upsert(vectors=..., batch_size=...)`, `upsert_from_dataframe()` | `documents.batch_upsert()` |
| Returns | `UpsertResponse` | `BatchResult` |
| Accepted-item count | `upserted_count` | `successful_item_count` (alias `success_count`) |
| Default `batch_size` | unbatched for `upsert`; 500 for `upsert_from_dataframe` | 50; the server accepts at most 1000 per request |

`documents.upsert()` is a single request, so nothing on this page applies to it.

## How a batched call runs

The SDK splits your input into batches of `batch_size` items. A per-host
admission gate decides how many batches may be in flight at once; this is
where backpressure lives. Each admitted batch is one request with its own
automatic retries. When every batch has settled, the call returns one result
that counts what landed and lists what failed. Per-batch failures are
reported, not raised.

## Why your ingest slows down sometimes

The gate keeps a concurrency limit for each host that adapts to what the
backend can absorb:

- When the server signals overload (HTTP 429, or a gRPC "slow down"), the
  limit drops sharply. If the server says to retry after some interval, the
  gate holds every new batch until that interval passes.
- When batches succeed steadily, the limit recovers one step at a time.

A slowdown mid-ingest usually means the SDK heard the backend ask for room
and gave it. The ingest finishes sooner than it would by hammering a server
that is shedding load, and other clients of the same index stay healthy.

The gate is shared by every bulk call to the same host in the process, so
two `Index` handles pointed at the same index cannot double the load on it.
Retries are budgeted per host as well: once retryable failures outnumber
successes by enough, the SDK stops retrying, so retries switch themselves off
during a full outage instead of multiplying load.

## The knobs

### `batch_size`

How many items go in one request. Defaults are in the table above.

Bigger batches mean fewer but heavier requests, and a larger unit of failure
(a batch succeeds or fails as a whole). The ceiling is the server's request
size limit, which a batch of large items reaches sooner than the item count
suggests. Lower it when your items are large; raise it when they are tiny.

### `max_concurrency`

The most batches this call keeps in flight at once. Default 8 on every
method and transport.

This is a cap, not a target: the gate can hold effective concurrency below
it while the backend pushes back. Raising it raises the best case, and also
the retry burst you commit to during an outage. See
[Tuning max_concurrency](performance.md#tuning-max_concurrency).

### `total_timeout`

A deadline in seconds for the whole call. `timeout`, by contrast, bounds one
attempt of one batch. Default `None`: no wall-clock bound.

When the deadline passes, the SDK stops submitting batches. Batches already
in flight finish and are never cancelled, so `total_timeout` never tears down
work the server may already be applying. Everything unsent comes back as
failed items you can retry.

There is no default deadline because a slow but progressing ingest should be
allowed to finish; a dead backend is caught by the stall detector (below),
which watches progress rather than the clock. Set `total_timeout` when your
job has a real external deadline, such as a cron slot, and size it
generously.

### `timeout`

The per-attempt timeout for one batch request. A batch that times out is
retried like any other transient failure, so one batch can take several
multiples of `timeout`, plus backoff, before it is reported failed. Raise it
for very large batches on slow networks; to bound the whole ingest, use
`total_timeout`.

## When the backend is actually down: the stall detector

If the gate has dropped to its floor and several consecutive batches then
fail outright (every retry exhausted, no successes), the SDK concludes the
backend is unavailable and abandons the rest of the call instead of grinding
through every remaining batch:

- The result carries the unsent work in `failed_items`, and the abandonment
  errors say `backend appears unavailable`.
- A warning is logged with how many batches were abandoned.
- For a short cool-down, further bulk calls to the same host fail fast the
  same way. After it, the SDK probes again and recovers on the first success.

Treat it like any partial failure: inspect the result, wait or alert, and
retry the failed items later (pattern below). The detector never trips on a
backend that is slow but working.

## Partial failures: the reporting contract

Batched calls do not raise on per-batch failures. The result tells you what
happened:

- `upserted_count` (vectors) or `successful_item_count` (documents): items
  the server accepted.
- `failed_item_count` and `failed_items`: exactly what did not land, ready
  to feed back into a retry call.
- `errors`: one entry per failed batch (several can share a `batch_index`
  when `documents.batch_upsert` split an oversize batch and only some pieces
  failed). Each carries the items, the error, a `retryable` hint, and a
  `disposition` saying how it failed:
  - `rejected`: the attempt completed with an error, after retries. The
    write may still have landed, because a response can be lost after the
    server applied it.
  - `unsent`: never attempted because `total_timeout` passed first.
  - `abandoned`: never attempted because the stall detector fired.

  Dispositions are an open set. Branch on the values you handle and send any
  other value to your default path.

`upsert_from_dataframe` accepts `on_error="raise"` if you would rather have
the lowest-indexed batch failure re-raised after all batches settle; the
partial result is on the exception's `response` attribute.

## Reading the gate counters on the result

`documents.batch_upsert()` also reports what the admission gate did during the
call. The gate is shared by every bulk call to the same host in the process.

| Field | Meaning | How to read it |
|---|---|---|
| `throttle_event_count` | Throttle signals (HTTP 429, gRPC "slow down") the host's gate saw during the call | Includes throttles from concurrent calls to the same host. Nonzero means the backend pushed back, not that the call failed. |
| `peak_inflight` | Most batches this call had in flight at once | Never exceeds `max_concurrency`. Well below it means the gate or a small batch count held the call back. |
| `final_limit` | The host gate's concurrency limit when the call finished, from 1 to 64; `None` if the call never used the gate | Compare with 64, not `max_concurrency`: the gate starts at 64 regardless. Lower means the gate backed off, possibly during an earlier call in the same process. |
| `stalled` | `True` if the stall detector (above) fired and the rest of the call was abandoned | Abandoned batches are in `errors` with `disposition="abandoned"`. |

## Retrying safely

Upserts are idempotent by id: re-sending an item that already landed
overwrites it with the same data, so retrying `failed_items` is safe.
Retrying in an unbounded loop is not, because an unhealthy backend keeps it
spinning forever. Bound the attempts, keep only the items whose batch is
marked `retryable`, and back off between rounds:

```python
import time

from pinecone import Pinecone

pc = Pinecone()
index = pc.index(name="product-catalog")

result = index.documents.batch_upsert(namespace="products-en", documents=documents)

for attempt in range(3):
    if not result.has_errors:
        break
    retryable = [item for err in result.errors if err.retryable for item in err.items]
    if not retryable:
        break
    time.sleep(2**attempt)
    result = index.documents.batch_upsert(namespace="products-en", documents=retryable)
```

For vectors, call `index.upsert(vectors=..., batch_size=...)` in both places;
`UpsertResponse` has the same `has_errors` and `errors`.

A batch marked `retryable=False` was rejected for a reason a retry cannot
fix, such as malformed data, a dimension mismatch or bad credentials.
Re-sending it only spends your retry budget; log it and move on.

## What you don't have to manage

- **Thundering herds.** Retry delays are jittered, and server `Retry-After`
  hints are spread out, so many clients told to come back at the same moment
  do not all return at once.
- **Coordination within one process.** The gate and the retry budget are
  shared per host, so concurrent bulk calls share capacity instead of
  competing for it.
- **Transport differences.** REST sync, asyncio and gRPC run the same engine
  with the same knobs and the same reporting contract. Choose a transport on
  its own merits ([Sync vs Async](sync-vs-async.md), [gRPC](grpc.md)), not
  for bulk behavior.
