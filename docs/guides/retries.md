# Retries and Resilience

The SDK retries failed requests automatically. Three mechanisms stack on top of each other: a
per-request retry loop with decorrelated jitter, a per-host retry budget that suppresses retries
during an outage, and an adaptive concurrency gate that throttles bulk operations down when a host
pushes back.

Most of that runs on fixed policy. The one configurable piece, `RetryConfig`, reaches the control
plane and gRPC but not data-plane REST. Read [Where Retry Configuration Applies](#where-retry-configuration-applies)
before you reach for it.

For the exceptions the SDK raises when retries are exhausted, see {doc}`/guides/error-handling`.

---

## Defaults at a Glance

Out of the box, no configuration needed:

| What | Default behavior |
|------|------------------|
| Max retries (after initial attempt) | 3 for REST (4 total attempts), 5 for gRPC (6 total attempts) |
| HTTP methods retried | All of them: GET, HEAD, POST, PUT, PATCH, DELETE |
| Retryable HTTP status codes | 408, 429, 500, 502, 503, 504 |
| Retried transport failures | Any `httpx.TransportError`: connect, read, write, and pool timeouts, and connection resets |
| Retryable gRPC status codes | UNAVAILABLE, RESOURCE\_EXHAUSTED, ABORTED. DEADLINE\_EXCEEDED is **not** retried |
| Backoff algorithm | Decorrelated jitter: random walk floored at `backoff_factor` and capped at `max_wait` |
| `Retry-After` header | Honored when it parses as a number of seconds, capped at `max_wait`, plus a random smear. The HTTP-date form is not parsed |
| Retry budget | Per host, process-global. Retries are suppressed while a host is failing steadily |
| Adaptive concurrency (bulk paths) | Self-tunes downward on throttling. `max_concurrency` is a ceiling, not a constant |

`max_retries` counts retries, not attempts. The default of `3` means up to **four**
requests reach the network, and `max_retries=0` means exactly one.

### Writes are retried too

The SDK retries every method, including POST, because Pinecone's data-plane writes are
idempotent at the server: upsert overwrites by ID, and delete-by-ID and update-by-ID converge
on the same state however many times they land.

A retryable failure never tells you the write did *not* land: a response can be lost after
the server applied it. That is harmless for a data-plane write, but control-plane deletes are
not idempotent, so see [Retry it, or give up](error-handling.md#retry-it-or-give-up) before
you wrap one in a retry loop of your own.

---

## Where Retry Configuration Applies

`RetryConfig` is a constructor argument on `Pinecone` and `AsyncPinecone`. It does not reach every
request those clients make:

| Requests | Governed by |
|----------|-------------|
| Control plane (`pc.indexes`, `pc.collections`, `pc.backups`, `pc.inference`, `pc.assistants`) | The `retry_config` you passed to `Pinecone()` / `AsyncPinecone()` |
| gRPC data plane (`pc.index(grpc=True)`, `GrpcIndex`) | The same `retry_config`, but only when you set it explicitly. See [Transport differences](#transport-differences) |
| REST data plane (`Index`, `AsyncIndex`, including everything from `pc.index()`) | Built-in defaults only. `retry_config` is not forwarded, and neither class accepts one |

So this reduces the control plane to a single attempt and leaves the upsert at four:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(retry_config=RetryConfig(max_retries=0))

pc.indexes.list()                            # 1 attempt
idx = pc.index(name="product-search")
idx.upsert(vectors=[("doc-1", [0.1, 0.2])])  # still 4 attempts, the built-in default
```

`Index(retry_config=...)` and `AsyncIndex(retry_config=...)` raise `TypeError`. The knobs that
do change data-plane behavior are in [Tuning Data-Plane REST](#tuning-data-plane-rest).

The adaptive concurrency gate is not affected by any of this. It is fed from the transport itself,
so bulk operations self-tune against throttling on every path, configured or not.

---

## Configuring Retries

Pass a `RetryConfig` to the `Pinecone` constructor to customize control-plane retry behavior:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(
    retry_config=RetryConfig(
        max_retries=5,
        backoff_factor=0.5,
        max_wait=60.0,
        retryable_status_codes=frozenset({429, 500, 503}),
    )
)
```

### `RetryConfig` fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Number of retry attempts *after* the initial attempt. Total attempts = `max_retries + 1`. |
| `backoff_factor` | `float` | `0.25` | Minimum delay floor in seconds (lower bound of decorrelated jitter). See [Jitter strategy](#jitter-strategy) for the full formula. |
| `max_wait` | `float` | `60.0` | Maximum delay cap in seconds. Also caps how long a server's `Retry-After` can hold a request. |
| `retryable_status_codes` | `frozenset[int]` | `{408, 429, 500, 502, 503, 504}` | HTTP status codes that trigger a retry. The SDK retries on these codes and raises on all others. |

Those four are the whole configurable surface. `RetryConfig` carries a fifth field,
`on_throttle`, which is how the SDK wires its own adaptive-concurrency callback: the client
overwrites whatever you pass there, so it is not a hook you can use.

Transport-level failures are retried regardless of `retryable_status_codes`, which only classifies
responses that arrived. A connection reset or a read timeout never produces a status code, and the
retry loop treats every `httpx.TransportError` as retryable.

There is no method filter to configure, and no way to add one. See
[Writes are retried too](#writes-are-retried-too).

### Disabling retries

To disable retries entirely, set `max_retries=0`:

```python
from pinecone import Pinecone, RetryConfig

pc = Pinecone(retry_config=RetryConfig(max_retries=0))
```

With `max_retries=0`, control-plane calls make exactly one attempt and raise immediately on any
error. Data-plane REST calls keep their four attempts.

### Handling rate limits without retrying

By default, 429 responses are retried automatically. To receive `RateLimitError` immediately from
control-plane calls instead, exclude 429 from the retryable set:

```python
import time

from pinecone import Pinecone, RetryConfig
from pinecone.errors import RateLimitError

pc = Pinecone(
    retry_config=RetryConfig(
        retryable_status_codes=frozenset({408, 500, 502, 503, 504}),  # no 429
    )
)

schema = {"fields": {"your_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}}

try:
    pc.indexes.create(name="product-search", schema=schema)
except RateLimitError as exc:
    time.sleep(exc.retry_after or 30.0)
```

`RateLimitError.retry_after` carries the server's `Retry-After` in seconds, or `None` when
the header was absent or was an HTTP-date, which the SDK does not parse.

Data-plane `RateLimitError` still arrives only after the built-in four attempts are spent.

### Migration note: `backoff_factor` semantic change (v8 → v9)

In v8, `backoff_factor` was an exponential multiplier. Since v9 it is the floor of a
decorrelated-jitter window (see [Jitter strategy](#jitter-strategy)), so no value reproduces
v8's timing exactly. Leave it unset unless your code depends on retry timing.

---

## Tuning Data-Plane REST

The retry policy on `Index` and `AsyncIndex` is fixed, but three parameters still change how long a
data-plane call runs and how hard it pushes:

| Parameter | Where | Effect |
|-----------|-------|--------|
| `timeout` | `Pinecone(timeout=...)`, `Index(timeout=...)`, or per call | Deadline for one HTTP attempt. Each retry gets its own. |
| `total_timeout` | Bulk calls with `batch_size` set | Wall-clock deadline for the whole operation. Batches left unsent come back in `errors`. |
| `max_concurrency` | Bulk calls, default 8 | Ceiling on in-flight batches for that call. |

`total_timeout` is the closest thing to "stop retrying and hand control back to me" on the data
plane, because it bounds the operation rather than any single attempt:

```python
response = index.documents.batch_upsert(
    namespace="products-en",
    documents=documents,
    max_concurrency=4,
    total_timeout=30.0,
)
if response.has_errors:
    # Every item from every failed batch, flattened into one list.
    retry_later(response.failed_items)
```

`index.upsert(vectors=..., batch_size=...)` takes the same knobs on indexes created with
top-level `dimension=`/`metric=`. The deadline does not raise. Batches it cut off arrive as `BatchError` entries in
`response.errors`, each with `disposition="unsent"` and a `PineconeTimeoutError` as its
`error`; `response.failed_items` flattens their contents back into one list. Two other
dispositions can appear alongside them: `"abandoned"` for batches dropped by
[stall detection](#stall-detection), and `"rejected"` for a batch whose attempt actually
completed with an error.

Filter on `err.retryable` before re-sending anything. It is `False` for deterministic
failures, such as validation errors and 4xx rejections, that would fail identically on every
attempt:

```python
again = [
    item
    for err in response.errors
    if err.retryable
    for item in err.items
]
```

A single-request call such as `documents.upsert()` has no `total_timeout`. Bound it with `timeout` and accept
that the worst case is four attempts plus backoff.

---

## Jitter Strategy

Jitter spreads retries across time so that concurrent clients with the same retry budget
don't collide on the server at the same moment.

### Decorrelated jitter (backoff path)

When no server hint is present, the SDK uses decorrelated jitter:

```
upper = min(max_wait, prev_delay * 3)
delay = uniform(backoff_factor, max(backoff_factor, upper))
```

Starting from `prev_delay = backoff_factor`, each retry delay is drawn uniformly from
`[backoff_factor, prev_delay × 3]`. `max_wait` caps the window before the draw, so clients at
the cap still spread across the whole window. Because each window grows with the previous
delay, the delays follow a random walk rather than a fixed exponential schedule.

**Concrete example with defaults** (`backoff_factor=0.25`, `max_wait=60.0`):

| Attempt | Window (seconds) | Typical delay |
|---------|-----------------|---------------|
| 1st retry | [0.25, 0.75] | ~0.5 s |
| 2nd retry | [0.25, ~1.5] | ~0.9 s |
| 3rd retry | [0.25, ~2.6] | ~1.4 s |

### Retry-After (hinted path)

When a retryable response carries a `Retry-After` header the SDK follows it instead of drawing from
the backoff window. The header is read as a number of seconds; the HTTP-date form is not parsed, and
an unparseable or negative value falls back to decorrelated jitter. An accepted value is capped at
`max_wait`, then a random smear of up to half its length is added, so a fleet told to come back in
10 seconds arrives spread across 10 to 15 seconds rather than in one burst.

The delay the SDK actually waited seeds the next window either way, so one `Retry-After` hint widens
the subsequent backoff draws too.

---

## Retry Budget

Retrying harder is the wrong response to a host that fails every request, so each host has a
retry budget. Retryable failures spend it and successful responses slowly refill it. While it
is depleted, the SDK stops retrying that host. First attempts never consult the budget, so a
struggling host still gets your traffic without the retry multiplier on top. Against a host
that returns 429 to everything, calls quickly drop from four attempts to one.

The budget is keyed by hostname and shared process-wide, so two `Pinecone` clients pointed at
one host share it. It resets in a forked child.

gRPC runs the same policy, with the budget held per channel rather than process-wide. Two `GrpcIndex` objects against one host therefore keep separate ledgers,
where two `Index` objects share one.

---

## Adaptive Concurrency for Bulk Operations

During bulk calls, the SDK reduces the number of in-flight requests when a host throttles,
and recovers as throttling subsides.

### How it works

Every bulk call passes through a per-host admission gate. The gate's limit starts at 64 and
moves by AIMD (additive increase, multiplicative decrease):

- **Decrease.** A retryable response halves the limit, measured from the number of requests
  in flight.
- **Increase.** A run of successful batches raises the limit by one, but only while the gate
  was the thing holding requests back. Idle headroom does not accumulate.
- **Pushback.** A `Retry-After` on the throttling response blocks admission until it elapses.

**You don't configure this directly.** The `max_concurrency` you pass to a bulk call is a
per-call bound layered on top: your call never has more than its own bound outstanding, and global
admission works out to `min(max_concurrency, gate limit)`.

### Example

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index("product-search")

# max_concurrency=8 bounds this call. If the index throttles mid-run, the gate
# halves the host-wide limit from what is in flight (8, then 4, then 2) and
# recovers as throttling subsides.
response = index.documents.batch_upsert(
    namespace="products-en",
    documents=documents,
    max_concurrency=8,
)
print(response.successful_item_count)
```

### Stall detection

If batches keep failing outright with the limit already at its floor, the gate marks the host
stalled and refuses admission for a cool-down period. Waiting calls abandon their remaining
batches instead of queueing against a dead host. Those batches arrive in `errors` with
`disposition="abandoned"` and a message saying they were never sent, and
`response.failed_items` is what to re-submit once the host recovers. When the cool-down ends,
the gate probes again from the floor.

### Gate scope

One gate per host, process-global. Two `Pinecone` clients in one process targeting the same
index share it, and so do concurrent bulk calls from different threads or event loops. A forked
child starts with fresh gates.

`documents.batch_upsert()` reports what the gate did during the call (throttle signals, the
final limit, whether it stalled) on its `BatchResult`; see
[Reading the gate counters on the result](bulk-ingest.md#reading-the-gate-counters-on-the-result).
`index.upsert()` returns an `UpsertResponse`, which carries the per-batch outcome but nothing
about the gate.

---

## Transport Differences

REST and gRPC share the same retry shape. The differences are in the defaults, in what
`retryable_status_codes` means, and in which of them `retry_config` can reach:

| Aspect | REST control plane | REST data plane (`Index`, `AsyncIndex`) | gRPC (`GrpcIndex`) |
|--------|--------------------|------------------------------------------|---------------------|
| Default `max_retries` | 3 (4 total attempts) | 3 (4 total attempts) | 5 (6 total attempts) |
| Default `backoff_factor` | 0.25 | 0.25 | 0.1 |
| Configured via | `retry_config` passed to `Pinecone()` | Not configurable | `retry_config` on `Pinecone()` when set explicitly, or passed directly to `GrpcIndex()` |
| Retryable codes | `{408, 429, 500, 502, 503, 504}` | Same, fixed | UNAVAILABLE, RESOURCE\_EXHAUSTED, ABORTED (fixed); `retryable_status_codes` is ignored |
| Jitter algorithm | Decorrelated jitter (Python) | Decorrelated jitter (Python) | Decorrelated jitter (Rust) |
| Server hint honored | `Retry-After` | `Retry-After` | `grpc-retry-pushback-ms`, then `retry-after` |
| Async support | Yes (`AsyncPinecone`) | Yes (`AsyncIndex`) | No (gRPC transport is sync-only) |
| Adaptive concurrency | n/a | Yes, shared process-global gate | Yes, same gate |

**A `retry_config` you pass to `Pinecone()` is only forwarded to gRPC if you set it
explicitly.** gRPC's defaults differ from REST's, so leaving `retry_config` unset keeps
`GrpcIndex` on its own numbers rather than inheriting REST's. To configure gRPC independently of
the control-plane client, construct `GrpcIndex` directly and pass `retry_config` to it.

Three gRPC behaviors have no REST counterpart:

- **DEADLINE_EXCEEDED is not retried.** A gRPC call that runs out of time raises rather than
  going round again, and `max_retries` is not the knob for it: raise `timeout` instead. The
  three codes that *are* retried compound with `timeout`, so under the default
  `max_retries=5` a batch that keeps failing on them can run to six attempts plus backoff.
- **The first retry draws from a wider window.** gRPC seeds the jitter window at ten times
  `backoff_factor` rather than at `backoff_factor`, so the first draw with the defaults spans
  roughly 0.1 to 3 seconds instead of 0.1 to 0.3. This spreads clients out after a backend
  restart, which returns UNAVAILABLE to every client at once.
- **A negative pushback means stop.** `grpc-retry-pushback-ms: -1` is the server refusing the
  retry, and the transport fails fast instead of backing off.

The set of retryable gRPC codes is fixed from Python. `RetryConfig.retryable_status_codes`
carries HTTP status codes and is not translated onto this transport.

---

## Multi-Process and Serverless Workloads

### What the SDK cannot do

The gate and the retry budget are process-global, not machine- or fleet-global. If your workload
fans out across multiple Lambda invocations, Cloud Run instances, or Kubernetes pods, each process
runs its own retry loop and its own gate. There is no shared state across processes, no
cross-process coordination, and no distributed rate-limit awareness.

This means:

- N simultaneously throttled invocations each independently back off and retry. Without
  coordination, they can collide again at the end of the retry window.
- The gate starts from a limit of 64 in each new process instance, for example a fresh Lambda cold
  start. It cannot inherit a reduced limit that another invocation learned from throttling.
- The retry budget starts full in each new process, so a host that a sibling invocation already
  found to be down still costs this one its full four attempts.

### Recommended pattern for fan-out workloads

Let your orchestrator handle retries at the job level, and keep each invocation's window narrow with
`total_timeout` rather than with `retry_config`, which does not reach data-plane REST:

```python
from pinecone import Pinecone

pc = Pinecone()
index = pc.index("product-search")

# Bound the operation, not the individual attempt. Let the SQS visibility
# timeout / Cloud Tasks retry / Step Functions catch handle the outer loop.
response = index.documents.batch_upsert(
    namespace="products-en",
    documents=documents,
    max_concurrency=4,
    total_timeout=20.0,
)
if response.has_errors:
    # Fail the task so the orchestrator reschedules it after the visibility
    # timeout, and carry the unsent items forward.
    raise RuntimeError(f"{response.failed_item_count} items unsent")
```

For control-plane work in the same handler, `RetryConfig(max_retries=1)` on the client does what you
would expect.

### Why jitter still helps across processes

Even without coordination, decorrelated jitter spreads the retries of independently
throttled processes across a window instead of one instant, and a `Retry-After` hint is
smeared across up to 1.5× the requested wait.

### Summary: when to trust the SDK vs. the orchestrator

| Scenario | Recommended approach |
|----------|----------------------|
| Single-process bulk upsert | Use defaults; SDK handles everything |
| Long-running worker (persistent process) | Use defaults; the gate learns and recovers, and the retry budget refills |
| Lambda / Cloud Functions / Cloud Run (stateless) | Set `total_timeout` on bulk calls, re-raise on `has_errors` for orchestrator retry |
| Fan-out across many pods (e.g. Kubernetes Job) | Same as stateless, plus a lower `max_concurrency` per pod |
| Strict per-invocation SLA (must not block) | Small `total_timeout`, or `timeout` on non-batched calls; hand `failed_items` back to the orchestrator |

---

## Observability

The SDK logs retries and throttling with `key=value` fields in each message.

### Log namespaces

| Logger | Level | Events | Fires for |
|--------|-------|--------|-----------|
| `pinecone._internal.http_client` | DEBUG | Throttled response received, with the delay computed; connection error retried | Every REST request, control plane and data plane |
| `pinecone._internal.adaptive` | INFO, DEBUG | First throttle from a host (INFO); each concurrency-limit reduction (DEBUG) | Control-plane REST and gRPC only |
| `pinecone._internal.bulk.engine`, `.async_engine` | WARNING | Batches abandoned because the host looked unavailable | Bulk calls on any transport |

### INFO messages

An INFO-level record is emitted the **first time** a given host rate-limits a client
instance:

```
Rate limited by host=<host>. Adaptive concurrency will reduce in-flight requests.
See https://docs.pinecone.io/python/retries for details.
```

It fires once per host per `Pinecone` / `AsyncPinecone` object. Data-plane REST hosts do not
produce it, because the callback behind it rides on the `retry_config` that `Index` and
`AsyncIndex` never receive. The DEBUG records below are how you see data-plane throttling.

### DEBUG messages

Enable DEBUG-level logging on `pinecone._internal.http_client` to see granular retry events.
The SDK installs no log handler, so configure one too:

```python
import logging
logging.basicConfig()
logging.getLogger("pinecone._internal.http_client").setLevel(logging.DEBUG)
```

**Throttle record** (emitted once per retry attempt that receives a retryable response):

```
Throttled response: status=429 host=my-index.svc.pinecone.io attempt=1/4 delay=0.531s retry_after=absent
```

Fields: `status` (HTTP status code), `host`, `attempt` (N of total attempts),
`delay` (computed wait in seconds), `retry_after` (the response's `Retry-After` header
value, or `absent`).

**Connection error record** (emitted when a transport failure is retried):

```
Connection error on attempt 1/4, retrying: All connection attempts failed
```

Neither record is emitted for the final attempt, since nothing is retried after it. A request that
uses all four attempts logs three records and then raises.

**Limit reduction record**, on `pinecone._internal.adaptive` (control-plane REST and gRPC
only, per the table above):

```
AIMD limiter decreased: before=8 after=4 ceiling=8
```

For a data-plane REST bulk call, read the gate counters on the `BatchResult` instead.

---

## See Also

- {doc}`/guides/error-handling`: Exception hierarchy and how to catch specific errors
- {doc}`/guides/performance`: Bulk upsert patterns, `max_concurrency` tuning, and transport selection
- {doc}`/guides/sync-vs-async`: When to use the async client and how to manage concurrency with `asyncio`
