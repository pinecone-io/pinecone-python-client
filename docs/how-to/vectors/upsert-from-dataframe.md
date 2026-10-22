# Large ingests with `upsert_from_dataframe`

{meth}`~pinecone.index.Index.upsert_from_dataframe` takes a pandas DataFrame, splits it
into batches, sends the batches in parallel, and reports which rows landed and which did
not. This page covers sizing the batches, bounding the job in time, and handling partial
failure.

It serves indexes created with the deprecated top-level `dimension=`/`metric=`. On a
schema-based (document) index, write through {attr}`~pinecone.index.Index.documents`
instead.

The method has the same signature on {class}`~pinecone.index.Index`,
{class}`~pinecone.async_client.async_index.AsyncIndex` and
{class}`~pinecone.grpc.GrpcIndex`, so the examples carry over apart from the `await`. For
how the SDK paces and retries batches, see [How bulk ingest
behaves](../../guides/bulk-ingest.md).

:::{note}
This method writes through {meth}`~pinecone.index.Index.upsert`, so it serves indexes
created with top-level `dimension` and `metric`. The server rejects it on an index
created with a `schema`; use `index.documents` and see the
[quickstart](../../getting-started/quickstart.md#choosing-an-interface) for how the two
interfaces divide.
:::

:::{important}
`pandas` is not an SDK dependency. Install it yourself; without it the call raises a
`RuntimeError`. To skip pandas, {meth}`~pinecone.index.Index.upsert` with `batch_size`
does the same parallel batching from a list of vectors; see
[Large datasets](upsert-and-query.md#large-datasets).
:::

```{contents}
:local:
:depth: 2
```

## When to use it

| Your situation | Reach for |
|---|---|
| Vectors in a pandas DataFrame | `upsert_from_dataframe()`, this page |
| Vectors in a list, and no pandas | [`upsert(batch_size=...)`](upsert-and-query.md#large-datasets) |
| Far more rows than you want to stream, already in cloud storage | [`start_import()`](bulk-import.md) |

## The DataFrame contract

Required columns: `id` (string) and `values` (list of floats, matching the index
dimension). Optional columns: `sparse_values` and `metadata`. Any other column is ignored,
so move fields you want stored into `metadata`.

A missing cell (`None` or `NaN`) in an optional column leaves that field off the vector.
That matters when you build a frame from row dicts where only some rows carry `metadata`:
pandas fills the gaps with `NaN`, and those rows are written without metadata.

```python
import pandas as pd

df = pd.DataFrame(
    [
        {"id": "article-101", "values": [0.012, -0.087, 0.153], "metadata": {"lang": "en"}},
        {"id": "article-102", "values": [0.045, 0.021, -0.064]},
    ]
)
```

The values are truncated to three floats here; pass your index's full dimension.

## Minimal usage

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("article-search")

response = index.upsert_from_dataframe(df, namespace="articles-en")
print(response.upserted_count)
```

Omitting `namespace` writes to the default namespace; see
{doc}`/how-to/vectors/namespaces`. By default a failed batch is reported on the response
rather than raised, and there is no overall deadline.

## Sizing the knobs

```python
response = index.upsert_from_dataframe(
    df,
    namespace="articles-en",
    batch_size=200,
    max_concurrency=4,
    timeout=30,
    total_timeout=1800,
    on_error="collect",
)
```

The reference for {meth}`~pinecone.index.Index.upsert_from_dataframe` lists each
parameter and its default. How to choose values:

### `batch_size`: rows per request

Lower it when rows are heavy (high-dimension vectors or large metadata) so each request
stays inside the server's caps on vector count and request size, and each retry re-sends
less. Raise it for small vectors when you want fewer round trips, and check the result:
the server, not the SDK, decides when a request is too big. A batch succeeds or fails as
a unit, so `batch_size` is also your unit of retry.

### `max_concurrency`: parallelism ceiling

This is a ceiling, not a target: the SDK holds concurrency lower on its own when the
server pushes back, then recovers. Lower it when the ingest shares an index with
latency-sensitive query traffic. Raise it only after measuring; see
[The knobs](../../guides/bulk-ingest.md#the-knobs).

### `timeout`: one attempt of one batch

This bounds a single request attempt. The SDK retries a batch that times out like any
other transient failure, so one batch can take several multiples of `timeout` before it
is reported as failed. Raise `timeout` when batches are large or the network is slow.
{doc}`/guides/retries` explains what is retried and which retry settings reach the data
plane.

Do not use `timeout` to bound the whole job; that is what `total_timeout` is for.

### `total_timeout`: the whole job's deadline

Set it whenever the ingest runs inside a real window, such as a cron slot, a deploy step
or a pipeline stage. On expiry:

- The SDK stops submitting new batches.
- Batches already in flight finish; none is cancelled mid-write.
- Every row not sent comes back in `response.failed_items`, with
  `disposition="unsent"` on its batch's error.

`failed_items` is a flat list of dicts keyed `id`, `values`, `sparse_values` and
`metadata`, so `pd.DataFrame(response.failed_items)` rebuilds a frame you can pass
straight back in.

There is no default deadline, because a slow but progressing ingest is healthy. The SDK's
stall detector gives up early when the backend stops responding, but it does not cover
every outage, so `total_timeout` is the only guaranteed wall-clock bound.

## Recipe: the nightly job

Bounded in time, with leftover rows saved for the next run:

```python
response = index.upsert_from_dataframe(
    df,
    namespace="articles-en",
    total_timeout=25 * 60,  # headroom inside a 30-minute slot
)

if response.has_errors:
    pd.DataFrame(response.failed_items).to_parquet("retry-tomorrow.parquet")
    print(
        f"upserted {response.upserted_count}/{response.total_item_count}; "
        f"{response.failed_item_count} rows deferred"
    )
```

Re-sending the leftover rows is safe: an upsert overwrites by vector ID, so a row that
landed just before a timeout is overwritten with identical data.

## Recipe: retry within the same run

Bound the number of rounds, resend only batches marked `retryable`, and back off between
rounds. A loop on `while response.failed_items:` never ends when a batch fails the same
way every time.

```python
import time

response = index.upsert_from_dataframe(df, namespace="articles-en", total_timeout=1800)

for attempt in range(3):
    retryable = [item for err in response.errors if err.retryable for item in err.items]
    if not retryable:
        break
    time.sleep(2**attempt)
    response = index.upsert_from_dataframe(
        pd.DataFrame(retryable), namespace="articles-en", total_timeout=600
    )
```

A batch with `retryable=False` failed deterministically, for example a validation error
or another 4xx rejection, and fails the same way if re-sent. Log it and move on.

## Recipe: fail loudly instead

Pass `on_error="raise"` when a pipeline should stop on failure. Once every batch has
settled, the SDK re-raises the lowest-indexed batch's exception with the partial
{class}`~pinecone.models.vectors.responses.UpsertResponse` attached as `.response`. The
exception is whatever that batch raised: {exc}`~pinecone.errors.exceptions.PineconeTimeoutError`
for a batch that ran out of retries on timeouts, or an
{exc}`~pinecone.errors.exceptions.ApiError` subclass for a rejection.

```python
from pinecone import PineconeTimeoutError

try:
    index.upsert_from_dataframe(
        df, namespace="articles-en", total_timeout=1800, on_error="raise"
    )
except PineconeTimeoutError as exc:
    pd.DataFrame(exc.response.failed_items).to_parquet("retry.parquet")
    raise
```

## Recipe: asyncio

On {class}`~pinecone.async_client.pinecone.AsyncPinecone`, `index()` is a coroutine, so
await it:

```python
from pinecone import AsyncPinecone

async with AsyncPinecone(api_key="your-api-key") as pc:
    index = await pc.index("article-search")
    async with index:
        response = await index.upsert_from_dataframe(
            df, namespace="articles-en", total_timeout=1800
        )
```

## Reading the response

| Field | Meaning |
|---|---|
| `upserted_count` | Rows the server accepted |
| `total_item_count` / `failed_item_count` | Rows submitted / rows in a batch that failed |
| `has_errors` | `True` when any batch failed |
| `failed_items` | The rows from failed batches, ready to send again |
| `errors` | One {class}`~pinecone.models.batch.BatchError` per failed batch |

Each error's `disposition` says how far its batch got:

- `"rejected"`: the request reached the server and came back with an error, so the
  write may have landed anyway.
- `"unsent"`: `total_timeout` expired before the batch was sent.
- `"abandoned"`: the stall detector judged the backend unavailable and dropped the
  remaining batches without sending them. Treat it like any other partial failure and
  retry later.

Match the values you care about and let others fall through; new ones can appear.

## Checklist for a large production ingest

- [ ] `pandas` installed in the environment that runs the job
- [ ] `id` and `values` columns present, and your own fields moved into `metadata`
- [ ] `batch_size` sized to your row weight
- [ ] `total_timeout` set if the job has a real window
- [ ] `on_error` chosen: `"collect"` and persist `failed_items`, or `"raise"`
- [ ] Retry path bounded and filtered on `err.retryable`
