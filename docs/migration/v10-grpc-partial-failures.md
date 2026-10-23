# 10.0: gRPC `upsert_from_dataframe` reports partial failures instead of raising

In 9.1 and earlier, `GrpcIndex.upsert_from_dataframe` raised as soon as any batch
failed, and the exception did not say how many rows had landed. From 10.0 it
returns the partial result instead, as `upsert(batch_size=...)` and the REST
transport already did.

## What changed

```python
import pandas as pd

# 9.1 and earlier, gRPC
try:
    index.upsert_from_dataframe(df)
except Exception:
    # No way to tell how much landed. Re-run the whole frame.
    ...

# 10.0, gRPC and REST
response = index.upsert_from_dataframe(df)
if response.failed_item_count:
    for error in response.errors:
        print(error.error_message)
    index.upsert_from_dataframe(pd.DataFrame(response.failed_items))
```

`UpsertResponse` already carried `upserted_count`, `failed_item_count`, `errors`
and `failed_items`; the gRPC transport now fills them in.

### Retrying failures safely

Do not retry in an unbounded loop such as
`while response.failed_items: ...`. During an outage it re-sends the frame back
to back with no pause. Cap the attempts, back off between them, and re-send
only retryable failures:

```python
import time

import pandas as pd

remaining = df
for attempt in range(5):
    response = index.upsert_from_dataframe(remaining)
    retryable = [
        item
        for err in response.errors
        if err.retryable
        for item in err.items
    ]
    if not retryable:
        break
    remaining = pd.DataFrame(retryable)
    time.sleep(2**attempt)
else:
    raise RuntimeError(f"{len(remaining)} rows still failing after 5 attempts")
```

A batch rejected for a deterministic reason, such as a 400, fails the same way
on every attempt, which is why the loop filters on `err.retryable`.
`err.disposition` says how each batch failed (for example `"rejected"`,
`"unsent"` or `"abandoned"`; treat the set as open). `"abandoned"` means the SDK
concluded the backend was unavailable and stopped sending, so wait before
retrying.

The SDK already retries and backs off within a single call; the cap across
calls is yours to set. When `total_timeout` expires, rows that were never sent
also appear in `failed_items`, so pairing `total_timeout` with an unbounded
retry loop re-sends on a fixed period without end.

## If you depended on the raise

Pass `on_error="raise"`:

```python
index.upsert_from_dataframe(df, on_error="raise")
```

This re-raises the lowest-indexed batch failure, as 9.1 did, but only after
every batch has settled, and it attaches the partial result to the exception:

```python
try:
    index.upsert_from_dataframe(df, on_error="raise")
except Exception as exc:
    print(exc.response.upserted_count)
    retry_these = exc.response.failed_items
```

`on_error="raise"` is a supported choice for fail-fast ingest on both
transports, not a temporary shim.

## Finding out at runtime

The first time a gRPC ingest hits a partial failure without an explicit
`on_error`, the SDK warns once per process, naming `response.errors` and
`on_error="raise"`. Passing `on_error` explicitly, with either value, silences
the warning. REST does not warn, because its behavior did not change.

## Related

- `total_timeout` follows `on_error` the same way: under `"collect"` an expired
  deadline logs a warning and returns the partial response, and under `"raise"`
  it raises `PineconeTimeoutError` with the same partial result on its
  `response` attribute. [Bulk ingest](../guides/bulk-ingest.md) covers the
  deadline itself.
- [Using the gRPC client](../guides/grpc.md) covers the transport this change
  applies to.
