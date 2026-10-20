# Upgrading from 10.0 to 10.1

10.1 removes nothing and deprecates nothing. Most of what changed is when a
mistake gets reported: calls that 10.0.0 sent to the server as given, or
quietly reshaped, now raise before a request goes out. One change can stop
working code without raising: a loop that polls `documents.fetch` to wait for
a write never exits. If you wait for writes to become visible, start with
[Waiting for a write to become visible](#v10-1-read-your-writes).

Coming from `9.x`? Read [Migrating to V10](v10-migration.md) first, then this
page.

The snippets run in order against one client. `movies` is a document index
whose schema declares `title` as a `string` field with full-text search:

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.index("movies")
```

## Contents

- [Waiting for a write to become visible](#v10-1-read-your-writes)
- [Documents API arguments are checked before sending](#v10-1-documents-arguments)
- [Timeouts must be positive](#v10-1-timeouts)
- [Names and IDs must be strings](#v10-1-path-parameters)
- [Creating a client](#v10-1-clients)
- [Bulk document upserts](#v10-1-bulk-upsert)
- [Summary](#v10-1-summary)

(v10-1-read-your-writes)=
## Waiting for a write to become visible

10.1.0 reads the log-position headers the Documents API sends. A documents
upsert, update or delete response now carries
`response_info.lsn_committed`, and a documents search response carries
`response_info.lsn_reconciled`. In 10.0.0 both were always `None` on the
Documents API.

That breaks the read-your-writes loop the 10.0.0 `ResponseInfo` reference
showed:

```python
# 10.0.0 recipe: never exits on 10.1.0
import time

written = index.documents.upsert(
    namespace="movies-en", documents=[{"_id": "movie-42", "title": "Arrival"}]
)
target = written.response_info.lsn_committed

while True:
    fetched = index.documents.fetch(namespace="movies-en", ids=["movie-42"])
    if target is None or fetched.response_info.is_reconciled(target):
        break
    time.sleep(0.5)
```

On 10.0.0, `target` was `None`, so the loop exited at once without waiting
for anything. On 10.1.0, `target` is a number, but a fetch response never
reports a reconciled position, so `is_reconciled(target)` stays `False` and
the loop runs forever. Nothing raises or warns.

Poll `documents.search` instead, and give the loop a deadline:

```python
import time

from pinecone import TextQuery

written = index.documents.upsert(
    namespace="movies-en", documents=[{"_id": "movie-42", "title": "Arrival"}]
)
target = written.response_info.lsn_committed

deadline = time.monotonic() + 30
while target is not None:  # None: caught-up cannot be confirmed
    found = index.documents.search(
        namespace="movies-en",
        top_k=1,
        score_by=[TextQuery(query="arrival", fields=["title"])],
    )
    if found.response_info.is_reconciled(target):
        break
    if time.monotonic() > deadline:
        raise TimeoutError(f"search had not reached LSN {target} after 30s")
    time.sleep(0.5)
```

Fetch and search can see a new write at different times, so a fetch that
returns the document does not mean a search will find it. After
`documents.batch_upsert`, use `result.response_info.lsn_committed` as the
target; it is the highest position any of the batch's requests reached.

(v10-1-documents-arguments)=
## Documents API arguments are checked before sending

A dict `score_by` clause with a key its type does not define now raises
`PineconeValueError`. 10.0.0 dropped the key and sent the rest, so an option
such as `boost` silently did nothing:

```python
index.documents.search(
    namespace="movies-en",
    top_k=5,
    score_by=[{"type": "text", "fields": ["title"], "query": "arrival", "boost": 2}],
)
# PineconeValueError: Object contains unknown field `boost`
```

A bare `str` where a list of names belongs now raises `PineconeTypeError`.
10.0.0 split `ids`, `include_fields` and `remove_fields` into one entry per
character, so `ids="movie-42"` deleted the IDs `"m"`, `"o"`, `"v"` and so on.
A clause's `fields` given as one string, which 10.0.0 sent as is, is refused
the same way:

```python
index.documents.delete(namespace="movies-en", ids="movie-42")
# PineconeTypeError: ids must be a sequence of strings, not a single str; pass ['movie-42'] to name one entry.

TextQuery(query="arrival", fields="title")
# PineconeTypeError: 'fields' must be a list of field names, not a single str; pass ['title'].
```

Three more wrong-type arguments now raise `PineconeTypeError` with a message
that names the fix:

| Call | 10.0.0 | 10.1.0 |
| --- | --- | --- |
| `score_by=TextQuery(...)`, not in a list | `TypeError: 'TextQuery' object is not iterable` | `PineconeTypeError` saying to wrap the clause in a list |
| `score_by=[clause, 5]`, an item that is not a clause or a dict | sent to the server, or a misleading `PineconeValueError` about combining clauses | `PineconeTypeError` naming the item and the clause classes |
| `top_k` or `limit` that is not an `int`, such as `"5"` or `5.0` | a bare `TypeError`, or sent to the server | `PineconeTypeError` |

`PineconeTypeError` is a `TypeError`, so an `except TypeError` around these
calls still catches them.

(v10-1-timeouts)=
## Timeouts must be positive

A `timeout` or `total_timeout` of zero, a negative number, or NaN now raises
`PineconeValueError` before anything is sent. This covers the client-level
`timeout` on `Pinecone` and `AsyncPinecone`, the per-call `timeout` on every
Documents API method, and `total_timeout` on `documents.batch_upsert`,
`upsert(batch_size=...)` and `upsert_from_dataframe`. 10.0.0 passed these
values through: `timeout=0` reached the HTTP layer, and `total_timeout=0`
returned a result with every item failed and `timed_out` set.

```python
index.documents.upsert(
    namespace="movies-en", documents=[{"_id": "movie-42", "title": "Arrival"}], timeout=0
)
# PineconeValueError: timeout must be a positive number of seconds, got 0
```

`None` still means the default (no deadline, for `total_timeout`).

The wait `timeout` on `pc.indexes.create()`, `create_for_model()` and
`delete()` changed too. A negative value other than `-1` now raises
`PineconeValueError` before the request. 10.0.0 sent the request, so the
index was created or deleted anyway, and then raised `PineconeTimeoutError`
at the first check that did not find the operation finished:

```python
pc.indexes.create(
    name="movies-v2",
    schema={"fields": {"title": {"type": "string", "full_text_search": {}}}},
    timeout=-5,
)
# PineconeValueError: timeout must be None (wait indefinitely), -1 (do not wait), or a non-negative number of seconds, got -5. Pass -1 to skip waiting.
```

Pass `-1` to return without waiting, `None` to wait as long as it takes, or a
positive number of seconds.

(v10-1-path-parameters)=
## Names and IDs must be strings

A name or ID the SDK puts into a URL (an index name, a backup ID, an assistant
name or file ID, an admin resource ID) must be a `str`. Anything else now
raises `PineconeTypeError` naming the parameter and the type it got, before
any request:

```python
import uuid

pc.backups.list(index_name=1234)
# PineconeTypeError: index_name must be a string, got int

pc.assistants.describe(name=uuid.uuid4())
# PineconeTypeError: name must be a string, got UUID

pc.indexes.describe(None)
# PineconeTypeError: name must be a string, got NoneType
```

What each case did in 10.0.0:

- `backups.list` and the assistant methods converted the value with `str()`
  and sent it, asking for `GET /indexes/1234/backups`, a resource that cannot
  exist.
- Elsewhere a non-string escaped as a bare `AttributeError`
  (`'int' object has no attribute 'strip'`) or a `TypeError` from the URL
  encoder.
- `None` raised `PineconeValueError: name must be a non-empty string`. An
  `except PineconeValueError` (or `except ValueError`) written for that no
  longer catches it; catch `PineconeTypeError` as well. An empty string still
  raises `PineconeValueError`.

The same check covers twelve arguments that never reach the URL, which move
from `AttributeError` in 10.0.0 to `PineconeTypeError`:
`create_index_from_backup(name=)`, `indexes.create_for_model(cloud=, region=)`,
`collections.create(source=)`, `backup_schedules.create(name=)`,
`inference.rerank(query=)`, and, on the `Admin` client,
`api_keys.create(name=)`, `invites.create(email=)`, `projects.create(name=)`,
`projects.describe_by_name(name=)`, `role_bindings.create(principal_id=)` and
`service_accounts.create(name=)`.

Convert at the call site, with `str(uuid_value)` or `str(numeric_id)`. Two
cases keep their documented behavior: an optional path parameter still
accepts `None`, so `pc.backups.list(index_name=None)` lists backups across
every index, and `describe_import()` and `cancel_import()` still accept an
`int` ID and convert it.

(v10-1-clients)=
## Creating a client

An API key that cannot be sent as an HTTP header, such as one with a trailing
newline read from a file, now raises `PineconeValueError` when you create the
client. The message never repeats the key. 10.0.0 accepted the key, and the
first request failed inside the HTTP library with a message that could
include it:

```python
api_key = "your-api-key\n"  # as read from a file
Pinecone(api_key=api_key)
# PineconeValueError: The API key (api_key or PINECONE_API_KEY) is not a valid HTTP header value: it is blank or whitespace-only, has leading or trailing whitespace (such as a newline read from a file), or contains a control or non-ASCII character. Strip the key, for example api_key=key.strip().

pc = Pinecone(api_key=api_key.strip())
```

`GrpcIndex` still accepts leading and trailing spaces and tabs, which gRPC can
send, and refuses the rest.

An index host given with `http://` now emits a `RuntimeWarning`, because the
API key and every request travel unencrypted. Loopback and private IP
addresses, `localhost`, and `*.localhost` are exempt, so a local emulator
stays quiet. Pass the host with `https://`, or with no scheme:

```python
pc.index(host="http://movies-abc123.svc.aped-4627-b74a.pinecone.io")
# RuntimeWarning: The index host http://movies-abc123.svc.aped-4627-b74a.pinecone.io uses http, so the API key and every request and response body travel unencrypted.
```

`additional_headers` now replaces an SDK header whatever its capitalization.
In 10.0.0 only the exact spelling, such as `"X-Pinecone-Api-Version"`,
replaced it; any other spelling was sent alongside the SDK's own header, so
the server received two values. This client now sends one
`X-Pinecone-Api-Version`, `2025-10`:

```python
pinned = Pinecone(
    api_key="your-api-key", additional_headers={"x-pinecone-api-version": "2025-10"}
)
```

`Pinecone()` and `AsyncPinecone()` called with a keyword they do not take
raise `PineconeTypeError` instead of a plain `TypeError`. It is a subclass of
`TypeError`, so existing handlers still catch it.

(v10-1-bulk-upsert)=
## Bulk document upserts

When the server rejects a `documents.batch_upsert` batch as too large
(HTTP 413), 10.1.0 halves the batch and resends each half, down to single
documents. 10.0.0 recorded the whole batch as failed. Code that retried
`result.failed_items` with a smaller `batch_size` can drop that step; what
remains in `failed_items` after a 413 is a single document over the server's
size cap, which no resend fixes. One batch can now land in part, so several
`result.errors` rows can share a `batch_index`.

A 413 from any call now raises `PayloadTooLargeError` rather than a plain
`ApiError`. It is a subclass of `ApiError`, so `except ApiError` still
catches it.

On an index with an `embed` or `sparse_embed` field, the server takes at most
96 documents per upsert or update request. The default `batch_size` of 50
fits; keep any value you pass at 96 or below:

```python
index.documents.batch_upsert(
    namespace="movies-en",
    documents=[{"_id": f"movie-{i}", "title": f"Movie {i}"} for i in range(500)],
    batch_size=96,
)
```

(v10-1-summary)=
## Summary

| Change | Before (10.0.0) | Now (10.1.0) |
| --- | --- | --- |
| Polling `documents.fetch` until `is_reconciled(target)` | exits at once (`lsn_committed` was `None`) | never exits; poll `documents.search` with a deadline |
| Unknown key in a dict `score_by` clause | dropped | `PineconeValueError` |
| Bare `str` for `ids`, `include_fields` or `remove_fields` | split into characters | `PineconeTypeError` |
| Bare `str` for a clause's `fields` | sent as one string | `PineconeTypeError` |
| `timeout` / `total_timeout` of zero, negative or NaN | passed through | `PineconeValueError` |
| Negative wait `timeout` other than `-1` on create, create-for-model, delete | request sent, then `PineconeTimeoutError` | `PineconeValueError`, nothing sent |
| Non-`str` or `None` name or ID | coerced, `AttributeError`, `TypeError` or `PineconeValueError` | `PineconeTypeError` |
| API key with whitespace or control characters | first request fails | `PineconeValueError` at construction |
| Public index host with `http://` | silent | `RuntimeWarning` |
| `additional_headers` in a different capitalization | sent alongside the SDK's header | replaces it |
| 413 in `documents.batch_upsert` | whole batch failed | halved and resent |
| 413 from any call | `ApiError` | `PayloadTooLargeError` (an `ApiError`) |
