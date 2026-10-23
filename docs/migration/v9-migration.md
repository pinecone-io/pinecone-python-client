# Migrating from v8.x to v9.x

v9 is a rewrite of the SDK. This guide shows the v9 equivalent of each v8 pattern that
changed. Upgrading further? Continue with [Migrating to v10](v10-migration.md).

## Key changes

### 1. Namespace pattern for control-plane operations

In v8, control-plane methods lived directly on the `Pinecone` client:

```python
# v8
pc.create_index(
    name="my-index", dimension=1536, metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
indexes = pc.list_indexes()
pc.delete_index("my-index")
```

In v9, they are grouped under namespace properties:

```python
# v9
pc.indexes.create(
    name="my-index", dimension=1536, metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
indexes = pc.indexes.list()
pc.indexes.delete("my-index")
```

The same pattern applies to collections, backups, and inference:

```python
pc.collections.create(name="catalog-snapshot", source="my-index")
pc.backups.list()
pc.inference.embed(model="multilingual-e5-large", inputs=["hello"])
```

**You are not forced to move.** `pc.create_index`, `pc.describe_index`,
`pc.list_indexes`, `pc.configure_index`, `pc.delete_index`,
`pc.create_collection`, `pc.list_collections` and `pc.delete_collection` still
exist on the client and forward to the namespace methods, so you can migrate
call sites one at a time. They are hidden from the reference docs.

The shims keep their positional signatures, so
`pc.create_index("my-index", ServerlessSpec(cloud="aws", region="us-east-1"), 1536)`
and `pc.configure_index("my-index", 4)` still work. `pc.indexes.create` is
keyword-only: `pc.indexes.create("my-index")` raises `TypeError`.

`create` and `delete` both block until the index reaches its end state, polling
with no upper bound unless you pass `timeout`. Pass `timeout=-1` to return as
soon as the request is accepted.

### 2. Async client rename

`PineconeAsyncio` is renamed to `AsyncPinecone`. The old name is still importable as an
alias of the same class, so `isinstance` checks are unaffected.

```python
# v8
from pinecone import PineconeAsyncio
async with PineconeAsyncio(api_key="...") as pc:
    ...

# v9
from pinecone import AsyncPinecone
async with AsyncPinecone(api_key="...") as pc:
    ...
```

### 3. Response models

v9 returns `msgspec.Struct` instances. Attribute access (`idx.name`) and subscripting a
declared field (`idx["name"]`) work as before. A model is not a mapping, so `dict(idx)`
raises `KeyError`.

```python
import msgspec

idx = pc.indexes.describe("my-index")
print(idx.name)                     # works
print(idx["name"])                  # works for declared fields
print(idx.to_dict())                # nested plain dicts, JSON-ready
print(msgspec.structs.asdict(idx))  # shallow: nested models stay as objects
print(dict(idx))                    # raises KeyError
```

Use `to_dict()` for JSON and `msgspec.structs.asdict()` for the top-level fields with
nested models intact.

### 4. HTTP transport: httpx replaces urllib3

The SDK uses `httpx` instead of `urllib3`. Retry behavior is now configured
with `RetryConfig` passed at client construction:

```python
# v8: retry parameters were keyword args on the client
pc = Pinecone(api_key="...", retries=3)

# v9
from pinecone import Pinecone, RetryConfig
pc = Pinecone(
    api_key="...",
    retry_config=RetryConfig(max_retries=3, backoff_factor=1.5),
)
```

`retries` has no compatibility shim: `Pinecone(retries=3)` raises `PineconeTypeError`
(a `TypeError`). See
[Retries and Resilience](../guides/retries.md) for what `retry_config` reaches and what
it does not.

### 5. gRPC: Rust extension replaces grpcio

`GrpcIndex` is backed by a compiled Rust extension that ships in the base `pinecone`
package. Drop the `grpc` extra and any `grpcio` dependency. The data-plane interface
(`upsert`, `query`, `fetch`, `delete`) is unchanged.

```python
# v9: no grpcio dependency required
index = pc.index("my-index", grpc=True)
index.upsert(vectors=[("product-42", [0.012, -0.087, 0.153])])
```

`from pinecone.grpc import PineconeGRPC` still works, and `PineconeGRPC(...).Index(name=...)`
still returns a `GrpcIndex`. [Using the gRPC Client](../guides/grpc.md) covers
the transport, including what `GrpcIndex` does not carry.

### 6. Import paths

Most public classes are still importable directly from `pinecone`:

```python
from pinecone import Pinecone, AsyncPinecone, Index, GrpcIndex
from pinecone import ServerlessSpec, PodSpec
from pinecone import ConflictError, NotFoundError, ForbiddenError
```

The v8 module paths `pinecone.pinecone`, `pinecone.pinecone_asyncio`,
`pinecone.exceptions`, `pinecone.control`, `pinecone.data`, `pinecone.config`,
`pinecone.db_control` and `pinecone.db_data` survive as shims that re-export from the
new locations, so imports such as `from pinecone.db_data import Index` keep working.
They are hidden from the reference docs; new code should import from `pinecone`.

The generated OpenAPI tree is gone: `from pinecone.core.client.api...` raises
`ModuleNotFoundError`. Use the top-level package instead.

### 7. Python version requirement

Python 3.9 support is dropped. The minimum supported version is Python 3.10.

### 8. Removal of the `pinecone_plugins.assistant` import path

In v8, the assistant API shipped as a separate plugin package
(`pinecone-plugin-assistant`), and code imported model classes from
`pinecone_plugins.assistant.*`:

```python
# v8
from pinecone_plugins.assistant.models import (
    AssistantModel, ContextOptions, Message, FileModel,
)
from pinecone_plugins.assistant.models.chat import ChatResponse
```

In v9 the assistant API is built into `pinecone` and the `pinecone_plugins` import
tree is gone. Every class is importable from `pinecone.models.assistant`, under its
canonical name or a legacy alias:

| v8 import path | v9 import path |
|---|---|
| `from pinecone_plugins.assistant.models import AssistantModel` | `from pinecone.models.assistant import AssistantModel` |
| `from pinecone_plugins.assistant.models import ContextOptions` | `from pinecone.models.assistant import ContextOptions` |
| `from pinecone_plugins.assistant.models import Message` | `from pinecone.models.assistant import Message` |
| `from pinecone_plugins.assistant.models import FileModel` | `from pinecone.models.assistant import FileModel` *(deprecated alias for `AssistantFileModel`)* |
| `from pinecone_plugins.assistant.models.chat import ChatResponse` | `from pinecone.models.assistant import ChatResponse` |
| `from pinecone_plugins.assistant.models.chat import Citation, Reference, Highlight` | `from pinecone.models.assistant import Citation, Reference, Highlight` *(deprecated aliases for `ChatCitation` etc.)* |
| `from pinecone_plugins.assistant.models.chat import StreamChatResponseMessageStart, StreamChatResponseContentDelta, StreamChatResponseCitation, StreamChatResponseMessageEnd` | `from pinecone.models.assistant import StreamChatResponseMessageStart, StreamChatResponseContentDelta, StreamChatResponseCitation, StreamChatResponseMessageEnd` *(deprecated aliases for `StreamMessageStart` etc.)* |
| `from pinecone_plugins.assistant.models.chat_completion import ChatCompletionResponse, StreamingChatCompletionChunk` | `from pinecone.models.assistant import ChatCompletionResponse, StreamingChatCompletionChunk` |
| `from pinecone_plugins.assistant.models.context_responses import ContextResponse, TextSnippet, MultimodalSnippet` | `from pinecone.models.assistant import ContextResponse, TextSnippet, MultimodalSnippet` |
| `from pinecone_plugins.assistant.models.context_responses import TextBlock, ImageBlock, Image` | `from pinecone.models.assistant import TextBlock, ImageBlock, Image` *(deprecated aliases for `ContextTextBlock`, `ContextImageBlock`, `ContextImageData`)* |
| `from pinecone_plugins.assistant.models.context_responses import PdfReference, TextReference, JsonReference, MarkdownReference, DocxReference` | `from pinecone.models.assistant import PdfReference, TextReference, JsonReference, MarkdownReference, DocxReference` *(all five alias the consolidated `FileReference`)* |
| `from pinecone_plugins.assistant.models.evaluation_responses import AlignmentResponse, Metrics, EvaluatedFact` | `from pinecone.models.assistant import AlignmentResponse, Metrics, EvaluatedFact` *(deprecated aliases for `AlignmentResult`, `AlignmentScores`, `EntailmentResult`)* |
| `from pinecone_plugins.assistant.models.list_files_response import ListFilesResponse` | `from pinecone.models.assistant import ListFilesResponse` |
| `from pinecone_plugins.assistant.models.list_assistants_response import ListAssistantsResponse` | `from pinecone.models.assistant import ListAssistantsResponse` |
| `from pinecone_plugins.assistant.models.shared import Message, Usage, TokenCounts` | `from pinecone.models.assistant import Message, Usage, TokenCounts` *(`Usage` and `TokenCounts` are deprecated aliases for `ChatUsage`)* |
| `from pinecone_plugins.assistant.assistant.assistant import Assistant` | No replacement; see below. |

**Method calls do not change:**

```python
# Both v8 and v9
pc = Pinecone(api_key="...")
pc.assistant.create_assistant("my-assistant")        # works in v9
pc.assistant.list_assistants()                       # works in v9
assistant = pc.assistant.describe_assistant("my-assistant")
assistant.upload_file(file_path="report.pdf")        # works in v9
assistant.chat(messages=[...])                       # works in v9
```

`pc.assistant` and `pc.assistants` are interchangeable, and legacy method names such
as `create_assistant`, `delete_assistant` and `list_assistants_paginated` work alongside
the canonical `pc.assistants.create`, `.delete` and `.list_page`.

**The plugin class is removed.** Code that constructed the plugin directly
(`Assistant(config, client_builder)`) has no v9 equivalent; use `pc.assistants` instead.


### 9. Partial-success contract for batched upsert

`index.upsert(...)` keeps v8's `batch_size` and `show_progress` and adds
`max_concurrency`, so v8 call sites still run. What happens on failure changed.

In v8, `idx.upsert(vectors=[...], batch_size=N)` sent batches one at a time and raised
on the first failed batch; later batches never ran.

In v9, batches run concurrently, up to `max_concurrency` at a time, and each batch is
retried on transient errors. A batch that still fails is recorded on the returned
{class}`~pinecone.models.UpsertResponse` instead of raised. The response reports
`has_errors`, `errors`, per-batch and per-item counts, and `failed_items`, the items from
failed batches ready to resubmit.

Code that relies on `try/except` around a batched upsert misses those failures:

```python
# v8: relied on the exception to roll back
try:
    idx.upsert(vectors=batch, batch_size=100)
except Exception:
    rollback()

# v9: inspect response.has_errors instead
response = idx.upsert(vectors=batch, batch_size=100)
if response.has_errors:
    # Optionally retry only the failures:
    idx.upsert(vectors=response.failed_items, batch_size=100)
    # …or roll back if any failure is unacceptable
```

Without `batch_size` (the default), upsert sends one request and raises on failure,
as in v8. `AsyncIndex.upsert(...)` and `GrpcIndex.upsert(...)` follow the same contract.

---

## v8 → v9 migration table

| Operation | v8 | v9 |
|---|---|---|
| Create index | `pc.create_index(name=..., dimension=..., spec=...)` | `pc.indexes.create(name=..., dimension=..., spec=...)` |
| List indexes | `pc.list_indexes()` | `pc.indexes.list()` |
| Describe index | `pc.describe_index("name")` | `pc.indexes.describe("name")` |
| Delete index | `pc.delete_index("name")` | `pc.indexes.delete("name")` |
| Configure index | `pc.configure_index("name", ...)` | `pc.indexes.configure("name", ...)` |
| Check index exists | `pc.describe_index("name")` + try/except | `pc.indexes.exists("name")` |
| Get data-plane index | `pc.Index("name")` | `pc.index("name")` *(`pc.Index()` still works)* |
| Get gRPC index | `PineconeGRPC(...).Index("name")` | `pc.index("name", grpc=True)` *(`PineconeGRPC` still works)* |
| Create collection | `pc.create_collection(name=..., source=...)` | `pc.collections.create(name=..., source=...)` |
| List collections | `pc.list_collections()` | `pc.collections.list()` |
| Delete collection | `pc.delete_collection("name")` | `pc.collections.delete("name")` |
| Upsert vectors | `index.upsert(vectors=[...])` | `index.upsert(vectors=[...])` *(batched form reports failures on the response; see §9)* |
| Query vectors | `index.query(vector=[...], top_k=10)` | `index.query(vector=[...], top_k=10)` *(unchanged)* |
| Fetch vectors | `index.fetch(ids=[...])` | `index.fetch(ids=[...])` *(unchanged)* |
| Delete vectors | `index.delete(ids=[...])` | `index.delete(ids=[...])` *(unchanged)* |
| Async client | `PineconeAsyncio(api_key=...)` | `AsyncPinecone(api_key=...)` |
| Retry config | `Pinecone(retries=3)` | `Pinecone(retry_config=RetryConfig(max_retries=3))` |
| Convert response to dict | `dict(idx)` or `idx.to_dict()` | `idx.to_dict()` *(`dict(idx)` raises; see §3)* |
| Embed text | `pc.inference.embed(...)` | `pc.inference.embed(...)` *(unchanged)* |

---

## Legacy aliases

These aliases remain importable from `pinecone` but are deprecated:

| Deprecated name | Canonical name |
|---|---|
| `PineconeAsyncio` | `AsyncPinecone` |
| `ValidationError` | `PineconeValueError` *(the one alias that emits a `DeprecationWarning` on access)* |
| `ForbiddenException` | `ForbiddenError` |
| `NotFoundException` | `NotFoundError` |

Each is the same class as its canonical name, so `isinstance` checks and `except`
clauses behave identically. Switch to the canonical names when convenient.
