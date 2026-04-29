# Migrating from v8.x to v9.x

v9 is a ground-up rewrite focused on simplicity, performance, and type safety. This guide
covers the breaking changes and shows you the v9 equivalent for each v8 pattern.

## Key changes

### 1. Namespace pattern for control-plane operations

In v8, control-plane methods lived directly on the `Pinecone` client:

```python
# v8
pc.create_index(name="my-index", dimension=1536, metric="cosine", spec=...)
indexes = pc.list_indexes()
pc.delete_index("my-index")
```

In v9, they are grouped under namespace properties:

```python
# v9
pc.indexes.create(name="my-index", dimension=1536, metric="cosine", spec=...)
indexes = pc.indexes.list()
pc.indexes.delete("my-index")
```

The same pattern applies to collections, backups, and inference:

```python
pc.collections.create(...)
pc.backups.list()
pc.inference.embed(...)
```

### 2. Async client rename

`PineconeAsyncio` is renamed to `AsyncPinecone`. The old name still works but is
deprecated and will be removed in a future release.

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

v8 returned a mix of plain dicts and Pydantic models. v9 returns `msgspec.Struct` instances.
Field access is identical—`idx.name`, `idx.dimension`—but the objects are immutable.
`dict()` no longer works; use `msgspec.structs.asdict(idx)` if you need a dict.

```python
# v9 — field access is unchanged
idx = pc.indexes.describe("my-index")
print(idx.name)        # works
print(idx.dimension)   # works
print(dict(idx))       # TypeError — structs are not dict-convertible
```

### 4. HTTP transport: httpx replaces urllib3

The SDK uses `httpx` with HTTP/2 instead of `urllib3`. Retry behavior is now configured
with `RetryConfig` passed at client construction:

```python
# v8 — retry parameters were keyword args on the client
pc = Pinecone(api_key="...", retries=3)

# v9
from pinecone import Pinecone, RetryConfig
pc = Pinecone(
    api_key="...",
    retry_config=RetryConfig(max_retries=3, backoff_factor=1.5),
)
```

### 5. gRPC: Rust extension replaces grpcio

`GrpcIndex` is now backed by a compiled Rust extension instead of the Python `grpcio`
package. You do not need to install `grpcio` or `grpcio-tools`. The interface—`upsert`,
`query`, `fetch`, `delete`—is unchanged.

```python
# v9 — interface is the same; no grpcio dependency required
index = pc.index("my-index", grpc=True)
index.upsert(vectors=[...])
```

### 6. Import paths

Most public classes are still importable directly from `pinecone`:

```python
from pinecone import Pinecone, AsyncPinecone, Index, GrpcIndex
from pinecone import ServerlessSpec, PodSpec
from pinecone import ConflictError, NotFoundError, ForbiddenError
```

Deep imports (`from pinecone.core.client.api...`) are no longer supported. Use the
top-level package instead.

### 7. Python version requirement

Python 3.9 support is dropped. The minimum supported version is Python 3.10.

### 8. Removal of the `pinecone_plugins.assistant` import path

In v8, the assistant SDK shipped as a separate plugin package
(`pinecone-plugin-assistant`) installed alongside `pinecone`. Code
imported model classes from `pinecone_plugins.assistant.*`:

```python
# v8
from pinecone_plugins.assistant.models import (
    AssistantModel, ContextOptions, Message, FileModel,
)
from pinecone_plugins.assistant.models.chat import ChatResponse
```

In v9 the assistant API is built into the main `pinecone` package
and the `pinecone_plugins` import tree has been removed. **All
classes are now reachable from `pinecone.models.assistant`** under
either the canonical name or a legacy alias.

Replace each legacy import with the canonical path:

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
| `from pinecone_plugins.assistant.assistant.assistant import Assistant` | No replacement — see note below. |

**What does *not* change.** Method-call backcompat is preserved:

```python
# Both v8 and v9
pc = Pinecone(api_key="...")
pc.assistant.create_assistant("my-assistant")        # works in v9
pc.assistant.list_assistants()                       # works in v9
assistant = pc.assistant.describe_assistant("my-assistant")
assistant.upload_file(file_path="report.pdf")        # works in v9
assistant.chat(messages=[...])                       # works in v9
```

The `pc.assistant` namespace is preserved and singular/plural
forms (`pc.assistant` and `pc.assistants`) are interchangeable.
Legacy method names like `create_assistant`, `delete_assistant`,
`list_assistants_paginated`, etc. continue to work alongside the
canonical `pc.assistants.create`, `.delete`, `.list_page`.

**The legacy plugin class is removed.** Code that manually
instantiated the plugin (`Assistant(config, client_builder)` from
`pinecone_plugins.assistant.assistant.assistant`) has no v9
equivalent — the plugin discovery system was retired and
`pc.assistant` is now a property on the `Pinecone` client. Such
code must be rewritten to use `pc.assistants` directly.

**Environment variables** `PINECONE_PLUGIN_ASSISTANT_CONTROL_HOST`
and `PINECONE_PLUGIN_ASSISTANT_DATA_HOST` are no longer consulted.
To target a non-prod control plane, pass `host=` to the `Pinecone`
constructor or set `PINECONE_HOST`. The data-plane host is
discovered automatically from the `host` field of the
`describe_assistant` response.

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
| Get data-plane index | `pc.Index("name")` | `pc.Index("name")` *(unchanged)* |
| Get gRPC index | `Pinecone(...).GrpcIndex("name")` | `pc.index("name", grpc=True)` |
| Create collection | `pc.create_collection(name=..., source=...)` | `pc.collections.create(name=..., source=...)` |
| List collections | `pc.list_collections()` | `pc.collections.list()` |
| Delete collection | `pc.delete_collection("name")` | `pc.collections.delete("name")` |
| Upsert vectors | `index.upsert(vectors=[...])` | `index.upsert(vectors=[...])` *(unchanged)* |
| Query vectors | `index.query(vector=[...], top_k=10)` | `index.query(vector=[...], top_k=10)` *(unchanged)* |
| Fetch vectors | `index.fetch(ids=[...])` | `index.fetch(ids=[...])` *(unchanged)* |
| Delete vectors | `index.delete(ids=[...])` | `index.delete(ids=[...])` *(unchanged)* |
| Async client | `PineconeAsyncio(api_key=...)` | `AsyncPinecone(api_key=...)` |
| Retry config | `Pinecone(retries=3)` | `Pinecone(retry_config=RetryConfig(max_retries=3))` |
| Convert response to dict | `dict(idx)` or `idx.to_dict()` | `msgspec.structs.asdict(idx)` |
| Embed text | `pc.inference.embed(...)` | `pc.inference.embed(...)` *(unchanged)* |

---

## Legacy aliases

The following aliases remain importable from `pinecone` but are deprecated:

| Deprecated name | Canonical name |
|---|---|
| `PineconeAsyncio` | `AsyncPinecone` |
| `ForbiddenException` | `ForbiddenError` *(`ForbiddenException` still works as a deprecated alias)* |
| `NotFoundException` | `NotFoundError` *(`NotFoundException` still works as a deprecated alias)* |
| `pinecone_plugins.assistant.*` (any submodule) | `pinecone.models.assistant`, `pinecone.client.assistants.Assistants` (via `pc.assistant` / `pc.assistants`) |

These aliases will be removed in a future major release. Update your code to use the
canonical names.
