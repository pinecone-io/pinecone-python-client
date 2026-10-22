# Working with the assistant

A Pinecone assistant is a managed question-answering service grounded in documents you
upload to it. Create the assistant, upload files, then ask questions and get answers with
citations back to the files that supported each claim. Pinecone does both the retrieval
and the generation, which is what separates an assistant from an index: an index hands
you records to feed to a model of your own.

Everything is reached through `pc.assistants`. Methods that work on one assistant's files
and conversations need that assistant's host, so the first such call for a given name
runs {meth}`~pinecone.client.assistants.Assistants.describe` and caches the host for the
life of the client. A name that does not exist therefore raises
{exc}`~pinecone.errors.exceptions.NotFoundError` from that lookup.

`AsyncPinecone` exposes the same methods with `await`; see
[Sync vs async](../guides/sync-vs-async.md). For which exceptions these methods raise and
which are worth retrying, see [Error handling](../guides/error-handling.md). Every
parameter is documented on {class}`~pinecone.client.assistants.Assistants`.

## Create an assistant

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

assistant = pc.assistants.create(
    name="research-assistant",
    instructions="Answer questions based on the uploaded documents.",
    region="us",
)
print(assistant.status)  # "Ready"
```

{meth}`~pinecone.client.assistants.Assistants.create` polls until the assistant is
`"Ready"`, so you can upload files as soon as it returns. Pass `timeout=-1` to return
immediately and check on it later with `describe`. If the assistant fails to initialize,
polling stops with {exc}`~pinecone.errors.exceptions.PineconeError` instead of waiting
forever. The region cannot be changed later; an assistant in the wrong one has to be
recreated.

## List, describe, and update assistants

```python
for asst in pc.assistants.list():
    print(asst.name, asst.status)

asst = pc.assistants.describe(name="research-assistant")
print(asst.region, asst.instructions, asst.host)

pc.assistants.update(
    name="research-assistant",
    instructions="Always cite the source document.",
)
```

`update` changes only the fields you pass. A `metadata` you pass replaces the whole
dictionary rather than merging into it. To clear a field, pass its empty value:
`instructions=""` or `metadata={}`.

## Upload files

```python
file = pc.assistants.upload_file(
    assistant_name="research-assistant",
    file_path="/data/q3-revenue-review.pdf",
    metadata={"department": "finance", "quarter": "2024-Q3"},
)
print(file.id, file.status)  # "Available"
```

{meth}`~pinecone.client.assistants.Assistants.upload_file` waits for processing to
finish, so the file is ready to answer questions when the call returns. The server picks
how to process a file from its extension; the method reference lists the supported types.
The `metadata` you attach is what the `filter` argument of `chat`, `context` and
`list_files` matches against.

To upload bytes you already hold, pass `file_stream` with a `file_name` that carries the
extension:

```python
import io

with open("/data/q3-revenue-review.pdf", "rb") as f:
    pdf_bytes = f.read()

file = pc.assistants.upload_file(
    assistant_name="research-assistant",
    file_stream=io.BytesIO(pdf_bytes),
    file_name="q3-revenue-review.pdf",
)
```

Passing `file_id` replaces the file stored under that id instead of adding a new one.

To see what an assistant holds, `list_files` iterates over its files and `describe_file`
fetches one by id:

```python
for file in pc.assistants.list_files(
    assistant_name="research-assistant",
    filter={"department": "finance"},
):
    print(file.id, file.name, file.status)

file = pc.assistants.describe_file(
    assistant_name="research-assistant",
    file_id="file-abc123",
    include_url=True,
)
print(file.signed_url)  # temporary download link
```

A file whose processing failed drops out of `list_files` after a while, but
`describe_file` still returns it by id.

## Track file operations

`upload_file` and `delete_file` poll the server-side operation for you. Called with
`timeout=-1`, they return as soon as the server accepts the request, and you follow the
work with the operations methods:

```python
for op in pc.assistants.list_operations(
    assistant_name="research-assistant",
    operation_type="upload_file",
    status="Processing",
):
    print(op.operation_id, op.file_id, op.percent_complete)

operation = pc.assistants.describe_operation(
    assistant_name="research-assistant",
    operation_id="op-1234-abcd-5678",
)
if operation.status == "Failed":
    print(operation.error)
```

`status` is `"Processing"`, `"Completed"` or `"Failed"`. Check `status` rather than
`error`: a retried operation can still carry an earlier attempt's error message. The
server keeps finished operations for a limited time. For page-at-a-time listing, see
{meth}`~pinecone.client.assistants.Assistants.list_operations_page` and
[Pagination](../guides/pagination.md).

## Ask a question

```python
response = pc.assistants.chat(
    assistant_name="research-assistant",
    messages=[{"role": "user", "content": "How did Q3 revenue change?"}],
)
print(response.message.content)
print(response.model)          # the model that answered
print(response.finish_reason)  # "stop" when the model finished on its own
```

Messages can be dicts or {class}`~pinecone.models.assistant.message.Message` objects.
`role` is `"user"` or `"assistant"`, and defaults to `"user"` when a dict leaves it out.

The assistant answers only from files you uploaded. Until at least one file is
`"Available"`, `chat` raises {exc}`~pinecone.errors.exceptions.ApiError` instead of
answering from general knowledge.

`model` picks the large language model that writes the answer; the
{meth}`~pinecone.client.assistants.Assistants.chat` reference lists the documented
choices and the default. A name outside that list may be served by a successor model
rather than rejected, so read `response.model` to see which model answered.

### Render the citations

Each citation carries the character `position` in the answer it supports, so you can
insert footnote markers as you render the text, and the references behind it:

```python
for citation in response.citations:
    for reference in citation.references:
        print(citation.position, reference.file.name, reference.pages)
```

`reference.file` is an {class}`~pinecone.models.assistant.file_model.AssistantFileModel`
with the `name`, `id` and `metadata` of the source file. `reference.pages` lists page
numbers for paginated sources such as PDFs, and is `None` otherwise. Pass
`include_highlights=True` to also get `reference.highlight.content`, the passage the
citation drew on.

An empty `response.citations` means the answer drew on no file. If
`response.context_snippet_count` is `0`, retrieval found nothing relevant to the question.

### `chat`, `chat_completions`, or `context`?

All three take the same conversation. Pick by what you need back.

{meth}`~pinecone.client.assistants.Assistants.chat`
: The default. Structured `citations`, plus `include_highlights`, `context_options` and
  `json_response`.

{meth}`~pinecone.client.assistants.Assistants.chat_completions`
: The answer in OpenAI's chat completion shape, at `response.choices[0].message.content`,
  for code already written against that shape. Citations arrive woven into the text, and
  the three `chat`-only options above are not available.

{meth}`~pinecone.client.assistants.Assistants.context`
: Retrieval without generation: the snippets the assistant would answer from, for a
  prompt you send to your own model. Takes exactly one of `query` or `messages`.

```python
response = pc.assistants.chat_completions(
    assistant_name="research-assistant",
    messages=[{"content": "How did Q3 revenue change?"}],
)
print(response.choices[0].message.content)

context = pc.assistants.context(
    assistant_name="research-assistant",
    query="Q3 revenue",
    top_k=5,
)
for snippet in context.snippets:
    print(snippet.score, snippet.content, snippet.reference.file.name)
```

With `multimodal=True`, `context` can also return a
{class}`~pinecone.models.assistant.context.MultimodalSnippet`, whose `content` is a list
of blocks rather than a string, so check the snippet type before reading `content`.

The assistant models returned by `create`, `describe` and `list` also carry legacy
methods such as `assistant.chat(...)`. They are deprecated, work only on the sync client,
and forward to the `pc.assistants` methods; call those instead.

## Stream a chat response

Pass `stream=True` to get a {class}`~pinecone.models.assistant.streaming.ChatStream`.
For text only, `text()` yields fragments as they arrive:

```python
stream = pc.assistants.chat(
    assistant_name="research-assistant",
    messages=[{"content": "Summarize the revenue review."}],
    stream=True,
)
for fragment in stream.text():
    print(fragment, end="", flush=True)
```

`collect()` returns the fragments joined into one string. A stream can be consumed only
once, so use one of iteration, `text()` or `collect()`.

`text()` drops citations and token usage. To get them, iterate the stream and branch on
the chunk type:

```python
from pinecone import (
    StreamCitationChunk,
    StreamContentChunk,
    StreamMessageEnd,
    StreamMessageStart,
)

stream = pc.assistants.chat(
    assistant_name="research-assistant",
    messages=[{"content": "Summarize the revenue review."}],
    stream=True,
)

sources: list[tuple[int, str]] = []
for chunk in stream:
    if isinstance(chunk, StreamMessageStart):
        if chunk.context_snippet_count == 0:
            print("nothing relevant was retrieved")
    elif isinstance(chunk, StreamContentChunk):
        print(chunk.delta.content, end="", flush=True)
    elif isinstance(chunk, StreamCitationChunk):
        for reference in chunk.citation.references:
            sources.append((chunk.citation.position, reference.file.name))
    elif isinstance(chunk, StreamMessageEnd):
        print(f"\nfinish_reason={chunk.finish_reason}")
```

{class}`~pinecone.models.assistant.streaming.StreamMessageStart`
: Arrives first. A `context_snippet_count` of `0` means retrieval found nothing.

{class}`~pinecone.models.assistant.streaming.StreamContentChunk`
: The only chunk that carries answer text, at `chunk.delta.content`. Concatenate the
  fragments in arrival order.

{class}`~pinecone.models.assistant.streaming.StreamCitationChunk`
: Interleaved with the content, with the same `position` and `references` as a
  non-streaming citation.

{class}`~pinecone.models.assistant.streaming.StreamMessageEnd`
: Arrives last. `finish_reason` tells a complete answer from one cut short, and `usage`
  holds the token counts.

`stream=True` cannot be combined with `json_response=True`. On a streaming request,
`timeout` bounds the gap between chunks rather than the whole response, and when you do
not pass it the SDK uses a longer default so a model that pauses to reason is not taken
for a dead connection. A stream that times out raises
{exc}`~pinecone.errors.exceptions.PineconeTimeoutError` partway through iteration, after
earlier chunks have been yielded.

`chat_completions(..., stream=True)` returns a
{class}`~pinecone.models.assistant.streaming.ChatCompletionStream`, whose text is at
`chunk.choices[0].delta.content`. That value is `None` on chunks that carry no text.

## Evaluate an answer against a ground truth

{meth}`~pinecone.client.assistants.Assistants.evaluate_alignment` scores an answer
against one you know is correct. It takes no `assistant_name`, so you can score answers
from any source.

```python
result = pc.assistants.evaluate_alignment(
    question="What is the capital of Spain?",
    answer="Barcelona.",
    ground_truth_answer="Madrid.",
)
print(result.scores.correctness)   # precision: how much of the answer holds up
print(result.scores.completeness)  # recall: how much of the ground truth it covers
print(result.scores.alignment)     # harmonic mean of the two

for fact in result.facts:
    print(fact.entailment, fact.fact, fact.reasoning)
```

A low correctness or completeness drags `alignment` down, so read all three. Each entry
in `result.facts` is one fact judged `"entailed"`, `"contradicted"` or `"neutral"`, which
shows where an answer went wrong. `result.usage` counts the tokens the evaluation spent.

## Delete files and assistants

```python
pc.assistants.delete_file(
    assistant_name="research-assistant",
    file_id="file-abc123",
)

pc.assistants.delete(name="research-assistant")
```

Both calls wait until the deletion finishes; pass `timeout=-1` to return as soon as the
server accepts the request. Deleting a file id that no longer exists raises
{exc}`~pinecone.errors.exceptions.NotFoundError`. Deleting an assistant also deletes its
files.
