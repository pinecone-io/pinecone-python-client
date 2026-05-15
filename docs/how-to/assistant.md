# Working with the Assistant

The assistant client lets you create and manage AI assistants that can answer questions
over your uploaded documents.

## Create an assistant

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

assistant = pc.assistant.create(
    name="my-assistant",
    instructions="Answer questions based on the uploaded documents.",
)
print(assistant.name)    # "my-assistant"
print(assistant.status)  # "Ready"
```

By default, ``create()`` polls until the assistant reaches ``"Ready"`` status before
returning. To return immediately without waiting (for example to kick off creation
asynchronously), pass ``timeout=-1`` — the returned assistant will be in
``"Initializing"`` status and you can check readiness later via ``describe()``.

## List and describe assistants

``list`` returns all assistants in the project:

```python
for asst in pc.assistant.list():
    print(asst.name, asst.status)
```

``describe`` returns details for a single assistant:

```python
asst = pc.assistant.describe(name="my-assistant")
print(asst.name)         # "my-assistant"
print(asst.status)       # "Ready"
print(asst.instructions) # the instruction string
```

## Upload a file

Pass a local file path to upload context documents for the assistant to read:

```python
file = pc.assistant.upload_file(
    assistant_name="my-assistant",
    file_path="data.pdf",
)
print(file.id)     # file ID used for later operations
print(file.name)   # "data.pdf"
print(file.status) # "Processing" → "Available"
```

## Chat

Send a conversation and receive a response:

```python
response = pc.assistant.chat(
    assistant_name="my-assistant",
    messages=[{"role": "user", "content": "What is the main topic of the document?"}],
)
print(response.message.content)
```

## Streaming chat

Pass ``stream=True`` to receive tokens incrementally as text fragments.  Use
``stream.text()`` — the idiomatic text-only accessor — to iterate over plain
strings.  Iterating ``stream`` directly instead yields typed chunk objects
(``StreamMessageStart``, ``StreamContentChunk``, ``StreamCitationChunk``,
``StreamMessageEnd``), which is useful when you need full metadata but would
print their ``repr`` rather than the assistant's text.

```python
stream = pc.assistant.chat(
    assistant_name="my-assistant",
    messages=[{"role": "user", "content": "Summarize the document."}],
    stream=True,
)
for text in stream.text():
    print(text, end="", flush=True)
```

## Delete a file

Remove an uploaded file from an assistant:

```python
pc.assistant.delete_file(
    assistant_name="my-assistant",
    file_id="file-id-here",
)
```

Raises {exc}`~pinecone.errors.exceptions.NotFoundError` if the file does not exist.

## Delete an assistant

```python
pc.assistant.delete(name="my-assistant")
```

Raises {exc}`~pinecone.errors.exceptions.NotFoundError` if the assistant does not exist.
