"""Unit tests for streaming chunk models — model field presence and wrapper classes.

Covers IT-0022: verifies that StreamContentChunk, StreamCitationChunk, and
StreamMessageEnd accept an optional ``model`` field when present in the wire JSON,
and default to ``None`` when absent.

Covers IT-0023: verifies that ChatCompletionStreamChunk accepts the optional
``model``, ``object``, ``created``, and ``system_fingerprint`` fields from the
OpenAI-compatible streaming endpoint.

Also covers DX-0069: verifies ChatStream, AsyncChatStream, ChatCompletionStream,
and AsyncChatCompletionStream wrapper classes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import msgspec

from pinecone.models.assistant.streaming import (
    AsyncChatCompletionStream,
    AsyncChatStream,
    ChatCompletionStream,
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatCompletionStreamDelta,
    ChatStream,
    ChatStreamChunk,
    StreamCitationChunk,
    StreamContentChunk,
    StreamContentDelta,
    StreamMessageEnd,
    StreamMessageStart,
)

# ---------------------------------------------------------------------------
# StreamContentChunk — model field
# ---------------------------------------------------------------------------


def test_stream_content_chunk_accepts_model() -> None:
    """StreamContentChunk decodes a JSON fixture that includes a model field."""
    raw = b'{"type": "content_chunk", "id": "c1", "delta": {"content": "Hello"}, "model": "gpt-4o"}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamContentChunk)
    assert chunk.id == "c1"
    assert chunk.delta.content == "Hello"
    assert chunk.model == "gpt-4o"


def test_stream_content_chunk_model_defaults_none() -> None:
    """StreamContentChunk.model defaults to None when absent from the wire response."""
    raw = b'{"type": "content_chunk", "id": "c1", "delta": {"content": "Hello"}}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamContentChunk)
    assert chunk.model is None


# ---------------------------------------------------------------------------
# StreamCitationChunk — model field
# ---------------------------------------------------------------------------


def test_stream_citation_chunk_accepts_model() -> None:
    """StreamCitationChunk decodes a JSON fixture that includes a model field."""
    raw = (
        b'{"type": "citation", "id": "cit1",'
        b' "citation": {"position": 5, "references": []},'
        b' "model": "gpt-4o"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamCitationChunk)
    assert chunk.id == "cit1"
    assert chunk.model == "gpt-4o"


def test_stream_citation_chunk_model_defaults_none() -> None:
    """StreamCitationChunk.model defaults to None when absent from the wire response."""
    raw = b'{"type": "citation", "id": "cit1", "citation": {"position": 5, "references": []}}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamCitationChunk)
    assert chunk.model is None


# ---------------------------------------------------------------------------
# StreamMessageEnd — model field
# ---------------------------------------------------------------------------


def test_stream_message_end_accepts_model() -> None:
    """StreamMessageEnd decodes a JSON fixture that includes a model field."""
    raw = (
        b'{"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},'
        b' "model": "gpt-4o"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageEnd)
    assert chunk.id == "end1"
    assert chunk.usage.total_tokens == 15
    assert chunk.model == "gpt-4o"


def test_stream_message_end_model_defaults_none() -> None:
    """StreamMessageEnd.model defaults to None when absent from the wire response."""
    raw = (
        b'{"type": "message_end", "id": "end1",'
        b' "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}'
    )
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageEnd)
    assert chunk.model is None


def test_stream_message_end_null_usage() -> None:
    """StreamMessageEnd decodes successfully when backend sends null for usage."""
    raw = b'{"type": "message_end", "id": "end1", "usage": null}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageEnd)
    assert chunk.id == "end1"
    assert chunk.usage is None


def test_stream_message_end_absent_usage() -> None:
    """StreamMessageEnd decodes successfully when usage field is absent entirely."""
    raw = b'{"type": "message_end", "id": "end1"}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageEnd)
    assert chunk.id == "end1"
    assert chunk.usage is None


# ---------------------------------------------------------------------------
# StreamMessageStart — model field (already present, sanity check)
# ---------------------------------------------------------------------------


def test_stream_message_start_has_model() -> None:
    """StreamMessageStart.model was already required; confirm it still works."""
    raw = b'{"type": "message_start", "model": "gpt-4o", "role": "assistant"}'
    chunk = msgspec.json.decode(raw, type=ChatStreamChunk)
    assert isinstance(chunk, StreamMessageStart)
    assert chunk.model == "gpt-4o"
    assert chunk.role == "assistant"


# ---------------------------------------------------------------------------
# ChatCompletionStreamChunk — extra fields (IT-0023)
# ---------------------------------------------------------------------------


def test_chat_completion_stream_chunk_accepts_extra_fields() -> None:
    """ChatCompletionStreamChunk decodes a realistic JSON fixture with all OpenAI fields."""
    raw = (
        b'{"id": "chatcmpl-abc123",'
        b' "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}],'
        b' "model": "gpt-4o",'
        b' "object": "chat.completion.chunk",'
        b' "created": 1714000000,'
        b' "system_fingerprint": "fp_abc123"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatCompletionStreamChunk)
    assert chunk.id == "chatcmpl-abc123"
    assert len(chunk.choices) == 1
    assert chunk.choices[0].index == 0
    assert chunk.choices[0].delta.content == "Hello"
    assert chunk.model == "gpt-4o"
    assert chunk.object == "chat.completion.chunk"
    assert chunk.created == 1714000000
    assert chunk.system_fingerprint == "fp_abc123"


def test_chat_completion_stream_chunk_extra_fields_default_none() -> None:
    """ChatCompletionStreamChunk optional fields default to None when absent."""
    raw = b'{"id": "chatcmpl-xyz", "choices": []}'
    chunk = msgspec.json.decode(raw, type=ChatCompletionStreamChunk)
    assert chunk.id == "chatcmpl-xyz"
    assert chunk.choices == []
    assert chunk.model is None
    assert chunk.object is None
    assert chunk.created is None
    assert chunk.system_fingerprint is None


def test_chat_completion_stream_chunk_model_only() -> None:
    """ChatCompletionStreamChunk with only model field set (no object/created/fingerprint)."""
    raw = (
        b'{"id": "chatcmpl-model-only",'
        b' "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}],'
        b' "model": "gpt-4o-mini"}'
    )
    chunk = msgspec.json.decode(raw, type=ChatCompletionStreamChunk)
    assert chunk.id == "chatcmpl-model-only"
    assert chunk.model == "gpt-4o-mini"
    assert chunk.object is None
    assert chunk.created is None
    assert chunk.system_fingerprint is None


# ---------------------------------------------------------------------------
# Helpers — build typed chunk fixtures
# ---------------------------------------------------------------------------


def _make_chat_chunks() -> list[ChatStreamChunk]:
    start = StreamMessageStart(model="gpt-4o", role="assistant")
    c1 = StreamContentChunk(id="c1", delta=StreamContentDelta(content="Hello"))
    c2 = StreamContentChunk(id="c2", delta=StreamContentDelta(content=" world"))
    end_raw = b'{"type": "message_end", "id": "end1", "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}'
    end = msgspec.json.decode(end_raw, type=ChatStreamChunk)
    return [start, c1, c2, end]


def _make_completion_chunks() -> list[ChatCompletionStreamChunk]:
    role_chunk = ChatCompletionStreamChunk(
        id="c0",
        choices=[
            ChatCompletionStreamChoice(
                index=0, delta=ChatCompletionStreamDelta(role="assistant", content=None)
            )
        ],
    )
    content_chunk1 = ChatCompletionStreamChunk(
        id="c1",
        choices=[
            ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content="Hello"))
        ],
    )
    content_chunk2 = ChatCompletionStreamChunk(
        id="c2",
        choices=[
            ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content=" world"))
        ],
    )
    finish_chunk = ChatCompletionStreamChunk(
        id="c3",
        choices=[
            ChatCompletionStreamChoice(
                index=0, delta=ChatCompletionStreamDelta(), finish_reason="stop"
            )
        ],
    )
    return [role_chunk, content_chunk1, content_chunk2, finish_chunk]


# ---------------------------------------------------------------------------
# ChatStream
# ---------------------------------------------------------------------------


def test_chat_stream_iterates_as_chat_stream_chunks() -> None:
    """Iterating a ChatStream yields the original ChatStreamChunk objects."""
    chunks = _make_chat_chunks()
    stream = ChatStream(iter(chunks))
    result = list(stream)
    assert len(result) == 4
    assert isinstance(result[0], StreamMessageStart)
    assert isinstance(result[1], StreamContentChunk)
    assert isinstance(result[2], StreamContentChunk)
    assert isinstance(result[3], StreamMessageEnd)


def test_chat_stream_text_yields_content_fragments() -> None:
    """ChatStream.text() yields only content strings from StreamContentChunk."""
    chunks = _make_chat_chunks()
    stream = ChatStream(iter(chunks))
    texts = list(stream.text())
    assert texts == ["Hello", " world"]


def test_chat_stream_text_skips_non_content_chunks() -> None:
    """ChatStream.text() skips start, citation, and end chunks."""
    start = StreamMessageStart(model="gpt-4o", role="assistant")
    stream = ChatStream(iter([start]))
    assert list(stream.text()) == []


def test_chat_stream_collect_returns_concatenated_content() -> None:
    """ChatStream.collect() returns all content fragments joined."""
    chunks = _make_chat_chunks()
    stream = ChatStream(iter(chunks))
    assert stream.collect() == "Hello world"


def test_chat_stream_collect_empty_stream() -> None:
    """ChatStream.collect() returns empty string when no content chunks present."""
    start = StreamMessageStart(model="gpt-4o", role="assistant")
    stream = ChatStream(iter([start]))
    assert stream.collect() == ""


def test_chat_stream_single_pass_iter_then_text() -> None:
    """After iterating all chunks, text() yields nothing (single-pass)."""
    chunks = _make_chat_chunks()
    stream = ChatStream(iter(chunks))
    _ = list(stream)  # exhaust via __iter__
    assert list(stream.text()) == []


def test_chat_stream_single_pass_collect_then_iter() -> None:
    """After collect(), iterating yields nothing (single-pass)."""
    chunks = _make_chat_chunks()
    stream = ChatStream(iter(chunks))
    _ = stream.collect()
    assert list(stream) == []


# ---------------------------------------------------------------------------
# AsyncChatStream
# ---------------------------------------------------------------------------


async def test_async_chat_stream_aiterates_as_chat_stream_chunks() -> None:
    """Async iterating an AsyncChatStream yields the original ChatStreamChunk objects."""
    chunks = _make_chat_chunks()

    async def _gen() -> AsyncIterator[ChatStreamChunk]:
        for c in chunks:
            yield c

    stream = AsyncChatStream(_gen())
    result = [c async for c in stream]
    assert len(result) == 4
    assert isinstance(result[0], StreamMessageStart)
    assert isinstance(result[1], StreamContentChunk)


async def test_async_chat_stream_text_yields_content_fragments() -> None:
    """AsyncChatStream.text() yields only content strings."""
    chunks = _make_chat_chunks()

    async def _gen() -> AsyncIterator[ChatStreamChunk]:
        for c in chunks:
            yield c

    stream = AsyncChatStream(_gen())
    texts = [t async for t in stream.text()]
    assert texts == ["Hello", " world"]


async def test_async_chat_stream_collect_returns_concatenated_content() -> None:
    """AsyncChatStream.collect() returns all content fragments joined."""
    chunks = _make_chat_chunks()

    async def _gen() -> AsyncIterator[ChatStreamChunk]:
        for c in chunks:
            yield c

    stream = AsyncChatStream(_gen())
    assert await stream.collect() == "Hello world"


async def test_async_chat_stream_single_pass() -> None:
    """After collect(), text() yields nothing (single-pass)."""
    chunks = _make_chat_chunks()

    async def _gen() -> AsyncIterator[ChatStreamChunk]:
        for c in chunks:
            yield c

    stream = AsyncChatStream(_gen())
    _ = await stream.collect()
    assert [t async for t in stream.text()] == []


# ---------------------------------------------------------------------------
# ChatCompletionStream
# ---------------------------------------------------------------------------


def test_chat_completion_stream_iterates_as_chunks() -> None:
    """Iterating ChatCompletionStream yields original ChatCompletionStreamChunk objects."""
    chunks = _make_completion_chunks()
    stream = ChatCompletionStream(iter(chunks))
    result = list(stream)
    assert len(result) == 4
    assert all(isinstance(c, ChatCompletionStreamChunk) for c in result)


def test_chat_completion_stream_text_skips_none_content() -> None:
    """ChatCompletionStream.text() skips chunks with None content."""
    chunks = _make_completion_chunks()
    stream = ChatCompletionStream(iter(chunks))
    texts = list(stream.text())
    assert texts == ["Hello", " world"]


def test_chat_completion_stream_text_skips_empty_string() -> None:
    """ChatCompletionStream.text() skips chunks with empty string content."""
    chunk = ChatCompletionStreamChunk(
        id="c1",
        choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content=""))],
    )
    stream = ChatCompletionStream(iter([chunk]))
    assert list(stream.text()) == []


def test_chat_completion_stream_text_skips_empty_choices() -> None:
    """ChatCompletionStream.text() handles chunks with no choices."""
    chunk = ChatCompletionStreamChunk(id="c1", choices=[])
    stream = ChatCompletionStream(iter([chunk]))
    assert list(stream.text()) == []


def test_chat_completion_stream_collect() -> None:
    """ChatCompletionStream.collect() returns concatenated non-None content."""
    chunks = _make_completion_chunks()
    stream = ChatCompletionStream(iter(chunks))
    assert stream.collect() == "Hello world"


def test_chat_completion_stream_collect_skips_empty_choices() -> None:
    """ChatCompletionStream.collect() skips chunks with no choices."""
    empty1 = ChatCompletionStreamChunk(id="c0", choices=[])
    content = ChatCompletionStreamChunk(
        id="c1",
        choices=[
            ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content="hello"))
        ],
    )
    empty2 = ChatCompletionStreamChunk(id="c2", choices=[])
    stream = ChatCompletionStream(iter([empty1, content, empty2]))
    assert stream.collect() == "hello"


# ---------------------------------------------------------------------------
# AsyncChatCompletionStream
# ---------------------------------------------------------------------------


async def test_async_chat_completion_stream_text_yields_content() -> None:
    """AsyncChatCompletionStream.text() yields non-None, non-empty content strings."""
    chunks = _make_completion_chunks()

    async def _gen() -> AsyncIterator[ChatCompletionStreamChunk]:
        for c in chunks:
            yield c

    stream = AsyncChatCompletionStream(_gen())
    texts = [t async for t in stream.text()]
    assert texts == ["Hello", " world"]


async def test_async_chat_completion_stream_collect() -> None:
    """AsyncChatCompletionStream.collect() returns concatenated content."""
    chunks = _make_completion_chunks()

    async def _gen() -> AsyncIterator[ChatCompletionStreamChunk]:
        for c in chunks:
            yield c

    stream = AsyncChatCompletionStream(_gen())
    assert await stream.collect() == "Hello world"


async def test_async_chat_completion_stream_skips_none_content() -> None:
    """AsyncChatCompletionStream.text() skips chunks where content is None."""
    chunks = _make_completion_chunks()

    async def _gen() -> AsyncIterator[ChatCompletionStreamChunk]:
        for c in chunks:
            yield c

    stream = AsyncChatCompletionStream(_gen())
    # role_chunk has content=None, finish_chunk has content=None
    texts = [t async for t in stream.text()]
    assert "None" not in texts
    assert all(t for t in texts)


async def test_async_chat_completion_stream_text_skips_empty_choices() -> None:
    """AsyncChatCompletionStream.text() skips chunks with no choices."""
    empty1 = ChatCompletionStreamChunk(id="c0", choices=[])
    content = ChatCompletionStreamChunk(
        id="c1",
        choices=[
            ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content="hello"))
        ],
    )
    empty2 = ChatCompletionStreamChunk(id="c2", choices=[])

    async def _gen() -> AsyncIterator[ChatCompletionStreamChunk]:
        for c in [empty1, content, empty2]:
            yield c

    stream = AsyncChatCompletionStream(_gen())
    texts = [t async for t in stream.text()]
    assert texts == ["hello"]


async def test_async_chat_completion_stream_collect_skips_empty_choices() -> None:
    """AsyncChatCompletionStream.collect() skips chunks with no choices."""
    empty1 = ChatCompletionStreamChunk(id="c0", choices=[])
    content = ChatCompletionStreamChunk(
        id="c1",
        choices=[
            ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamDelta(content="hello"))
        ],
    )
    empty2 = ChatCompletionStreamChunk(id="c2", choices=[])

    async def _gen() -> AsyncIterator[ChatCompletionStreamChunk]:
        for c in [empty1, content, empty2]:
            yield c

    stream = AsyncChatCompletionStream(_gen())
    assert await stream.collect() == "hello"
