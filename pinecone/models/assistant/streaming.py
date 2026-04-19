"""Streaming chunk models for the Assistant API.

Models for chat streaming (Pinecone-native format with type-based dispatch)
and chat completion streaming (OpenAI-compatible format).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TypeAlias

from msgspec import Struct

from pinecone.models.assistant.chat import ChatCitation, ChatUsage


class StreamMessageStart(Struct, kw_only=True, tag="message_start", tag_field="type"):
    """First chunk in a chat stream, containing the model and role.

    Attributes:
        model: The model used to generate the response.
        role: The role of the message author (e.g. ``"assistant"``).
    """

    model: str
    role: str


class StreamContentDelta(Struct, kw_only=True):
    """The delta payload within a content chunk.

    Attributes:
        content: The text fragment for this chunk.
    """

    content: str


class StreamContentChunk(Struct, kw_only=True, tag="content_chunk", tag_field="type"):
    """A content chunk containing a text fragment in a delta object.

    Attributes:
        id: Unique identifier for this chunk.
        delta: The delta object containing the text fragment.
        model: The model used to generate this response, or ``None`` if not provided.
    """

    id: str
    delta: StreamContentDelta
    model: str | None = None


class StreamCitationChunk(Struct, kw_only=True, tag="citation", tag_field="type"):
    """A citation chunk linking response text to source references.

    Attributes:
        id: Unique identifier for this chunk.
        citation: The citation data with position and references.
        model: The model used to generate this response, or ``None`` if not provided.
    """

    id: str
    citation: ChatCitation
    model: str | None = None


class StreamMessageEnd(Struct, kw_only=True, tag="message_end", tag_field="type"):
    """Final chunk in a chat stream, containing token usage statistics.

    Attributes:
        id: Unique identifier for this chunk.
        usage: Token usage statistics for the request.
        model: The model used to generate this response, or ``None`` if not provided.
    """

    id: str
    usage: ChatUsage
    model: str | None = None


ChatStreamChunk: TypeAlias = (
    StreamMessageStart | StreamContentChunk | StreamCitationChunk | StreamMessageEnd
)
"""Union of all Pinecone-native chat streaming chunk types."""


class ChatStream:
    """Wraps a Pinecone-native streaming response for convenient text access.

    Iterating over this object yields the full :class:`ChatStreamChunk` sequence,
    preserving the existing typed-chunk contract for callers that need it.
    :meth:`text` and :meth:`collect` give direct access to text content without
    manual type dispatch.

    The stream is single-pass: iterating, calling :meth:`text`, or calling
    :meth:`collect` all consume the same underlying iterator.

    Example::

        stream = pc.assistants.chat(assistant_name="my-assistant",
                                    messages=[{"content": "Hi"}], stream=True)
        for text in stream.text():
            print(text, end="", flush=True)
    """

    def __init__(self, stream: Iterator[ChatStreamChunk]) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[ChatStreamChunk]:
        return self._stream

    def text(self) -> Iterator[str]:
        """Yield text fragments, skipping start/citation/end chunks."""
        for chunk in self._stream:
            if isinstance(chunk, StreamContentChunk):
                yield chunk.delta.content

    def collect(self) -> str:
        """Drain the stream and return all content fragments concatenated."""
        return "".join(
            chunk.delta.content for chunk in self._stream if isinstance(chunk, StreamContentChunk)
        )


class AsyncChatStream:
    """Async version of :class:`ChatStream` for use with ``AsyncPinecone``.

    The stream is single-pass: iterating, calling :meth:`text`, or calling
    :meth:`collect` all consume the same underlying async iterator.

    Example::

        stream = await pc.assistants.chat(assistant_name="my-assistant",
                                          messages=[{"content": "Hi"}], stream=True)
        async for text in stream.text():
            print(text, end="", flush=True)
    """

    def __init__(self, stream: AsyncIterator[ChatStreamChunk]) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[ChatStreamChunk]:
        return self._stream

    async def text(self) -> AsyncIterator[str]:
        """Yield text fragments, skipping start/citation/end chunks."""
        async for chunk in self._stream:
            if isinstance(chunk, StreamContentChunk):
                yield chunk.delta.content

    async def collect(self) -> str:
        """Drain the stream and return all content fragments concatenated."""
        return "".join(
            [
                chunk.delta.content
                async for chunk in self._stream
                if isinstance(chunk, StreamContentChunk)
            ]
        )


class ChatCompletionStream:
    """Wraps an OpenAI-compatible streaming response for convenient text access.

    Iterating over this object yields the full :class:`ChatCompletionStreamChunk`
    sequence.  :meth:`text` filters to non-empty content fragments and handles
    the ``None`` sentinel values that appear in role-only and finish chunks.

    The stream is single-pass: iterating, calling :meth:`text`, or calling
    :meth:`collect` all consume the same underlying iterator.

    Example::

        stream = pc.assistants.chat_completions(assistant_name="my-assistant",
                                                messages=[{"content": "Hi"}],
                                                stream=True)
        for text in stream.text():
            print(text, end="", flush=True)
    """

    def __init__(self, stream: Iterator[ChatCompletionStreamChunk]) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[ChatCompletionStreamChunk]:
        return self._stream

    def text(self) -> Iterator[str]:
        """Yield non-empty content strings, skipping role-only and finish chunks."""
        for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    yield content

    def collect(self) -> str:
        """Drain the stream and return all content fragments concatenated."""
        parts: list[str] = []
        for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    parts.append(content)
        return "".join(parts)


class AsyncChatCompletionStream:
    """Async version of :class:`ChatCompletionStream` for use with ``AsyncPinecone``.

    The stream is single-pass: iterating, calling :meth:`text`, or calling
    :meth:`collect` all consume the same underlying async iterator.

    Example::

        stream = await pc.assistants.chat_completions(assistant_name="my-assistant",
                                                      messages=[{"content": "Hi"}],
                                                      stream=True)
        async for text in stream.text():
            print(text, end="", flush=True)
    """

    def __init__(self, stream: AsyncIterator[ChatCompletionStreamChunk]) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[ChatCompletionStreamChunk]:
        return self._stream

    async def text(self) -> AsyncIterator[str]:
        """Yield non-empty content strings, skipping role-only and finish chunks."""
        async for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    yield content

    async def collect(self) -> str:
        """Drain the stream and return all content fragments concatenated."""
        parts: list[str] = []
        async for chunk in self._stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    parts.append(content)
        return "".join(parts)


class ChatCompletionStreamDelta(Struct, kw_only=True):
    """The delta payload within a chat completion streaming chunk.

    Attributes:
        role: The role of the message author, or ``None`` if not provided.
        content: The text content fragment, or ``None`` if not provided.
    """

    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(Struct, kw_only=True):
    """A single choice in a chat completion streaming chunk.

    Attributes:
        index: The index of this choice in the choices list.
        delta: The delta message for this choice.
        finish_reason: The reason the model stopped generating,
            or ``None`` if generation is ongoing.
    """

    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamChunk(Struct, kw_only=True):
    """A streaming chunk from the OpenAI-compatible chat completion endpoint.

    Attributes:
        id: Unique identifier for this chunk.
        choices: List of streaming choices.
        model: The model used to generate the response, or ``None`` if not provided.
        object: The object type (typically ``"chat.completion.chunk"``), or ``None``.
        created: Unix timestamp when the chunk was created, or ``None``.
        system_fingerprint: Opaque fingerprint identifying the backend, or ``None``.
    """

    id: str
    choices: list[ChatCompletionStreamChoice]
    model: str | None = None
    object: str | None = None
    created: int | None = None
    system_fingerprint: str | None = None
