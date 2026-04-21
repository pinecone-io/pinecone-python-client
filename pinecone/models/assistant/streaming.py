"""Streaming chunk models for the Assistant API.

Models for chat streaming (Pinecone-native format with type-based dispatch)
and chat completion streaming (OpenAI-compatible format).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, TypeAlias

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.chat import ChatCitation, ChatUsage


class StreamMessageStart(
    StructDictMixin, Struct, kw_only=True, tag="message_start", tag_field="type"
):
    """First chunk in a chat stream, containing the model and role.

    Attributes:
        type: Discriminator value ``"message_start"``.
        model: The model used to generate the response.
        role: The role of the message author (e.g. ``"assistant"``).
    """

    model: str
    role: str

    @property
    def type(self) -> str:
        """Discriminator value, always ``"message_start"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        return f"StreamMessageStart(model={self.model!r}, role={self.role!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamMessageStart(...)")
            return
        with p.group(2, "StreamMessageStart(", ")"):
            p.breakable()
            p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"role={self.role!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamMessageStart")
        builder.row("Type:", self.type)
        builder.row("Model:", self.model)
        builder.row("Role:", self.role)
        return builder.build()


class StreamContentDelta(StructDictMixin, Struct, kw_only=True):
    """The delta payload within a content chunk.

    Attributes:
        content: The text fragment for this chunk.
    """

    content: str

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        return f"StreamContentDelta(content={truncate_text(self.content, 80)!r})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamContentDelta(...)")
            return
        with p.group(2, "StreamContentDelta(", ")"):
            p.breakable()
            p.text(f"content={truncate_text(self.content, 200)!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamContentDelta")
        builder.row("Content", truncate_text(self.content, 500))
        return builder.build()


class StreamContentChunk(
    StructDictMixin, Struct, kw_only=True, tag="content_chunk", tag_field="type"
):
    """A content chunk containing a text fragment in a delta object.

    Attributes:
        type: Discriminator value ``"content_chunk"``.
        id: Unique identifier for this chunk.
        delta: The delta object containing the text fragment.
        model: The model used to generate this response, or ``None`` if not provided.
    """

    id: str
    delta: StreamContentDelta
    model: str | None = None

    @property
    def type(self) -> str:
        """Discriminator value, always ``"content_chunk"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        model_part = f", model={self.model!r}" if self.model is not None else ""
        return f"StreamContentChunk(id={self.id!r}, delta={self.delta!r}{model_part})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamContentChunk(...)")
            return
        with p.group(2, "StreamContentChunk(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"delta={self.delta!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamContentChunk")
        builder.row("Type:", self.type)
        builder.row("Id:", self.id)
        if self.model is not None:
            builder.row("Model:", self.model)
        builder.row("Content:", truncate_text(self.delta.content, 500))
        return builder.build()


class StreamCitationChunk(StructDictMixin, Struct, kw_only=True, tag="citation", tag_field="type"):
    """A citation chunk linking response text to source references.

    Attributes:
        type: Discriminator value ``"citation"``.
        id: Unique identifier for this chunk.
        citation: The citation data with position and references.
        model: The model used to generate this response, or ``None`` if not provided.
    """

    id: str
    citation: ChatCitation
    model: str | None = None

    @property
    def type(self) -> str:
        """Discriminator value, always ``"citation"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        model_part = f", model={self.model!r}" if self.model is not None else ""
        return f"StreamCitationChunk(id={self.id!r}, citation={self.citation!r}{model_part})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamCitationChunk(...)")
            return
        with p.group(2, "StreamCitationChunk(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"citation={self.citation!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamCitationChunk")
        builder.row("Type:", self.type)
        builder.row("Id:", self.id)
        if self.model is not None:
            builder.row("Model:", self.model)
        builder.row("Position:", self.citation.position)
        builder.row("References:", len(self.citation.references))
        return builder.build()


class StreamMessageEnd(StructDictMixin, Struct, kw_only=True, tag="message_end", tag_field="type"):
    """Final chunk in a chat stream, containing token usage statistics.

    Attributes:
        type: Discriminator value ``"message_end"``.
        id: Unique identifier for this chunk.
        usage: Token usage statistics for the request.
        model: The model used to generate this response, or ``None`` if not provided.
    """

    id: str
    usage: ChatUsage
    model: str | None = None

    @property
    def type(self) -> str:
        """Discriminator value, always ``"message_end"``."""
        return str(self.__struct_config__.tag)

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        model_part = f", model={self.model!r}" if self.model is not None else ""
        return f"StreamMessageEnd(id={self.id!r}, usage={self.usage!r}{model_part})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("StreamMessageEnd(...)")
            return
        with p.group(2, "StreamMessageEnd(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            p.breakable()
            p.text(f"usage={self.usage!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("StreamMessageEnd")
        builder.row("Type:", self.type)
        builder.row("Id:", self.id)
        if self.model is not None:
            builder.row("Model:", self.model)
        builder.row("Prompt tokens:", self.usage.prompt_tokens)
        builder.row("Completion tokens:", self.usage.completion_tokens)
        builder.row("Total tokens:", self.usage.total_tokens)
        return builder.build()


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

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        return (
            "ChatStream(single-pass, Pinecone-native chat stream"
            " — iterate with `for chunk in stream` or `stream.text()`)"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatStream")
        builder.row("Type", "Pinecone-native chat stream")
        builder.row("Iteration", "single-pass")
        builder.row(
            "Usage hint",
            "Iterate with `for chunk in stream`, or call `.text()` for"
            " text-only fragments, or `.collect()` for the full message",
        )
        return builder.build()

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

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        return (
            "AsyncChatStream(single-pass async, Pinecone-native chat stream"
            " — iterate with `async for chunk in stream` or `await stream.collect()`)"
        )

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("AsyncChatStream")
        builder.row("Type", "Pinecone-native chat stream")
        builder.row("Iteration", "single-pass async")
        builder.row(
            "Usage hint",
            "Iterate with `async for chunk in stream`, or call `.text()` for"
            " text-only fragments, or `await .collect()` for the full message",
        )
        return builder.build()

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


class ChatCompletionStreamDelta(StructDictMixin, Struct, kw_only=True):
    """The delta payload within a chat completion streaming chunk.

    Attributes:
        role: The role of the message author, or ``None`` if not provided.
        content: The text content fragment, or ``None`` if not provided.
    """

    role: str | None = None
    content: str | None = None

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        parts: list[str] = []
        if self.role is not None:
            parts.append(f"role={self.role!r}")
        if self.content is not None:
            parts.append(f"content={truncate_text(self.content, 80)!r}")
        inner = ", ".join(parts) if parts else "<empty>"
        return f"ChatCompletionStreamDelta({inner})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionStreamDelta(...)")
            return
        parts: list[str] = []
        if self.role is not None:
            parts.append(f"role={self.role!r}")
        if self.content is not None:
            parts.append(f"content={truncate_text(self.content, 200)!r}")
        if not parts:
            p.text("ChatCompletionStreamDelta(<empty>)")
            return
        with p.group(2, "ChatCompletionStreamDelta(", ")"):
            for part in parts:
                p.breakable()
                p.text(f"{part},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionStreamDelta")
        if self.role is not None:
            builder.row("Role", self.role)
        if self.content is not None:
            builder.row("Content", truncate_text(self.content, 500))
        return builder.build()


class ChatCompletionStreamChoice(StructDictMixin, Struct, kw_only=True):
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

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        parts: list[str] = [f"index={self.index!r}", f"delta={self.delta!r}"]
        if self.finish_reason is not None:
            parts.append(f"finish_reason={self.finish_reason!r}")
        return f"ChatCompletionStreamChoice({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionStreamChoice(...)")
            return
        with p.group(2, "ChatCompletionStreamChoice(", ")"):
            p.breakable()
            p.text(f"index={self.index!r},")
            p.breakable()
            p.text(f"delta={self.delta!r},")
            if self.finish_reason is not None:
                p.breakable()
                p.text(f"finish_reason={self.finish_reason!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionStreamChoice")
        builder.row("Index", self.index)
        if self.delta.role is not None:
            builder.row("Role", self.delta.role)
        if self.delta.content is not None:
            builder.row("Content", truncate_text(self.delta.content, 500))
        if self.finish_reason is not None:
            builder.row("Finish reason", self.finish_reason)
        return builder.build()


class ChatCompletionStreamChunk(StructDictMixin, Struct, kw_only=True):
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

    @safe_display
    def __repr__(self) -> str:  # type: ignore[override]
        parts: list[str] = [f"id={self.id!r}"]
        if self.model is not None:
            parts.append(f"model={self.model!r}")
        parts.append(f"choices={len(self.choices)}")
        return f"ChatCompletionStreamChunk({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("ChatCompletionStreamChunk(...)")
            return
        first_content: str | None = None
        if self.choices and self.choices[0].delta.content is not None:
            first_content = truncate_text(self.choices[0].delta.content, max_chars=200)
        with p.group(2, "ChatCompletionStreamChunk(", ")"):
            p.breakable()
            p.text(f"id={self.id!r},")
            if self.model is not None:
                p.breakable()
                p.text(f"model={self.model!r},")
            if self.object is not None:
                p.breakable()
                p.text(f"object={self.object!r},")
            if self.created is not None:
                p.breakable()
                p.text(f"created={self.created!r},")
            if self.system_fingerprint is not None:
                p.breakable()
                p.text(f"system_fingerprint={self.system_fingerprint!r},")
            p.breakable()
            p.text(f"choices={len(self.choices)},")
            if first_content is not None:
                p.breakable()
                p.text(f"first_choice_content={first_content!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("ChatCompletionStreamChunk")
        builder.row("Id", self.id)
        if self.model is not None:
            builder.row("Model", self.model)
        if self.object is not None:
            builder.row("Object", self.object)
        if self.created is not None:
            builder.row("Created", self.created)
        if self.system_fingerprint is not None:
            builder.row("System fingerprint", self.system_fingerprint)
        builder.row("Choices", len(self.choices))
        if self.choices:
            first = self.choices[0]
            section_rows: list[tuple[str, Any]] = [("Index", first.index)]
            if first.delta.role is not None:
                section_rows.append(("Role", first.delta.role))
            if first.delta.content is not None:
                section_rows.append(("Content", truncate_text(first.delta.content, 500)))
            if first.finish_reason is not None:
                section_rows.append(("Finish reason", first.finish_reason))
            builder.section("First choice", section_rows)
        return builder.build()
