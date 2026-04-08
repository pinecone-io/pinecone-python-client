"""Streaming chunk models for the Assistant API.

Models for chat streaming (Pinecone-native format with type-based dispatch)
and chat completion streaming (OpenAI-compatible format).
"""

from __future__ import annotations

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
    """

    id: str
    delta: StreamContentDelta


class StreamCitationChunk(Struct, kw_only=True, tag="citation", tag_field="type"):
    """A citation chunk linking response text to source references.

    Attributes:
        id: Unique identifier for this chunk.
        citation: The citation data with position and references.
    """

    id: str
    citation: ChatCitation


class StreamMessageEnd(Struct, kw_only=True, tag="message_end", tag_field="type"):
    """Final chunk in a chat stream, containing token usage statistics.

    Attributes:
        id: Unique identifier for this chunk.
        usage: Token usage statistics for the request.
    """

    id: str
    usage: ChatUsage


ChatStreamChunk: TypeAlias = (
    StreamMessageStart | StreamContentChunk | StreamCitationChunk | StreamMessageEnd
)
"""Union of all Pinecone-native chat streaming chunk types."""


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
    """

    id: str
    choices: list[ChatCompletionStreamChoice]
