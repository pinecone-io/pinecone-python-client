"""Chat response models for the Assistant API."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models.assistant._mixin import StructDictMixin
from pinecone.models.assistant.file_model import AssistantFileModel


class ChatUsage(StructDictMixin, Struct, kw_only=True):
    """Token usage information for a chat request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatUsage:
        """Construct a ``ChatUsage`` from a plain dict representation.

        Missing token count fields default to 0.
        """
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )


class ChatHighlight(StructDictMixin, Struct, kw_only=True):
    """A highlighted portion of a referenced document.

    Attributes:
        type: The type of highlight (e.g. ``"text"``).
        content: The highlighted text content.
    """

    type: str
    content: str


class ChatReference(StructDictMixin, Struct, kw_only=True):
    """A single reference within a citation.

    Attributes:
        file: The source file object with metadata.
        pages: Optional list of page numbers in the source file.
        highlight: Optional highlight from the referenced document,
            or ``None`` when highlights are not requested.
    """

    file: AssistantFileModel
    pages: list[int] | None = None
    highlight: ChatHighlight | None = None


class ChatCitation(StructDictMixin, Struct, kw_only=True):
    """A citation linking a position in the response to source references.

    Attributes:
        position: The character position of the citation in the response content.
        references: The list of references supporting this citation.
    """

    position: int
    references: list[ChatReference]


class ChatMessage(StructDictMixin, Struct, kw_only=True):
    """A message in a chat conversation.

    Attributes:
        role: The role of the message author (e.g. ``"user"``, ``"assistant"``).
        content: The text content of the message.
    """

    role: str
    content: str


class ChatResponse(StructDictMixin, Struct, kw_only=True):
    """Non-streaming response from the assistant chat endpoint.

    Attributes:
        id: Unique identifier for the chat response.
        model: The model used to generate the response.
        usage: Token usage statistics for the request.
        message: The assistant's response message.
        finish_reason: The reason the model stopped generating
            (e.g. ``"stop"``, ``"length"``).
        citations: List of citations linking response text to source documents.
    """

    id: str
    model: str
    usage: ChatUsage
    message: ChatMessage
    finish_reason: str
    citations: list[ChatCitation]


class ChatCompletionChoice(StructDictMixin, Struct, kw_only=True):
    """A single choice in a chat completion response.

    Attributes:
        index: The index of this choice in the choices list.
        message: The message content for this choice.
        finish_reason: The reason the model stopped generating.
    """

    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(StructDictMixin, Struct, kw_only=True):
    """Non-streaming response from the OpenAI-compatible chat completion endpoint.

    Attributes:
        id: Unique identifier for the chat completion.
        model: The model used to generate the response.
        usage: Token usage statistics for the request.
        choices: List of completion choices.
    """

    id: str
    model: str
    usage: ChatUsage
    choices: list[ChatCompletionChoice]
