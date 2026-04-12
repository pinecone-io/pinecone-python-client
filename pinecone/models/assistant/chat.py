"""Chat response models for the Assistant API."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.assistant.file_model import AssistantFileModel


class ChatUsage(Struct, kw_only=True):
    """Token usage information for a chat request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatHighlight(Struct, kw_only=True):
    """A highlighted portion of a referenced document.

    Attributes:
        type: The type of highlight (e.g. ``"text"``).
        content: The highlighted text content.
    """

    type: str
    content: str


class ChatReference(Struct, kw_only=True):
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


class ChatCitation(Struct, kw_only=True):
    """A citation linking a position in the response to source references.

    Attributes:
        position: The character position of the citation in the response content.
        references: The list of references supporting this citation.
    """

    position: int
    references: list[ChatReference]


class ChatMessage(Struct, kw_only=True):
    """A message in a chat conversation.

    Attributes:
        role: The role of the message author (e.g. ``"user"``, ``"assistant"``).
        content: The text content of the message.
    """

    role: str
    content: str


class ChatResponse(Struct, kw_only=True):
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


class ChatCompletionChoice(Struct, kw_only=True):
    """A single choice in a chat completion response.

    Attributes:
        index: The index of this choice in the choices list.
        message: The message content for this choice.
        finish_reason: The reason the model stopped generating.
    """

    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(Struct, kw_only=True):
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
