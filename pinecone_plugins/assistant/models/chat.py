"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.chat`.

Re-exports classes that used to live at :mod:`pinecone_plugins.assistant.models.chat` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any, Optional

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.file_model import FileModel
from pinecone_plugins.assistant.models.shared import Message, Usage


@dataclass
class Highlight(BaseDataclass):
    """A highlighted excerpt from a referenced document."""

    type: str
    content: str


@dataclass
class Reference(BaseDataclass):
    """A source reference within a citation."""

    file: FileModel
    pages: list[int]
    highlight: Optional[Highlight] = None


@dataclass
class Citation(BaseDataclass):
    """A citation linking a position in the response to source references."""

    position: int
    references: list[Reference]


@dataclass
class ChatResponse(BaseDataclass):
    """Non-streaming response from the assistant chat endpoint."""

    id: str
    model: str
    usage: Usage
    message: Message
    finish_reason: str
    citations: list[Citation]


@dataclass
class ContextOptions(BaseDataclass):
    """Options controlling how context is retrieved for assistant operations."""

    top_k: Optional[int] = None
    snippet_size: Optional[int] = None
    multimodal: Optional[bool] = None
    include_binary_content: Optional[bool] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextOptions":
        return cls(
            top_k=d.get("top_k"),
            snippet_size=d.get("snippet_size"),
            multimodal=d.get("multimodal"),
            include_binary_content=d.get("include_binary_content"),
        )


@dataclass
class BaseStreamChatResponseChunk(BaseDataclass):
    """Abstract base for all streaming chat response chunk types."""


@dataclass
class MessageDelta(BaseDataclass):
    """The text content delta within a streaming content chunk."""

    content: str


@dataclass
class StreamChatResponseMessageStart(BaseStreamChatResponseChunk):
    """First chunk in a streaming chat response, containing model and role."""

    type: str
    model: str
    role: str


@dataclass
class StreamChatResponseContentDelta(BaseStreamChatResponseChunk):
    """A content chunk with a text fragment in a delta object."""

    id: str
    type: str
    model: str
    delta: MessageDelta


@dataclass
class StreamChatResponseCitation(BaseStreamChatResponseChunk):
    """A citation chunk linking response text to source references."""

    type: str
    id: str
    model: str
    citation: Citation


@dataclass
class StreamChatResponseMessageEnd(BaseStreamChatResponseChunk):
    """Final chunk in a streaming chat response, containing token usage."""

    type: str
    model: str
    id: str
    usage: Usage


__all__ = [
    "BaseStreamChatResponseChunk",
    "ChatResponse",
    "Citation",
    "ContextOptions",
    "Highlight",
    "Message",
    "MessageDelta",
    "Reference",
    "StreamChatResponseCitation",
    "StreamChatResponseContentDelta",
    "StreamChatResponseMessageEnd",
    "StreamChatResponseMessageStart",
    "Usage",
]
