"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.chat`.

Re-exports chat completion classes that used to live at
:mod:`pinecone_plugins.assistant.models.chat_completion` before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any, List

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.shared import Message, Usage


@dataclass
class ChatCompletionChoice(BaseDataclass):
    """A single choice in a chat completion response."""

    index: int
    message: Message
    finish_reason: str


@dataclass
class ChatCompletionResponse(BaseDataclass):
    """A non-streaming chat completion response."""

    id: str
    choices: List[ChatCompletionChoice]
    model: str
    usage: Usage

    @classmethod
    def from_openapi(cls, chat_completion_model: Any) -> "ChatCompletionResponse":
        return cls(
            id=chat_completion_model.id,
            choices=[
                ChatCompletionChoice(
                    index=choice.index,
                    message=Message.from_openapi(choice.message),
                    finish_reason=choice.finish_reason,
                )
                for choice in chat_completion_model.choices
            ],
            model=chat_completion_model.model,
            usage=Usage.from_openapi(chat_completion_model.usage),
        )


@dataclass
class StreamingChatCompletionChoice(BaseDataclass):
    """A single choice in a streaming chat completion chunk."""

    index: int
    delta: Message
    finish_reason: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamingChatCompletionChoice":
        return cls(
            index=d.get("index", 0),
            delta=Message.from_dict(d.get("delta") or {}),
            finish_reason=d.get("finish_reason", ""),
        )


@dataclass
class StreamingChatCompletionChunk(BaseDataclass):
    """A streaming chunk from the OpenAI-compatible chat completion endpoint."""

    id: str
    choices: List[StreamingChatCompletionChoice]
    model: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamingChatCompletionChunk":
        return cls(
            id=d.get("id", ""),
            choices=[StreamingChatCompletionChoice.from_dict(c) for c in d.get("choices", [])],
            model=d.get("model", ""),
        )


__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "StreamingChatCompletionChoice",
    "StreamingChatCompletionChunk",
]
