"""Backwards-compatibility shim for :mod:`pinecone.models.assistant`.

Re-exports shared model classes that used to live at
:mod:`pinecone_plugins.assistant.models.shared` before the ``python-sdk2``
rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass


@dataclass
class Message(BaseDataclass):
    """A chat message with role and content."""

    content: str
    role: str = "user"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(role=d.get("role", "user"), content=d.get("content", ""))

    @classmethod
    def from_openapi(cls, message_model: Any) -> "Message":
        return cls(role=message_model.role, content=message_model.content)


@dataclass
class Usage(BaseDataclass):
    """Token usage statistics for a chat request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Usage":
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )

    @classmethod
    def from_openapi(cls, usage_model: Any) -> "Usage":
        return cls(
            prompt_tokens=usage_model.prompt_tokens,
            completion_tokens=usage_model.completion_tokens,
            total_tokens=usage_model.total_tokens,
        )


@dataclass
class TokenCounts(BaseDataclass):
    """Token usage statistics (alias for :class:`Usage`)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenCounts":
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )

    @classmethod
    def from_openapi(cls, token_counts: Any) -> "TokenCounts":
        return cls(
            prompt_tokens=token_counts.prompt_tokens,
            completion_tokens=token_counts.completion_tokens,
            total_tokens=token_counts.total_tokens,
        )


__all__ = ["Message", "TokenCounts", "Usage"]
