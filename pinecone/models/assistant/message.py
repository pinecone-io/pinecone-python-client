"""Message model for the Assistant API."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models.assistant._mixin import StructDictMixin


class Message(StructDictMixin, Struct, kw_only=True):
    """A message to send to an assistant.

    Attributes:
        content: The text content of the message.
        role: The role of the message author. Defaults to ``"user"``.
    """

    content: str
    role: str = "user"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        """Create a ``Message`` from a dictionary.

        Extracts ``"content"`` and ``"role"`` keys, defaulting role to
        ``"user"`` when not present.

        Args:
            d: A dictionary with at least a ``"content"`` key.

        Returns:
            A new ``Message`` instance.
        """
        return cls(content=d["content"], role=d.get("role", "user"))
