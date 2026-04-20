"""Backwards-compatibility shim for :mod:`pinecone.models.assistant`.

Re-exports shared model classes that used to live at this path before the
``python-sdk2`` rewrite. ``Message`` aliases the canonical
:class:`pinecone.models.assistant.message.Message`. ``Usage`` and
``TokenCounts`` alias :class:`pinecone.models.assistant.chat.ChatUsage`
(the rewrite consolidated the two structurally-identical types under a
single canonical class).

New code should import from :mod:`pinecone.models.assistant`.

:meta private:
"""

from __future__ import annotations

from pinecone.models.assistant import Message
from pinecone.models.assistant.chat import ChatUsage

# ``Usage`` and ``TokenCounts`` are structurally identical (prompt_tokens,
# completion_tokens, total_tokens) and alias the canonical ``ChatUsage``.
Usage = ChatUsage
TokenCounts = ChatUsage

__all__ = ["Message", "TokenCounts", "Usage"]
