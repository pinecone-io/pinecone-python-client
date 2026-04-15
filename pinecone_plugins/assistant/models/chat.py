"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.chat`.

Re-exports classes that used to live at :mod:`pinecone_plugins.assistant.models.chat` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from __future__ import annotations

from pinecone.models.assistant.chat import (
    ChatCitation as Citation,
    ChatHighlight as Highlight,
    ChatMessage as Message,
    ChatReference as Reference,
    ChatResponse,
    ChatUsage as TokenCounts,
    ChatUsage as Usage,
)
from pinecone.models.assistant.options import ContextOptions
from pinecone.models.assistant.streaming import (
    StreamCitationChunk as StreamChatResponseCitation,
    StreamContentChunk as StreamChatResponseContentDelta,
    StreamContentDelta as MessageDelta,
    StreamMessageEnd as StreamChatResponseMessageEnd,
    StreamMessageStart as StreamChatResponseMessageStart,
)

# BaseStreamChatResponseChunk — no direct canonical equivalent in the rewrite.
# The new SDK uses msgspec.Struct tag dispatch instead of a common base class.
# Expose StructDictMixin as a stand-in so legacy isinstance/issubclass checks
# that test for the presence of the symbol do not fail with NameError.
try:
    from pinecone.models.assistant.chat import BaseStreamChunk as BaseStreamChatResponseChunk  # type: ignore[attr-defined]
except ImportError:
    from pinecone.models.assistant._mixin import StructDictMixin as BaseStreamChatResponseChunk

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
    "TokenCounts",
    "Usage",
]
