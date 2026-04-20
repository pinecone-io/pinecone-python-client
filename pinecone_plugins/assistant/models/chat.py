"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.chat`.

Re-exports the chat-related model classes that used to live at this path
before the ``python-sdk2`` rewrite. Legacy names (``Highlight``,
``Reference``, ``Citation``, ``MessageDelta``, ``StreamChatResponse*``)
alias the canonical classes (``ChatHighlight`` etc.) so pre-rewrite callers
continue to work. ``BaseStreamChatResponseChunk`` aliases the canonical
:data:`ChatStreamChunk` union so ``isinstance`` / ``issubclass`` checks
resolve against the four concrete streaming-chunk classes.

New code should import from :mod:`pinecone.models.assistant.chat` and
:mod:`pinecone.models.assistant.streaming`.

:meta private:
"""

from __future__ import annotations

from pinecone.models.assistant import (
    ChatResponse,
    ChatStreamChunk,
    Citation,
    ContextOptions,
    Highlight,
    Message,
    MessageDelta,
    Reference,
    StreamChatResponseCitation,
    StreamChatResponseContentDelta,
    StreamChatResponseMessageEnd,
    StreamChatResponseMessageStart,
)
from pinecone_plugins.assistant.models.shared import Usage

# Legacy callers use ``BaseStreamChatResponseChunk`` as a marker base; the
# canonical SDK models streaming chunks as a tagged-union. Aliasing to the
# union keeps ``isinstance`` / ``issubclass`` working against any of the
# four concrete chunk classes.
BaseStreamChatResponseChunk = ChatStreamChunk

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
