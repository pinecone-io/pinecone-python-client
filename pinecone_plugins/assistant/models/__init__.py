"""Backwards-compatibility shim for :mod:`pinecone.models.assistant`.

Re-exports classes that used to live at :mod:`pinecone_plugins.assistant.models` before
the `python-sdk2` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from pinecone_plugins.assistant.models.assistant_model import AssistantModel
from pinecone_plugins.assistant.models.chat import (
    ChatResponse,
    ContextOptions,
    Message,
    StreamChatResponseCitation,
    StreamChatResponseContentDelta,
    StreamChatResponseMessageEnd,
    StreamChatResponseMessageStart,
)
from pinecone_plugins.assistant.models.chat_completion import (
    ChatCompletionResponse,
    StreamingChatCompletionChunk,
)
from pinecone_plugins.assistant.models.context_responses import (
    ContextResponse,
    DocxReference,
    JsonReference,
    MarkdownReference,
    PdfReference,
    TextReference,
)
from pinecone_plugins.assistant.models.evaluation_responses import AlignmentResponse
from pinecone_plugins.assistant.models.file_model import FileModel
from pinecone_plugins.assistant.models.list_assistants_response import ListAssistantsResponse
from pinecone_plugins.assistant.models.list_files_response import ListFilesResponse

__all__ = [
    "AlignmentResponse",
    "AssistantModel",
    "ChatCompletionResponse",
    "ChatResponse",
    "ContextOptions",
    "ContextResponse",
    "DocxReference",
    "FileModel",
    "JsonReference",
    "ListAssistantsResponse",
    "ListFilesResponse",
    "MarkdownReference",
    "Message",
    "PdfReference",
    "StreamChatResponseCitation",
    "StreamChatResponseContentDelta",
    "StreamChatResponseMessageEnd",
    "StreamChatResponseMessageStart",
    "StreamingChatCompletionChunk",
    "TextReference",
]
