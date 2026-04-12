"""Assistant data models."""
from __future__ import annotations

from pinecone.models.assistant.chat import (
    ChatCitation,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatHighlight,
    ChatMessage,
    ChatReference,
    ChatResponse,
    ChatUsage,
)
from pinecone.models.assistant.context import (
    ContextContentBlock,
    ContextImageBlock,
    ContextImageData,
    ContextReference,
    ContextResponse,
    ContextSnippet,
    ContextTextBlock,
    FileReference,
    MultimodalSnippet,
    PageReference,
    TextSnippet,
)
from pinecone.models.assistant.evaluation import (
    AlignmentResult,
    AlignmentScores,
    EntailmentResult,
)
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.message import Message
from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.options import ContextOptions
from pinecone.models.assistant.streaming import (
    ChatCompletionStreamChoice,
    ChatCompletionStreamChunk,
    ChatCompletionStreamDelta,
    ChatStreamChunk,
    StreamCitationChunk,
    StreamContentChunk,
    StreamContentDelta,
    StreamMessageEnd,
    StreamMessageStart,
)

__all__ = [
    "AlignmentResult",
    "AlignmentScores",
    "AssistantFileModel",
    "AssistantModel",
    "ChatCitation",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "ChatCompletionStreamChoice",
    "ChatCompletionStreamChunk",
    "ChatCompletionStreamDelta",
    "ChatHighlight",
    "ChatMessage",
    "ChatReference",
    "ChatResponse",
    "ChatStreamChunk",
    "ChatUsage",
    "ContextContentBlock",
    "ContextImageBlock",
    "ContextImageData",
    "ContextOptions",
    "ContextReference",
    "ContextResponse",
    "ContextSnippet",
    "ContextTextBlock",
    "EntailmentResult",
    "FileReference",
    "ListAssistantsResponse",
    "ListFilesResponse",
    "Message",
    "MultimodalSnippet",
    "PageReference",
    "StreamCitationChunk",
    "StreamContentChunk",
    "StreamContentDelta",
    "StreamMessageEnd",
    "StreamMessageStart",
    "TextSnippet",
]
