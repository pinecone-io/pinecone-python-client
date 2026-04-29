"""Unit tests for backwards-compatibility type aliases in pinecone.models.assistant."""

from __future__ import annotations


def test_legacy_type_aliases_exist() -> None:
    from pinecone.models.assistant import (
        AlignmentResponse,
        AlignmentResult,
        AlignmentScores,
        AssistantFileModel,
        ChatCitation,
        ChatHighlight,
        ChatReference,
        ChatUsage,
        Citation,
        ContextImageBlock,
        ContextImageData,
        ContextTextBlock,
        DocxReference,
        EntailmentResult,
        EvaluatedFact,
        FileModel,
        FileReference,
        Highlight,
        Image,
        ImageBlock,
        JsonReference,
        MarkdownReference,
        MessageDelta,
        Metrics,
        PdfReference,
        Reference,
        StreamContentDelta,
        TextBlock,
        TextReference,
        TokenCounts,
        Usage,
    )

    assert FileModel is AssistantFileModel
    assert Usage is ChatUsage
    assert TokenCounts is ChatUsage
    assert Citation is ChatCitation
    assert Reference is ChatReference
    assert Highlight is ChatHighlight
    assert MessageDelta is StreamContentDelta
    assert AlignmentResponse is AlignmentResult
    assert Metrics is AlignmentScores
    assert EvaluatedFact is EntailmentResult
    assert TextBlock is ContextTextBlock
    assert ImageBlock is ContextImageBlock
    assert Image is ContextImageData
    assert PdfReference is FileReference
    assert TextReference is FileReference
    assert JsonReference is FileReference
    assert MarkdownReference is FileReference
    assert DocxReference is FileReference


def test_stream_response_aliases_exist() -> None:
    from pinecone.models.assistant import (
        ChatCompletionStreamChoice,
        ChatCompletionStreamChunk,
        StreamChatResponseCitation,
        StreamChatResponseContentDelta,
        StreamChatResponseMessageEnd,
        StreamChatResponseMessageStart,
        StreamCitationChunk,
        StreamContentChunk,
        StreamingChatCompletionChoice,
        StreamingChatCompletionChunk,
        StreamMessageEnd,
        StreamMessageStart,
    )

    assert StreamChatResponseMessageStart is StreamMessageStart
    assert StreamChatResponseContentDelta is StreamContentChunk
    assert StreamChatResponseCitation is StreamCitationChunk
    assert StreamChatResponseMessageEnd is StreamMessageEnd
    assert StreamingChatCompletionChunk is ChatCompletionStreamChunk
    assert StreamingChatCompletionChoice is ChatCompletionStreamChoice


def test_base_stream_chat_response_chunk_alias() -> None:
    from pinecone.models.assistant import BaseStreamChatResponseChunk, ChatStreamChunk

    assert BaseStreamChatResponseChunk is ChatStreamChunk


def test_dict_mixin_alias_exists() -> None:
    from pinecone.models.assistant._mixin import DictMixin, StructDictMixin

    assert DictMixin is StructDictMixin


def test_aliases_in_all() -> None:
    import pinecone.models.assistant as mod

    expected_aliases = [
        "AlignmentResponse",
        "BaseStreamChatResponseChunk",
        "Citation",
        "DocxReference",
        "EvaluatedFact",
        "FileModel",
        "Highlight",
        "Image",
        "ImageBlock",
        "JsonReference",
        "MarkdownReference",
        "MessageDelta",
        "Metrics",
        "PdfReference",
        "Reference",
        "StreamChatResponseCitation",
        "StreamChatResponseContentDelta",
        "StreamChatResponseMessageEnd",
        "StreamChatResponseMessageStart",
        "StreamingChatCompletionChoice",
        "StreamingChatCompletionChunk",
        "TextBlock",
        "TextReference",
        "TokenCounts",
        "Usage",
    ]
    for name in expected_aliases:
        assert name in mod.__all__, f"{name!r} missing from __all__"
        assert hasattr(mod, name), f"{name!r} not importable from module"
