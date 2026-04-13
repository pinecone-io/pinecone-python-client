"""Context response models for the Assistant API."""

from __future__ import annotations

from typing import TypeAlias

from msgspec import Struct

from pinecone.models.assistant.chat import ChatUsage
from pinecone.models.assistant.file_model import AssistantFileModel


class ContextImageData(Struct, kw_only=True):
    """Base64-encoded image data within a context snippet.

    Attributes:
        type: The format of the image data (e.g. ``"base64"``).
        mime_type: The MIME type of the image (e.g. ``"image/jpeg"``).
        data: The base64-encoded image data string.
    """

    type: str
    mime_type: str
    data: str


class ContextImageBlock(
    Struct,
    kw_only=True,
    tag="image",
    tag_field="type",
    rename={"image_data": "image"},
):
    """An image block within a multimodal context snippet.

    Attributes:
        caption: A text caption describing the image.
        image_data: The image data, or ``None`` when binary content
            is excluded from the response.
    """

    caption: str
    image_data: ContextImageData | None = None


class ContextTextBlock(Struct, kw_only=True, tag="text", tag_field="type"):
    """A text block within a multimodal context snippet.

    Attributes:
        text: The text content of the block.
    """

    text: str


ContextContentBlock: TypeAlias = ContextTextBlock | ContextImageBlock
"""A content block within a multimodal snippet — either text or image."""


class FileReference(Struct, kw_only=True):
    """A reference to a source file.

    Attributes:
        file: The source file object returned by the API.
        pages: The list of page numbers relevant to the snippet, when
            the source is a paginated document (e.g. PDF). ``None`` for
            text, JSON, or Markdown sources.
    """

    file: AssistantFileModel
    pages: list[int] | None = None


PageReference = FileReference
"""Alias kept for backwards compatibility. Use :class:`FileReference` instead."""

ContextReference: TypeAlias = FileReference
"""A reference to a source file."""


class TextSnippet(Struct, kw_only=True, tag="text", tag_field="type"):
    """A text context snippet from a source document.

    Attributes:
        content: The text content of the snippet.
        score: The relevance score of the snippet.
        reference: A reference to the source file.
    """

    content: str
    score: float
    reference: FileReference


class MultimodalSnippet(Struct, kw_only=True, tag="multimodal", tag_field="type"):
    """A multimodal context snippet containing text and/or image blocks.

    Attributes:
        content: The list of content blocks (text and/or image).
        score: The relevance score of the snippet.
        reference: A reference to the source file.
    """

    content: list[ContextContentBlock]
    score: float
    reference: FileReference


ContextSnippet: TypeAlias = TextSnippet | MultimodalSnippet
"""A context snippet — either text or multimodal."""


class ContextResponse(Struct, kw_only=True):
    """Response from the assistant context endpoint.

    Attributes:
        snippets: The list of context snippets.
        usage: Token usage statistics for the request.
        id: Unique identifier for the context response, or ``None`` if
            not included in the response.
    """

    snippets: list[ContextSnippet]
    usage: ChatUsage
    id: str | None = None
