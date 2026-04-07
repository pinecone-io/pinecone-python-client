"""Context response models for the Assistant API."""

from __future__ import annotations

from typing import TypeAlias

from msgspec import Struct

from pinecone.models.assistant.chat import ChatUsage


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


class ContextImageBlock(Struct, kw_only=True):
    """An image block within a multimodal context snippet.

    Attributes:
        caption: A text caption describing the image.
        image_data: The image data, or ``None`` when binary content
            is excluded from the response.
    """

    caption: str
    image_data: ContextImageData | None = None


class ContextTextBlock(Struct, kw_only=True):
    """A text block within a multimodal context snippet.

    Attributes:
        text: The text content of the block.
    """

    text: str


ContextContentBlock: TypeAlias = ContextTextBlock | ContextImageBlock
"""A content block within a multimodal snippet — either text or image."""


class FileReference(Struct, kw_only=True):
    """A reference to a text, markdown, or JSON source file.

    Attributes:
        file: The name or identifier of the source file.
    """

    file: str


class PageReference(Struct, kw_only=True):
    """A reference to a PDF or DOCX source file with page numbers.

    Attributes:
        file: The name or identifier of the source file.
        pages: The list of page numbers relevant to the snippet.
    """

    file: str
    pages: list[int]


ContextReference: TypeAlias = FileReference | PageReference
"""A reference to a source file — either file-only or file with pages."""


class TextSnippet(Struct, kw_only=True):
    """A text context snippet from a source document.

    Attributes:
        type: The snippet type (``"text"``).
        content: The text content of the snippet.
        score: The relevance score of the snippet.
        reference: A reference to the source file.
    """

    type: str
    content: str
    score: float
    reference: FileReference | PageReference


class MultimodalSnippet(Struct, kw_only=True):
    """A multimodal context snippet containing text and/or image blocks.

    Attributes:
        type: The snippet type (``"multimodal"``).
        content: The list of content blocks (text and/or image).
        score: The relevance score of the snippet.
        reference: A reference to the source file.
    """

    type: str
    content: list[ContextContentBlock]
    score: float
    reference: FileReference | PageReference


ContextSnippet: TypeAlias = TextSnippet | MultimodalSnippet
"""A context snippet — either text or multimodal."""


class ContextResponse(Struct, kw_only=True):
    """Response from the assistant context endpoint.

    Attributes:
        id: Unique identifier for the context response.
        snippets: The list of context snippets.
        usage: Token usage statistics for the request.
    """

    id: str
    snippets: list[ContextSnippet]
    usage: ChatUsage
