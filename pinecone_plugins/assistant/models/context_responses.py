"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.context`.

Re-exports context response classes that used to live at
:mod:`pinecone_plugins.assistant.models.context_responses` before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Union

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.file_model import FileModel
from pinecone_plugins.assistant.models.shared import TokenCounts

# ---------------------------------------------------------------------------
# Type variables for generic dispatch
# ---------------------------------------------------------------------------

RefType = TypeVar(
    "RefType",
    bound=Union[
        "TextReference",
        "PdfReference",
        "MarkdownReference",
        "JsonReference",
        "DocxReference",
    ],
)

SnippetType = TypeVar(
    "SnippetType",
    bound=Union["TextSnippet", "MultimodalSnippet"],
)

MultimodalContentBlockType = TypeVar(
    "MultimodalContentBlockType",
    bound=Union["TextBlock", "ImageBlock"],
)


# ---------------------------------------------------------------------------
# Reference models
# ---------------------------------------------------------------------------


@dataclass
class BaseReference(BaseDataclass):
    """Base class for all source file reference types."""

    type: str

    @classmethod
    def from_openapi(cls, value: Any) -> "BaseReference":
        raise NotImplementedError


@dataclass
class PdfReference(BaseReference):
    """A reference to a PDF source file."""

    pages: list[int]
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict[str, Any]) -> "PdfReference":
        return cls(
            type=ref_dict["type"],
            pages=ref_dict["pages"],
            file=FileModel.from_dict(ref_dict["file"]),
        )


@dataclass
class DocxReference(BaseReference):
    """A reference to a DOCX source file."""

    pages: list[int]
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict[str, Any]) -> "DocxReference":
        return cls(
            type=ref_dict["type"],
            pages=ref_dict["pages"],
            file=FileModel.from_dict(ref_dict["file"]),
        )


@dataclass
class TextReference(BaseReference):
    """A reference to a plain-text source file."""

    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict[str, Any]) -> "TextReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


@dataclass
class MarkdownReference(BaseReference):
    """A reference to a Markdown source file."""

    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict[str, Any]) -> "MarkdownReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


@dataclass
class JsonReference(BaseReference):
    """A reference to a JSON source file."""

    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict[str, Any]) -> "JsonReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


class TypedReference:
    """Factory that dispatches a raw reference dict to the correct reference class."""

    @classmethod
    def from_openapi(cls, d: dict[str, Any]) -> BaseReference:
        type_ = d["type"]
        ref_map: dict[str, type[BaseReference]] = {
            "text": TextReference,
            "doc_x": DocxReference,
            "pdf": PdfReference,
            "markdown": MarkdownReference,
            "json": JsonReference,
        }
        ref_cls = ref_map.get(type_)
        if ref_cls is None:
            raise ValueError(f"Unknown reference type: {type_!r}")
        return ref_cls.from_openapi(d)


# ---------------------------------------------------------------------------
# Multimodal content blocks
# ---------------------------------------------------------------------------


@dataclass
class Image(BaseDataclass):
    """Base64-encoded image data within a context snippet."""

    mime_type: str
    data: str
    type: str

    @classmethod
    def from_openapi(cls, d: dict[str, Any]) -> "Image":
        return cls(mime_type=d["mime_type"], data=d["data"], type=d["type"])


@dataclass
class TextBlock(BaseDataclass):
    """A text content block within a multimodal snippet."""

    type: str
    text: str

    @classmethod
    def from_openapi(cls, d: dict[str, Any]) -> "TextBlock":
        return cls(type=d["type"], text=d["text"])


@dataclass
class ImageBlock(BaseDataclass):
    """An image content block within a multimodal snippet."""

    type: str
    caption: str
    image: Optional[Image]

    @classmethod
    def from_openapi(cls, d: dict[str, Any]) -> "ImageBlock":
        image_info = d.get("image")
        image: Optional[Image] = Image.from_openapi(image_info) if image_info else None
        return cls(type=d["type"], caption=d["caption"], image=image)


# ---------------------------------------------------------------------------
# Snippet models
# ---------------------------------------------------------------------------


@dataclass
class TextSnippet(BaseDataclass):
    """A text context snippet from a source document."""

    type: str
    content: str
    score: float
    reference: BaseReference

    @classmethod
    def from_openapi(cls, d: dict[str, Any]) -> "TextSnippet":
        return cls(
            type=d["type"],
            content=d["content"],
            score=d["score"],
            reference=TypedReference.from_openapi(d["reference"]),
        )


@dataclass
class MultimodalSnippet(BaseDataclass):
    """A multimodal context snippet containing text and/or image blocks."""

    type: str
    content: list[Any]
    score: float
    reference: BaseReference

    @classmethod
    def from_openapi(cls, d: dict[str, Any]) -> "MultimodalSnippet":
        block_map: dict[str, type[Any]] = {"text": TextBlock, "image": ImageBlock}
        blocks = [block_map[block["type"]].from_openapi(block) for block in d["content"]]
        return cls(
            type=d["type"],
            content=blocks,
            score=d["score"],
            reference=TypedReference.from_openapi(d["reference"]),
        )


@dataclass
class Snippet:
    """Factory that dispatches a raw snippet dict to the correct snippet class."""

    @classmethod
    def from_openapi(cls, snippet: dict[str, Any]) -> TextSnippet | MultimodalSnippet:
        type_ = snippet["type"]
        snippet_map: dict[str, type[TextSnippet | MultimodalSnippet]] = {
            "text": TextSnippet,
            "multimodal": MultimodalSnippet,
        }
        snippet_cls = snippet_map.get(type_)
        if snippet_cls is None:
            raise ValueError(f"Unknown snippet type: {type_!r}")
        return snippet_cls.from_openapi(snippet)


# ---------------------------------------------------------------------------
# Context response
# ---------------------------------------------------------------------------


@dataclass
class ContextResponse(BaseDataclass):
    """Response from the assistant context (RAG) endpoint."""

    id: str
    snippets: list[Any]
    usage: TokenCounts

    @classmethod
    def from_openapi(cls, ctx: Any) -> "ContextResponse":
        return cls(
            id=ctx.id,
            snippets=[Snippet.from_openapi(snippet) for snippet in ctx.snippets],
            usage=TokenCounts.from_openapi(ctx.usage),
        )


__all__ = [
    "BaseReference",
    "ContextResponse",
    "DocxReference",
    "Image",
    "ImageBlock",
    "JsonReference",
    "MarkdownReference",
    "MultimodalSnippet",
    "PdfReference",
    "Snippet",
    "TextBlock",
    "TextReference",
    "TextSnippet",
    "TypedReference",
]
