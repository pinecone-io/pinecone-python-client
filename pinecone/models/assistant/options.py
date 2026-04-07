"""Context options for assistant chat and context operations."""

from __future__ import annotations

from msgspec import Struct


class ContextOptions(Struct, kw_only=True):
    """Options controlling how context is retrieved for assistant operations.

    All fields are optional and default to ``None``, letting the server
    apply its own defaults.

    Attributes:
        top_k: Maximum number of context snippets to retrieve.
        snippet_size: Target size (in tokens) for each context snippet.
        multimodal: Whether to include multimodal (image) content in
            retrieved context.
        include_binary_content: Whether to include binary file content
            in retrieved context.
    """

    top_k: int | None = None
    snippet_size: int | None = None
    multimodal: bool | None = None
    include_binary_content: bool | None = None
