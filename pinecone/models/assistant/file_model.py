"""Assistant file response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class AssistantFileModel(Struct, kw_only=True):
    """Response model for a file attached to a Pinecone assistant.

    Attributes:
        name: The name of the file.
        id: Unique identifier for the file.
        metadata: Optional metadata dictionary associated with the file,
            or ``None`` if not set.
        created_on: ISO 8601 timestamp when the file was created, or ``None``.
        updated_on: ISO 8601 timestamp when the file was last updated, or ``None``.
        status: Current status of the file (e.g. ``"Processing"``,
            ``"Available"``, ``"Deleting"``, ``"ProcessingFailed"``),
            or ``None``.
        size: Size of the file in bytes, or ``None``.
        multimodal: Whether the file was processed as multimodal, or ``None``.
        signed_url: A temporary signed URL for downloading the file, or ``None``
            when not requested or unavailable.
        content_hash: Hash of the file content, or ``None`` when not available.
    """

    name: str
    id: str
    metadata: dict[str, Any] | None = None
    created_on: str | None = None
    updated_on: str | None = None
    status: str | None = None
    size: int | None = None
    multimodal: bool | None = None
    signed_url: str | None = None
    content_hash: str | None = None
    percent_done: float | None = None
    error_message: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. ``file['name']``)."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'name' in file``)."""
        return key in self.__struct_fields__
