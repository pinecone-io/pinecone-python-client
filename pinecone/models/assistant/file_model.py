"""Assistant file response model."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.assistant._mixin import StructDictMixin


class AssistantFileModel(
    StructDictMixin,
    Struct,
    kw_only=True,
    rename={"content_hash": "crc32c_hash"},
):
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
        content_hash: Hash of the file content (wire key ``crc32c_hash``), or
            ``None`` when not available.  Legacy callers can also access this
            value via the :attr:`crc32c_hash` property alias.
        percent_done: Processing progress as a percentage (0.0–100.0), or ``None`` when not
            available or not applicable.
        error_message: Error message describing why processing failed, or ``None`` when
            processing succeeded or is still in progress.
    """

    name: str
    id: str
    metadata: dict[str, object] | None = None
    created_on: str | None = None
    updated_on: str | None = None
    status: str | None = None
    size: int | None = None
    multimodal: bool | None = None
    signed_url: str | None = None
    content_hash: str | None = None
    percent_done: float | None = None
    error_message: str | None = None

    @property
    def crc32c_hash(self) -> str | None:
        """Backwards-compatibility alias for :attr:`content_hash`."""
        return self.content_hash
