"""Assistant operation response model."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.assistant._mixin import StructDictMixin

__all__ = ["OperationModel"]


class OperationModel(StructDictMixin, Struct, kw_only=True):
    """Response model for a long-running assistant operation.

    Returned by the 2026-04 upsert endpoint (``PUT /files/{assistant_name}/{file_id}``).

    Attributes:
        operation_id: Unique identifier for the operation.
        status: Current status of the operation (e.g. ``"Processing"``, ``"Succeeded"``,
            ``"Failed"``).
        created_at: ISO 8601 timestamp when the operation was created, or ``None``.
        error: Error message if the operation failed, or ``None``.
    """

    operation_id: str
    status: str
    created_at: str | None = None
    error: str | None = None
