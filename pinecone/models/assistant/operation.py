"""Assistant operation response model."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import HtmlBuilder, safe_display, truncate_text
from pinecone.models.assistant._mixin import StructDictMixin

__all__ = ["OperationModel"]


class OperationModel(
    StructDictMixin,
    Struct,
    kw_only=True,
    rename={"operation_id": "id", "created_at": "created_on"},
):
    """Response model for a long-running assistant operation.

    Returned by the 2026-04 upsert endpoint (``PUT /files/{assistant_name}/{file_id}``).

    The API uses ``id`` and ``created_on``; the rename mapping presents them as
    ``operation_id`` and ``created_at`` in Python for clarity.

    Attributes:
        operation_id: Unique identifier for the operation (JSON field: ``id``).
        status: Current status of the operation (e.g. ``"Processing"``, ``"Succeeded"``,
            ``"Failed"``).
        created_at: ISO 8601 timestamp when the operation was created, or ``None``
            (JSON field: ``created_on``).
        error: Error message if the operation failed, or ``None``.
    """

    operation_id: str
    status: str
    created_at: str | None = None
    error: str | None = None

    @safe_display
    def __repr__(self) -> str:
        parts = [f"operation_id={self.operation_id!r}", f"status={self.status!r}"]
        if self.error is not None:
            parts.append(f"error={truncate_text(self.error, 80)!r}")
        return f"OperationModel({', '.join(parts)})"

    @safe_display
    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("OperationModel(...)")
            return
        with p.group(2, "OperationModel(", ")"):
            p.breakable()
            p.text(f"operation_id={self.operation_id!r},")
            p.breakable()
            p.text(f"status={self.status!r},")
            if self.created_at is not None:
                p.breakable()
                p.text(f"created_at={self.created_at!r},")
            if self.error is not None:
                p.breakable()
                p.text(f"error={truncate_text(self.error, 80)!r},")

    @safe_display
    def _repr_html_(self) -> str:
        builder = HtmlBuilder("OperationModel")
        builder.row("Operation ID:", self.operation_id)
        builder.row("Status:", self.status)
        if self.created_at is not None:
            builder.row("Created:", self.created_at)
        if self.error is not None:
            builder.section("Error", [("Message", truncate_text(self.error, 200))], theme="error")
        return builder.build()
