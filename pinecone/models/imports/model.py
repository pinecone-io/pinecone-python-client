"""Bulk import response models."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


class ImportModel(StructDictMixin, Struct, kw_only=True, rename="camel"):
    """Response model for a bulk import operation.

    Attributes:
        id: Unique identifier for the import operation.
        uri: Source URI for the import data.
        status: Current status of the import (Pending, InProgress, Failed, Completed, Cancelled).
        created_at: Timestamp when the import was created.
        finished_at: Timestamp when the import finished.
        percent_complete: Percentage of the import that has completed.
        records_imported: Number of records imported so far.
        error: Error message if the import failed.
    """

    id: str
    uri: str
    status: str
    created_at: str
    finished_at: str | None = None
    percent_complete: float | None = None
    records_imported: int | None = None
    error: str | None = None


class StartImportResponse(StructDictMixin, Struct, kw_only=True):
    """Response model for starting a bulk import operation.

    Attributes:
        id: Unique identifier for the created import operation.
    """

    id: str
