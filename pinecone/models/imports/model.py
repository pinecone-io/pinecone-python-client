"""Bulk import response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class ImportModel(Struct, kw_only=True, rename="camel"):
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

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. model['id'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'id' in model``)."""
        return key in self.__struct_fields__


class StartImportResponse(Struct, kw_only=True):
    """Response model for starting a bulk import operation.

    Attributes:
        id: Unique identifier for the created import operation.
    """

    id: str

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['id'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'id' in response``)."""
        return key in self.__struct_fields__
