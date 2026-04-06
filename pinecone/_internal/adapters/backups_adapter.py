"""Adapter for Backups API responses."""

from __future__ import annotations

import msgspec
from msgspec import Struct

from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel, CreateIndexFromBackupResponse


class _Pagination(Struct, kw_only=True):
    """Pagination token from the backup list response."""

    next: str | None = None


class _BackupListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-backups response."""

    data: list[BackupModel] = []
    pagination: _Pagination | None = None


class BackupsAdapter:
    """Transforms raw API JSON into BackupModel / BackupList instances."""

    @staticmethod
    def to_backup(data: bytes) -> BackupModel:
        """Decode raw JSON bytes into a BackupModel."""
        return msgspec.json.decode(data, type=BackupModel)

    @staticmethod
    def to_backup_list(data: bytes) -> BackupList:
        """Decode raw JSON bytes from a list-backups response into a BackupList."""
        envelope = msgspec.json.decode(data, type=_BackupListEnvelope)
        return BackupList(envelope.data)

    @staticmethod
    def to_create_index_from_backup_response(data: bytes) -> CreateIndexFromBackupResponse:
        """Decode raw JSON bytes into a CreateIndexFromBackupResponse."""
        return msgspec.json.decode(data, type=CreateIndexFromBackupResponse)
