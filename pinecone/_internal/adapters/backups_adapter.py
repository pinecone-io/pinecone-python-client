"""Adapter for Backups API responses."""

from __future__ import annotations

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone.models.backups.list import BackupList
from pinecone.models.backups.model import BackupModel, CreateIndexFromBackupResponse
from pinecone.models.vectors.responses import Pagination


class _BackupListEnvelope(Struct, kw_only=True):
    """Internal envelope for the list-backups response."""

    data: list[BackupModel] = []
    pagination: Pagination | None = None


class BackupsAdapter:
    """Transforms raw API JSON into BackupModel / BackupList instances."""

    @staticmethod
    def to_backup(data: bytes) -> BackupModel:
        """Decode raw JSON bytes into a BackupModel."""
        return decode_response(data, BackupModel)

    @staticmethod
    def to_backup_list(data: bytes) -> BackupList:
        """Decode raw JSON bytes from a list-backups response into a BackupList."""
        envelope = decode_response(data, _BackupListEnvelope)
        return BackupList(envelope.data, pagination=envelope.pagination)

    @staticmethod
    def to_create_index_from_backup_response(data: bytes) -> CreateIndexFromBackupResponse:
        """Decode raw JSON bytes into a CreateIndexFromBackupResponse."""
        return decode_response(data, CreateIndexFromBackupResponse)
