"""Preview backup adapters (2026-01.alpha API)."""

from __future__ import annotations

from msgspec import Struct

from pinecone._internal.adapters._decode import decode_response
from pinecone.preview.models.backups import PreviewBackupModel

__all__ = [
    "PreviewDescribeBackupAdapter",
    "PreviewListBackupsAdapter",
    "describe_backup_adapter",
    "list_backups_adapter",
]


class _Pagination(Struct, kw_only=True):
    next: str | None = None


class _BackupListEnvelope(Struct, kw_only=True):
    data: list[PreviewBackupModel] = []
    pagination: _Pagination | None = None


class PreviewDescribeBackupAdapter:
    """Adapter for preview describe_backup / create_backup operations."""

    @staticmethod
    def from_response(data: bytes) -> PreviewBackupModel:
        return decode_response(data, PreviewBackupModel)


class PreviewListBackupsAdapter:
    """Adapter for preview list_backups operation."""

    @staticmethod
    def from_response(data: bytes) -> tuple[list[PreviewBackupModel], str | None]:
        envelope = decode_response(data, _BackupListEnvelope)
        token = envelope.pagination.next if envelope.pagination is not None else None
        return (envelope.data, token)


describe_backup_adapter = PreviewDescribeBackupAdapter()
list_backups_adapter = PreviewListBackupsAdapter()
