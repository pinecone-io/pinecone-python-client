"""Preview backup adapters (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

import msgspec

from pinecone.preview.models.backups import PreviewBackupModel

__all__ = [
    "PreviewDescribeBackupAdapter",
    "PreviewListBackupsAdapter",
    "describe_backup_adapter",
    "list_backups_adapter",
]


class PreviewDescribeBackupAdapter:
    """Adapter for preview describe_backup / create_backup operations."""

    def from_response(self, data: dict[str, Any]) -> PreviewBackupModel:
        return msgspec.convert(data, PreviewBackupModel)


class PreviewListBackupsAdapter:
    """Adapter for preview list_backups operation."""

    def from_response(
        self, data: dict[str, Any]
    ) -> tuple[list[PreviewBackupModel], str | None]:
        items = [
            msgspec.convert(item, PreviewBackupModel) for item in data.get("data", [])
        ]
        token: str | None = (data.get("pagination") or {}).get("next")
        return (items, token)


describe_backup_adapter = PreviewDescribeBackupAdapter()
list_backups_adapter = PreviewListBackupsAdapter()
