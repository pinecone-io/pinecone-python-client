"""Preview adapters (2026-01.alpha API)."""

from __future__ import annotations

from pinecone.preview._internal.adapters.backups import (
    PreviewDescribeBackupAdapter,
    PreviewListBackupsAdapter,
    describe_backup_adapter,
    list_backups_adapter,
)
from pinecone.preview._internal.adapters.documents import (
    decode_fetch_response,
    decode_search_response,
)
from pinecone.preview._internal.adapters.indexes import (
    PreviewConfigureIndexAdapter,
    PreviewCreateIndexAdapter,
    PreviewDescribeIndexAdapter,
    PreviewListIndexesAdapter,
    configure_adapter,
    create_adapter,
    describe_adapter,
    list_adapter,
)

__all__ = [
    "PreviewConfigureIndexAdapter",
    "PreviewCreateIndexAdapter",
    "PreviewDescribeBackupAdapter",
    "PreviewDescribeIndexAdapter",
    "PreviewListBackupsAdapter",
    "PreviewListIndexesAdapter",
    "configure_adapter",
    "create_adapter",
    "decode_fetch_response",
    "decode_search_response",
    "describe_adapter",
    "describe_backup_adapter",
    "list_adapter",
    "list_backups_adapter",
]
