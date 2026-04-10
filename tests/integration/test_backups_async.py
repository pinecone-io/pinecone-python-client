"""Integration tests for backup lifecycle (async REST).

Phase 4 area tag: backup-lifecycle
Transport: rest-async

Backups can be created from serverless indexes.
Tests create a small index, create a backup, verify BackupModel fields,
list/describe the backup, then delete both the backup and the source index.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.models.backups.model import BackupModel
from pinecone.models.backups.list import BackupList
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name


# ---------------------------------------------------------------------------
# backup-lifecycle — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_backup_lifecycle_async(async_client: AsyncPinecone) -> None:
    """Full backup lifecycle via async REST: create a serverless index,
    create a backup, poll until Ready, verify BackupModel fields, list
    (with and without index_name filter), describe, then delete backup and index.

    Area tag: backup-lifecycle
    Transport: rest-async
    """
    index_name = unique_name("idx")
    backup_id: str | None = None

    try:
        # 1. Create a small serverless index to back up
        await async_client.indexes.create(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=120,
        )

        # 2. Create a backup from the index
        backup = await async_client.backups.create(
            index_name=index_name,
            name=unique_name("bk"),
            description="integration test backup",
        )
        assert isinstance(backup, BackupModel)
        backup_id = backup.backup_id

        # Verify initial BackupModel fields
        assert backup.backup_id != ""
        assert backup.source_index_name == index_name
        assert backup.status in ("Initializing", "Pending", "Ready")
        assert backup.cloud == "aws"
        assert backup.region == "us-east-1"
        assert backup.dimension == 2
        assert backup.created_at is not None

        # 3. Poll until the backup is Ready
        ready_backup = await async_poll_until(
            query_fn=lambda: async_client.backups.describe(backup_id=backup_id),
            check_fn=lambda b: b.status == "Ready",
            timeout=300,
            interval=10,
            description="backup Ready",
        )
        assert isinstance(ready_backup, BackupModel)
        assert ready_backup.backup_id == backup_id
        assert ready_backup.status == "Ready"
        assert ready_backup.source_index_name == index_name

        # 4. list() without index_name filter — backup appears in full list
        all_backups = await async_client.backups.list()
        assert isinstance(all_backups, BackupList)
        all_ids = [b.backup_id for b in all_backups]
        assert backup_id in all_ids

        # 5. list() with index_name filter — backup appears in filtered list
        idx_backups = await async_client.backups.list(index_name=index_name)
        assert isinstance(idx_backups, BackupList)
        idx_ids = [b.backup_id for b in idx_backups]
        assert backup_id in idx_ids

        # 6. describe() returns correct BackupModel
        desc = await async_client.backups.describe(backup_id=backup_id)
        assert isinstance(desc, BackupModel)
        assert desc.backup_id == backup_id
        assert desc.source_index_name == index_name
        assert desc.status == "Ready"

    finally:
        # Clean up backup first, then the index
        if backup_id is not None:
            await async_cleanup_resource(
                lambda: async_client.backups.delete(backup_id=backup_id),
                backup_id,
                "backup",
            )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(index_name),
            index_name,
            "index",
        )
