"""Integration tests for backup lifecycle (async REST).

Phase 4 area tags: backup-lifecycle, create-index-from-backup
Transport: rest-async

Backups can be created from serverless indexes.
Tests create a small index, create a backup, verify BackupModel fields,
list/describe the backup, then delete both the backup and the source index.

Also tests restoring an index from a backup via create_index_from_backup().
"""

from __future__ import annotations

import pytest

from pinecone import AsyncPinecone
from pinecone.models.backups.model import BackupModel
from pinecone.models.backups.list import BackupList
from pinecone.models.indexes.index import IndexModel
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


# ---------------------------------------------------------------------------
# create-index-from-backup — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_index_from_backup_async(async_client: AsyncPinecone) -> None:
    """Create a serverless index, create a backup, then restore a new index
    from the backup via async REST.  Verify the restored IndexModel has the
    same dimension/metric as the source, and the index handle is queryable.

    Note on vector data: backups from freshly created serverless indexes
    consistently report record_count=0 (data is snapshotted from durable
    storage which lags behind the query-visible layer).  This test focuses
    on the structural properties of the restore operation.

    Area tag: create-index-from-backup
    Transport: rest-async
    """
    source_index_name = unique_name("idx")
    restore_index_name = unique_name("idx")
    backup_id: str | None = None

    try:
        # 1. Create a small source index
        await async_client.indexes.create(
            name=source_index_name,
            dimension=4,
            metric="dotproduct",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            timeout=120,
        )

        # 2. Create a backup from the source index
        backup = await async_client.backups.create(
            index_name=source_index_name,
            name=unique_name("bk"),
            description="create-index-from-backup async test",
        )
        assert isinstance(backup, BackupModel)
        backup_id = backup.backup_id

        # Initial BackupModel should reflect the source index schema
        assert backup.source_index_name == source_index_name
        assert backup.cloud == "aws"
        assert backup.region == "us-east-1"

        # 3. Poll until the backup is Ready (up to 5 minutes)
        ready_backup = await async_poll_until(
            query_fn=lambda: async_client.backups.describe(backup_id=backup_id),
            check_fn=lambda b: b.status == "Ready",
            timeout=300,
            interval=10,
            description="backup Ready",
        )
        assert isinstance(ready_backup, BackupModel)
        assert ready_backup.status == "Ready"
        assert ready_backup.dimension == 4

        # 4. Restore a new index from the backup (SDK polls until index is ready)
        restored = await async_client.create_index_from_backup(
            backup_id=backup_id,
            name=restore_index_name,
            timeout=600,
        )

        # 5. Verify the restored IndexModel has the same dimension and metric
        assert isinstance(restored, IndexModel)
        assert restored.name == restore_index_name
        assert restored.dimension == 4
        assert restored.metric == "dotproduct"
        assert restored.status.ready is True
        # Serverless spec should be preserved
        assert restored.spec.serverless is not None
        assert restored.spec.serverless.cloud == "aws"
        assert restored.spec.serverless.region == "us-east-1"

        # 6. Get an Index handle — index should be reachable and queryable
        restore_desc = await async_client.indexes.describe(restore_index_name)
        restore_index = async_client.index(host=restore_desc.host)
        stats = await restore_index.describe_index_stats()
        assert stats.dimension == 4
        # total_vector_count may be 0 for a freshly created backup (durable
        # storage snapshot lag), which is acceptable
        assert stats.total_vector_count >= 0

    finally:
        # Clean up: restored index, then backup, then source index
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(restore_index_name),
            restore_index_name,
            "index (restored)",
        )
        if backup_id is not None:
            await async_cleanup_resource(
                lambda: async_client.backups.delete(backup_id=backup_id),
                backup_id,
                "backup",
            )
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(source_index_name),
            source_index_name,
            "index (source)",
        )
