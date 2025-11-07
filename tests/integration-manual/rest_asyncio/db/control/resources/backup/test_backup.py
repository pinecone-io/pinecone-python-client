import pytest
import random
import asyncio
from tests.integration.helpers import random_string
import logging
from pinecone import PineconeAsyncio

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
class TestBackups:
    async def test_create_backup(self, ready_sl_index, index_tags):
        async with PineconeAsyncio() as pc:
            desc = await pc.db.index.describe(name=ready_sl_index)
            dimension = desc.dimension

            # Upsert some sample data
            ns = random_string(10)
            async with pc.IndexAsyncio(host=desc.host) as idx:
                batch_size = 100
                num_batches = 10
                for _ in range(num_batches):
                    await idx.upsert(
                        vectors=[
                            {
                                "id": random_string(15),
                                "values": [random.random() for _ in range(dimension)],
                            }
                            for _ in range(batch_size)
                        ],
                        namespace=ns,
                    )

                logger.debug("Sleeping for 180 seconds to ensure vectors are indexed")
                await asyncio.sleep(180)

                index_stats = await idx.describe_index_stats()
                logger.debug(f"Index stats for index {ready_sl_index}: {index_stats}")

            backup_name = "backup-" + random_string(10)
            backup = await pc.db.backup.create(backup_name=backup_name, index_name=ready_sl_index)
            assert backup.backup_id is not None
            assert backup.name == backup_name
            assert backup.source_index_name == ready_sl_index

            # Describe the backup
            backup_desc = await pc.db.backup.describe(backup_id=backup.backup_id)
            assert backup_desc.name == backup_name
            assert backup_desc.backup_id == backup.backup_id
            assert backup_desc.source_index_name == ready_sl_index
            logger.info(f"Backup description: {backup_desc}")

            # Wait for the backup to be ready before proceeding
            backup_ready = False
            max_wait = 5 * 60
            while not backup_ready:
                backup_desc = await pc.db.backup.describe(backup_id=backup.backup_id)
                logger.info(f"Backup description: {backup_desc}")
                if backup_desc.status == "Ready":
                    backup_ready = True
                else:
                    if max_wait <= 0:
                        raise Exception("Backup did not become ready in time")
                    max_wait -= 5
                    await asyncio.sleep(5)

            # Verify that the backup shows in list
            backups_list = await pc.db.backup.list(index_name=ready_sl_index)
            assert len(backups_list) >= 1
            assert any(b.name == backup_name for b in backups_list)
            assert any(b.backup_id == backup.backup_id for b in backups_list)
            assert any(b.source_index_name == ready_sl_index for b in backups_list)

            # Create index from backup
            new_index_name = "from-backup-" + random_string(10)
            new_index = await pc.db.index.create_from_backup(
                name=new_index_name, backup_id=backup.backup_id, tags=index_tags
            )
            assert new_index.name == new_index_name
            assert new_index.tags is not None
            assert new_index.dimension == desc.dimension
            assert new_index.metric == desc.metric

            # Can list restore jobs
            restore_jobs = await pc.db.restore_job.list()
            assert len(restore_jobs) >= 1

            # Verify that the new index has the same data as the original index
            new_desc = await pc.db.index.describe(name=new_index_name)
            async with pc.IndexAsyncio(host=new_desc.host) as new_idx:
                stats = await new_idx.describe_index_stats()
                logger.info(f"New index stats: {stats}")
                assert stats.namespaces[ns].vector_count == batch_size * num_batches

            # Delete the new index
            await pc.db.index.delete(name=new_index_name)

            # Delete the backup
            await pc.db.backup.delete(backup_id=backup.backup_id)

            # Verify that the backup is deleted
            with pytest.raises(Exception):
                await pc.db.backup.describe(backup_id=backup.backup_id)

    async def test_create_backup_legacy_syntax(self, ready_sl_index, index_tags):
        async with PineconeAsyncio() as pc:
            desc = await pc.describe_index(name=ready_sl_index)
            dimension = desc.dimension

            # Upsert some sample data
            ns = random_string(10)
            async with pc.IndexAsyncio(host=desc.host) as idx:
                batch_size = 100
                num_batches = 10
                for _ in range(num_batches):
                    await idx.upsert(
                        vectors=[
                            {
                                "id": random_string(15),
                                "values": [random.random() for _ in range(dimension)],
                            }
                            for _ in range(batch_size)
                        ],
                        namespace=ns,
                    )

                logger.debug("Sleeping for 180 seconds to ensure vectors are indexed")
                await asyncio.sleep(180)

                index_stats = await idx.describe_index_stats()
                logger.debug(f"Index stats for index {ready_sl_index}: {index_stats}")

            backup_name = "backup-" + random_string(10)
            backup = await pc.create_backup(backup_name=backup_name, index_name=ready_sl_index)
            assert backup.backup_id is not None
            assert backup.name == backup_name
            assert backup.source_index_name == ready_sl_index

            # Describe the backup
            backup_desc = await pc.describe_backup(backup_id=backup.backup_id)
            assert backup_desc.name == backup_name
            assert backup_desc.backup_id == backup.backup_id
            assert backup_desc.source_index_name == ready_sl_index
            logger.info(f"Backup description: {backup_desc}")

            # Wait for the backup to be ready before proceeding
            backup_ready = False
            max_wait = 5 * 60
            while not backup_ready:
                backup_desc = await pc.describe_backup(backup_id=backup.backup_id)
                logger.info(f"Backup description: {backup_desc}")
                if backup_desc.status == "Ready":
                    backup_ready = True
                else:
                    if max_wait <= 0:
                        raise Exception("Backup did not become ready in time")
                    max_wait -= 5
                    await asyncio.sleep(5)

            # Verify that the backup shows in list
            backups_list = await pc.list_backups(index_name=ready_sl_index)
            assert len(backups_list) >= 1
            assert any(b.name == backup_name for b in backups_list)
            assert any(b.backup_id == backup.backup_id for b in backups_list)
            assert any(b.source_index_name == ready_sl_index for b in backups_list)

            # Create index from backup
            new_index_name = "from-backup-" + random_string(10)
            new_index = await pc.create_index_from_backup(
                name=new_index_name, backup_id=backup.backup_id, tags=index_tags
            )
            assert new_index.name == new_index_name
            assert new_index.tags is not None
            assert new_index.dimension == desc.dimension
            assert new_index.metric == desc.metric

            # Can list restore jobs
            restore_jobs = await pc.list_restore_jobs()
            assert len(restore_jobs) >= 1

            # Verify that the new index has the same data as the original index
            new_desc = await pc.db.index.describe(name=new_index_name)
            async with pc.IndexAsyncio(host=new_desc.host) as new_idx:
                stats = await new_idx.describe_index_stats()
                logger.info(f"New index stats: {stats}")
                assert stats.namespaces[ns].vector_count == batch_size * num_batches

            # Delete the new index
            await pc.delete_index(name=new_index_name)

            # Delete the backup
            await pc.delete_backup(backup_id=backup.backup_id)

            # Verify that the backup is deleted
            with pytest.raises(Exception):
                await pc.describe_backup(backup_id=backup.backup_id)

            # Verify that the new index is deleted
            backup_list = await pc.list_backups(index_name=ready_sl_index)
            assert len(backup_list) == 0
