import pytest
from pinecone import PineconeAsyncio, PineconeApiException
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
class TestRestoreJobDescribe:
    async def test_describe_restore_job(self):
        async with PineconeAsyncio() as pc:
            jobs = await pc.db.restore_job.list()
            assert len(jobs.data) >= 1

            restore_job_id = jobs.data[0].restore_job_id
            restore_job = await pc.db.restore_job.describe(job_id=restore_job_id)
            logger.debug(f"Restore job: {restore_job}")

            assert restore_job.restore_job_id == restore_job_id
            assert restore_job.backup_id is not None
            assert isinstance(restore_job.status, str)
            assert isinstance(restore_job.backup_id, str)
            if restore_job.status == "Completed":
                assert isinstance(restore_job.completed_at, datetime)
            assert isinstance(restore_job.created_at, datetime)
            if restore_job.status != "Pending":
                assert isinstance(restore_job.percent_complete, float)
            assert isinstance(restore_job.target_index_id, str)
            assert isinstance(restore_job.target_index_name, str)

    async def test_describe_restore_job_legacy_syntax(self):
        async with PineconeAsyncio() as pc:
            jobs = await pc.list_restore_jobs()
            assert len(jobs.data) >= 1

            restore_job_id = jobs.data[0].restore_job_id
            restore_job = await pc.describe_restore_job(job_id=restore_job_id)
            logger.debug(f"Restore job: {restore_job}")

    async def test_describe_restore_job_with_invalid_job_id(self):
        async with PineconeAsyncio() as pc:
            with pytest.raises(PineconeApiException):
                await pc.db.restore_job.describe(job_id="invalid")
