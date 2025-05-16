import pytest
import logging
from pinecone import PineconeAsyncio, PineconeApiValueError, PineconeApiException

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
class TestRestoreJobList:
    async def test_list_restore_jobs_no_arguments(self):
        async with PineconeAsyncio() as pc:
            restore_jobs = await pc.db.restore_job.list()
            assert restore_jobs.data is not None
            logger.debug(f"Restore jobs count: {len(restore_jobs.data)}")

            # This assumes the backup test has been run at least once
            # in the same project.
            assert len(restore_jobs.data) >= 1

    async def test_list_restore_jobs_with_optional_arguments(self):
        async with PineconeAsyncio() as pc:
            restore_jobs = await pc.db.restore_job.list(limit=2)
            assert restore_jobs.data is not None
            logger.debug(f"Restore jobs count: {len(restore_jobs.data)}")
            assert len(restore_jobs.data) <= 2

            if len(restore_jobs.data) == 2:
                logger.debug(f"Restore jobs pagination: {restore_jobs.pagination}")
                assert restore_jobs.pagination is not None
                assert restore_jobs.pagination.next is not None

                next_page = await pc.db.restore_job.list(
                    limit=2, pagination_token=restore_jobs.pagination.next
                )
                assert next_page.data is not None
                assert len(next_page.data) <= 2

    async def test_list_restore_jobs_legacy_syntax(self):
        async with PineconeAsyncio() as pc:
            restore_jobs = await pc.list_restore_jobs(limit=2)
            assert restore_jobs.data is not None
            logger.debug(f"Restore jobs count: {len(restore_jobs.data)}")
            assert len(restore_jobs.data) <= 2

            if len(restore_jobs.data) == 2:
                logger.debug(f"Restore jobs pagination: {restore_jobs.pagination}")
                assert restore_jobs.pagination is not None
                assert restore_jobs.pagination.next is not None

                next_page = await pc.list_restore_jobs(
                    limit=2, pagination_token=restore_jobs.pagination.next
                )
                assert next_page.data is not None
                assert len(next_page.data) <= 2


@pytest.mark.asyncio
class TestRestoreJobListErrors:
    async def test_list_restore_jobs_with_invalid_limit(self):
        async with PineconeAsyncio() as pc:
            with pytest.raises(PineconeApiValueError):
                await pc.db.restore_job.list(limit=-1)

    async def test_list_restore_jobs_with_invalid_pagination_token(self):
        async with PineconeAsyncio() as pc:
            with pytest.raises(PineconeApiException):
                await pc.db.restore_job.list(pagination_token="invalid")
