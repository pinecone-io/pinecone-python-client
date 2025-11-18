import pytest
import logging
from pinecone import Pinecone, PineconeApiValueError, PineconeApiException

logger = logging.getLogger(__name__)


@pytest.mark.skip
class TestRestoreJobList:
    def test_list_restore_jobs_no_arguments(self, pc: Pinecone):
        restore_jobs = pc.db.restore_job.list()
        assert restore_jobs.data is not None
        logger.debug(f"Restore jobs count: {len(restore_jobs.data)}")

        # This assumes the backup test has been run at least once
        # in the same project.
        assert len(restore_jobs.data) >= 1

    def test_list_restore_jobs_with_optional_arguments(self, pc: Pinecone):
        restore_jobs = pc.db.restore_job.list(limit=2)
        assert restore_jobs.data is not None
        logger.debug(f"Restore jobs count: {len(restore_jobs.data)}")
        assert len(restore_jobs.data) <= 2

        if len(restore_jobs.data) == 2:
            logger.debug(f"Restore jobs pagination: {restore_jobs.pagination}")
            assert restore_jobs.pagination is not None
            assert restore_jobs.pagination.next is not None

            next_page = pc.db.restore_job.list(
                limit=2, pagination_token=restore_jobs.pagination.next
            )
            assert next_page.data is not None
            assert len(next_page.data) <= 2

    def test_list_restore_jobs_legacy_syntax(self, pc: Pinecone):
        restore_jobs = pc.list_restore_jobs(limit=2)
        assert restore_jobs.data is not None
        logger.debug(f"Restore jobs count: {len(restore_jobs.data)}")
        assert len(restore_jobs.data) <= 2

        if len(restore_jobs.data) == 2:
            logger.debug(f"Restore jobs pagination: {restore_jobs.pagination}")
            assert restore_jobs.pagination is not None
            assert restore_jobs.pagination.next is not None

            next_page = pc.list_restore_jobs(limit=2, pagination_token=restore_jobs.pagination.next)
            assert next_page.data is not None
            assert len(next_page.data) <= 2


class TestRestoreJobListErrors:
    def test_list_restore_jobs_with_invalid_limit(self, pc: Pinecone):
        with pytest.raises(PineconeApiValueError):
            pc.db.restore_job.list(limit=-1)

    def test_list_restore_jobs_with_invalid_pagination_token(self, pc: Pinecone):
        with pytest.raises(PineconeApiException):
            pc.db.restore_job.list(pagination_token="invalid")
