from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
from pinecone.db_control.models import RestoreJobModel, RestoreJobList
from pinecone.utils import parse_non_empty_args, require_kwargs


class RestoreJobResourceAsyncio:
    def __init__(self, index_api: AsyncioManageIndexesApi):
        self._index_api = index_api
        """ :meta private: """

    @require_kwargs
    async def get(self, *, job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID.

        :param job_id: The ID of the restore job to get.
        :type job_id: str
        :return: The restore job.
        :rtype: RestoreJobModel
        """
        job = await self._index_api.describe_restore_job(job_id=job_id)
        return RestoreJobModel(job)

    @require_kwargs
    async def describe(self, *, job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID. Alias for get.

        :param job_id: The ID of the restore job to get.
        :type job_id: str
        :return: The restore job.
        :rtype: RestoreJobModel
        """
        return await self.get(job_id=job_id)

    @require_kwargs
    async def list(
        self, *, limit: int | None = 10, pagination_token: str | None = None
    ) -> RestoreJobList:
        """
        List all restore jobs.

        :param limit: The maximum number of restore jobs to return.
        :type limit: int, optional
        :param pagination_token: The pagination token to use for the next page of restore jobs.
        :type pagination_token: str, optional
        :return: The list of restore jobs.
        :rtype: RestoreJobList
        """
        args = parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])
        jobs = await self._index_api.list_restore_jobs(**args)
        return RestoreJobList(jobs)
