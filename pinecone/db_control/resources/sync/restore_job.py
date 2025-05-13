from typing import Optional

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
from pinecone.core.openapi.db_control.model.restore_job_model import RestoreJobModel
from pinecone.core.openapi.db_control.model.restore_job_list import RestoreJobList
from pinecone.utils import parse_non_empty_args, require_kwargs


class RestoreJobResource:
    def __init__(self, index_api: ManageIndexesApi):
        self._index_api = index_api
        """ @private """

    @require_kwargs
    def get(self, *, restore_job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID.

        Args:
            restore_job_id (str): The ID of the restore job to get.

        Returns:
            RestoreJobModel: The restore job.
        """
        return self._index_api.describe_restore_job(restore_job_id=restore_job_id)

    @require_kwargs
    def describe(self, *, restore_job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID. Alias for get.

        Args:
            restore_job_id (str): The ID of the restore job to get.

        Returns:
            RestoreJobModel: The restore job.
        """
        return self.get(restore_job_id=restore_job_id)

    @require_kwargs
    def list(
        self, *, limit: Optional[int] = 10, pagination_token: Optional[str] = None
    ) -> RestoreJobList:
        """
        List all restore jobs.

        Args:
            limit (int): The maximum number of restore jobs to return.
            pagination_token (str): The pagination token to use for the next page of restore jobs.

        Returns:
            List[RestoreJobModel]: The list of restore jobs.
        """
        args = parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])
        return self._index_api.list_restore_jobs(**args)
