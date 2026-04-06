"""RestoreJobs namespace — list and describe restore job operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pinecone._internal.adapters.restore_jobs_adapter import RestoreJobsAdapter
from pinecone._internal.validation import require_non_empty
from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import RestoreJobModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class RestoreJobs:
    """Control-plane operations for Pinecone restore jobs.

    Provides methods to list and describe restore jobs.

    Args:
        http (HTTPClient): HTTP client for making API requests.

    Examples:

        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        for job in pc.restore_jobs.list():
            print(job.restore_job_id)
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http
        self._adapter = RestoreJobsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "RestoreJobs()"

    def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> RestoreJobList:
        """List all restore jobs in the project.

        Supports cursor-based pagination. Defaults to at most 10 results
        per page when no limit is specified.

        Args:
            limit (int | None): Maximum number of results per page.
            pagination_token (str | None): Token for cursor-based pagination.

        Returns:
            A RestoreJobList supporting iteration, len(), and index access.

        Raises:
            ApiError: If the API returns an error response.

        Examples:

            jobs = pc.restore_jobs.list(limit=5)
            for job in jobs:
                print(job.restore_job_id, job.status)
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if pagination_token is not None:
            params["paginationToken"] = pagination_token

        logger.info("Listing restore jobs")
        response = self._http.get("/restore-jobs", params=params)
        result = self._adapter.to_restore_job_list(response.content)
        logger.debug("Listed %d restore jobs", len(result))
        return result

    def describe(self, *, job_id: str) -> RestoreJobModel:
        """Get detailed information about a restore job.

        Args:
            job_id (str): The identifier of the restore job to describe.

        Returns:
            A RestoreJobModel with full restore job details.

        Raises:
            ValidationError: If *job_id* is empty.
            NotFoundError: If the restore job does not exist.
            ApiError: If the API returns another error response.

        Examples:

            job = pc.restore_jobs.describe(job_id="rj-123")
            print(job.status, job.percent_complete)
        """
        require_non_empty("job_id", job_id)
        logger.info("Describing restore job %r", job_id)
        response = self._http.get(f"/restore-jobs/{job_id}")
        result = self._adapter.to_restore_job(response.content)
        logger.debug("Described restore job %r", job_id)
        return result
