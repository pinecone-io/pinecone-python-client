"""Async RestoreJobs namespace: list and describe restore jobs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.restore_jobs_adapter import RestoreJobsAdapter
from pinecone._internal.backups_helpers import restore_job_list_params
from pinecone._internal.validation import require_non_empty
from pinecone.models.backups.list import RestoreJobList
from pinecone.models.backups.model import RestoreJobModel

if TYPE_CHECKING:
    from pinecone._internal.http_client import AsyncHTTPClient

logger = logging.getLogger(__name__)


class AsyncRestoreJobs:
    """Progress reports for restores of a backup into a new index.

    :meth:`~pinecone.AsyncPinecone.create_index_from_backup` hands back a
    ``restore_job_id`` and leaves the restore running in the background; use
    this namespace to follow it to completion. Reached as
    ``pc.restore_jobs``; not constructed directly.

    A restore job is not a backup:
    :class:`~pinecone.async_client.backups.AsyncBackups` manages the snapshots
    themselves, while a job here is a read-only record of one attempt at
    turning a snapshot back into an index.

    Examples:

        .. code-block:: python

            from pinecone import AsyncPinecone

            async with AsyncPinecone(api_key="your-api-key") as pc:
                job = await pc.restore_jobs.describe(job_id="rj-abc123")
                print(job.status, job.target_index_name)

    .. seealso::
       `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__,
       for the exceptions these methods raise and which ones to retry.
    """

    def __init__(self, http: AsyncHTTPClient) -> None:
        self._http = http
        self._adapter = RestoreJobsAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "AsyncRestoreJobs()"

    async def list(
        self,
        *,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> RestoreJobList:
        """List one page of the project's restore jobs.

        One call returns one page, and iterating the result does not fetch
        the next one. Follow ``pagination.next`` yourself (see
        `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__).
        The listing is not exhaustive; see the warning below.

        Args:
            limit (int | None): Maximum number of results per page, e.g.
                ``100``. When ``None``, the server applies its default. The
                SDK drops *limit* when *pagination_token* is given, because
                the token already carries the page size.
            pagination_token (str | None): Token for the next page, taken
                from ``RestoreJobList.pagination.next``.

        Returns:
            A :class:`RestoreJobList` holding one page, supporting iteration,
            ``len()``, and index access. ``pagination`` is ``None`` on the
            final page.

        Examples:
            Walk every page, keying jobs by ``restore_job_id`` because pages
            can overlap:

            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    by_id = {}
                    page = await pc.restore_jobs.list(limit=100)
                    while True:
                        for job in page:
                            by_id[job.restore_job_id] = job
                        if not (page.pagination and page.pagination.next):
                            break
                        page = await pc.restore_jobs.list(
                            pagination_token=page.pagination.next
                        )

                    for job in by_id.values():
                        print(job.restore_job_id, job.target_index_name, job.status)

            When one page is all you want:

            .. code-block:: python

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    page = await pc.restore_jobs.list(limit=5)
                    print(len(page))

        .. warning::
           The restore-job listing is not exhaustive. Pages can overlap,
           pagination can end while jobs remain, and jobs whose target index
           has been deleted are left out. De-duplicate by ``restore_job_id``,
           and never conclude a job does not exist because the listing omits
           it. To track a specific restore, keep its ``restore_job_id`` and
           call :meth:`describe`.
        """
        params: dict[str, Any] = restore_job_list_params(
            limit=limit, pagination_token=pagination_token
        )

        logger.info("Listing restore jobs")
        response = await self._http.get("/restore-jobs", params=params)
        result = self._adapter.to_restore_job_list(response.content)
        logger.debug("Listed %d restore jobs", len(result))
        return result

    async def describe(self, *, job_id: str) -> RestoreJobModel:
        """Get the current state of one restore job.

        Args:
            job_id (str): The identifier of the restore job to describe.

        Returns:
            A :class:`RestoreJobModel` naming the ``backup_id`` restored and
            the ``target_index_name`` it lands in. ``status`` is one of
            ``"Pending"``, ``"Completed"``, ``"Failed"``, or ``"Cancelled"``;
            a restore that is running reports ``"Pending"``.
            ``percent_complete`` and ``completed_at`` are set only once
            ``status`` is ``"Completed"``, so ``percent_complete`` cannot
            report progress.

        Raises:
            :exc:`PineconeValueError`: If *job_id* is empty.
            :exc:`PineconeTypeError`: If *job_id* is not a ``str``.
            :exc:`NotFoundError`: If the server cannot read the job. This does
                not prove the job does not exist; see the warning below.

        Examples:
            .. code-block:: python

                from pinecone import AsyncPinecone

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    job = await pc.restore_jobs.describe(job_id="rj-abc123")
                    print(job.status, job.target_index_name)

            To wait for a restore, poll until ``status`` leaves ``"Pending"``.
            A deadline keeps a stuck job from looping forever:

            .. code-block:: python

                import asyncio
                import time

                async with AsyncPinecone(api_key="your-api-key") as pc:
                    deadline = time.monotonic() + 600
                    job = await pc.restore_jobs.describe(job_id="rj-abc123")
                    while job.status == "Pending" and time.monotonic() < deadline:
                        await asyncio.sleep(5)
                        job = await pc.restore_jobs.describe(job_id="rj-abc123")

                    print(job.status, job.completed_at)

        .. warning::
           A :exc:`NotFoundError` here does not prove the job does not
           exist. The server also answers 404 on a transient failure and for
           a job whose target index has been deleted. Treat a 404 as possibly
           transient unless you know the ID is wrong.
        """
        require_non_empty("job_id", job_id)
        logger.info("Describing restore job %r", job_id)
        response = await self._http.get(f"/restore-jobs/{quote(job_id, safe='')}")
        result = self._adapter.to_restore_job(response.content)
        logger.debug("Described restore job %r", job_id)
        return result
