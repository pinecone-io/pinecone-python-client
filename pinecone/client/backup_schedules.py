"""BackupSchedules namespace: automatic, time-based index backups.

A backup schedule backs up one index on a fixed cadence (``daily``,
``weekly`` or ``monthly``). The server picks the run time and reports it
through
:attr:`~pinecone.models.backups.schedules.BackupScheduleModel.next_scheduled_run`;
there is no cron expression or caller-chosen timezone. The SDK always sends
``"time-based"`` as the schedule type, so :meth:`create` takes no ``type``
argument.

Each of the two listings comes in two shapes. :meth:`BackupSchedules.list` and
:meth:`BackupSchedules.history` return one page plus its pagination token,
matching :meth:`~pinecone.client.backups.Backups.list`.
:meth:`BackupSchedules.iter_schedules` and
:meth:`BackupSchedules.iter_history` return a
:class:`~pinecone.models.pagination.Paginator` that walks every page, matching
:meth:`~pinecone.client.indexes.Indexes.list_backups`.
Prefer the iterators unless you are driving the token yourself; see
`Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pinecone._internal.adapters.backup_schedules_adapter import BackupSchedulesAdapter
from pinecone._internal.backups_helpers import (
    backup_schedule_list_params,
    schedule_request_validation,
    scheduled_backups_plan_gate,
)
from pinecone._internal.validation import require_non_empty, require_positive
from pinecone.models.backups.list import BackupScheduleHistoryList, BackupScheduleList
from pinecone.models.backups.schedules import (
    BackupScheduleHistoryItem,
    BackupScheduleModel,
    CreateBackupScheduleRequest,
    UpdateBackupScheduleRequest,
)
from pinecone.models.pagination import Page, Paginator

if TYPE_CHECKING:
    from pinecone._internal.http_client import HTTPClient

logger = logging.getLogger(__name__)


class BackupSchedules:
    """Recurring, time-based backups of a single index.

    A schedule snapshots its index on a fixed cadence and retains each backup
    for a set number of days, so you do not have to call
    :meth:`~pinecone.client.backups.Backups.create` on a timer of your own.
    Reached as ``pc.backup_schedules``; not constructed directly.

    At most one schedule per index can be enabled at a time. The backups a
    schedule produces are ordinary backups: read one with
    :meth:`~pinecone.client.backups.Backups.describe`, or list a schedule's
    own runs with :meth:`history`.

    Note:
        Backups are a plan entitlement. A project without it gets
        :exc:`ForbiddenError`, not :exc:`NotFoundError`, even for a schedule
        that does not exist. On-demand backups need the same entitlement.

    Examples:

        Back up an index every day and keep each backup for 90 days:

        >>> from pinecone import Pinecone
        >>> pc = Pinecone(api_key="your-api-key")
        >>> schedule = pc.backup_schedules.create(
        ...     index_name="product-search",
        ...     name="compliance-snapshots",
        ...     frequency="daily",
        ...     retention_days=90,
        ... )
        >>> print(schedule.schedule_id, schedule.next_scheduled_run)
        e88f7273-42aa-47e9-af73-593827136867 2026-04-03 06:00:00+00:00

    .. seealso::
       `Error handling <https://sdk.pinecone.io/python/guides/error-handling.html>`__
       for the exceptions these methods raise and which ones are worth
       retrying.
    """

    def __init__(self, http: HTTPClient) -> None:
        self._http = http
        self._adapter = BackupSchedulesAdapter()

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return "BackupSchedules()"

    def create(
        self,
        *,
        index_name: str,
        name: str,
        frequency: str,
        retention_days: int,
    ) -> BackupScheduleModel:
        """Create a time-based backup schedule for an index.

        The schedule is created enabled, with its first run already planned.
        For a single backup taken now, use
        :meth:`~pinecone.client.backups.Backups.create` instead.

        Args:
            index_name (str): Name of the serverless or BYOC index to back
                up, for example ``"product-search"``.
            name (str): Name for the schedule, for example
                ``"compliance-snapshots"``. Each backup it produces is named
                ``"{name}-{run timestamp}"``, such as
                ``"compliance-snapshots-20260403T060000Z"``.
            frequency (str): How often to back up the index: ``"daily"``,
                ``"weekly"``, or ``"monthly"``.
            retention_days (int): Days to keep each backup this schedule
                produces. Must be at least 1; the server enforces the
                maximum.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            for the new schedule, with ``next_scheduled_run`` set. The
            retention window comes back as ``retention_expire_after_days``.

        Raises:
            :exc:`PineconeValueError`: If *index_name* or *name* is empty, if
                *frequency* is not a supported cadence, or if *retention_days*
                is less than 1.
            :exc:`PineconeTypeError`: If *index_name* or *name* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the index does not exist.
            :exc:`ConflictError`: If the index already has an enabled
                schedule. Disable or delete that one first.
            :exc:`ApiError`: If *index_name* names a pod-based index, which
                cannot be scheduled.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> schedule = pc.backup_schedules.create(
            ...     index_name="product-search",
            ...     name="compliance-snapshots",
            ...     frequency="daily",
            ...     retention_days=90,
            ... )
            >>> print(schedule.schedule_id, schedule.next_scheduled_run)
            e88f7273-42aa-47e9-af73-593827136867 2026-04-03 06:00:00+00:00
        """
        require_non_empty("index_name", index_name)
        require_non_empty("name", name)
        with schedule_request_validation():
            request = CreateBackupScheduleRequest(
                name=name, frequency=frequency, retention_days=retention_days
            )

        logger.info("Creating backup schedule %r for index %r", name, index_name)
        with scheduled_backups_plan_gate():
            response = self._http.post(
                f"/indexes/{quote(index_name, safe='')}/backup-schedules", json=request.to_wire()
            )
        result = self._adapter.to_schedule(response.content)
        logger.debug("Created backup schedule %r", result.schedule_id)
        return result

    def list(
        self,
        *,
        index_name: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> BackupScheduleList:
        """List one page of an index's backup schedules.

        Disabled schedules are included, so an index can list several
        schedules even though at most one is enabled. To walk every page, use
        :meth:`iter_schedules`; see
        `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__.

        Args:
            index_name (str): Name of the index whose schedules to list.
            limit (int | None): Maximum results per page, for example
                ``20``. ``None`` uses the server's page size. Ignored when
                *pagination_token* is set, because the token carries its own
                page size.
            pagination_token (str | None): The ``pagination.next`` value from
                the previous page, or ``None`` for the first page.

        Returns:
            A :class:`~pinecone.models.backups.list.BackupScheduleList` of
            :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            that supports iteration, ``len()``, and index access. Its
            ``pagination`` is ``None`` on the last page.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty or *limit* is
                zero or negative.
            :exc:`PineconeTypeError`: If *index_name* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the index does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> schedules = pc.backup_schedules.list(index_name="product-search")
            >>> schedules.names()
            ['compliance-snapshots']
            >>> [s.schedule_id for s in schedules.enabled_schedules()]
            ['e88f7273-42aa-47e9-af73-593827136867']

            ``names()`` and ``enabled_schedules()`` read only this page, so
            check ``schedules.pagination`` before concluding that an index
            has no enabled schedule.
        """
        require_non_empty("index_name", index_name)
        if limit is not None:
            require_positive("limit", limit)
        params: dict[str, Any] = backup_schedule_list_params(
            limit=limit, pagination_token=pagination_token
        )

        logger.info("Listing backup schedules for index %r", index_name)
        with scheduled_backups_plan_gate():
            response = self._http.get(
                f"/indexes/{quote(index_name, safe='')}/backup-schedules", params=params
            )
        result = self._adapter.to_schedule_list(response.content)
        logger.debug("Listed %d backup schedules", len(result))
        return result

    def iter_schedules(
        self,
        *,
        index_name: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[BackupScheduleModel]:
        """Iterate every backup schedule on an index, fetching pages on demand.

        The auto-paginating twin of :meth:`list`.

        Args:
            index_name (str): Name of the index whose schedules to iterate.
            limit (int | None): Maximum number of schedules to yield across
                all pages. Must be positive. ``None`` yields all of them.
            pagination_token (str | None): Token to resume from a previous
                call. *limit* still caps the total yield.

        Returns:
            A :class:`~pinecone.models.pagination.Paginator` over
            :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            instances.

        Raises:
            :exc:`PineconeValueError`: If *index_name* is empty or *limit* is
                zero or negative. Raised as soon as you call this method,
                before the first page is fetched.
            :exc:`PineconeTypeError`: If *index_name* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Raised while iterating, when a page is
                fetched.
            :exc:`NotFoundError`: If the index does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> for s in pc.backup_schedules.iter_schedules(index_name="product-search"):
            ...     print(s.schedule_id, s.frequency, s.enabled)
            e88f7273-42aa-47e9-af73-593827136867 daily True

        """
        require_non_empty("index_name", index_name)
        if limit is not None:
            require_positive("limit", limit)

        def fetch_page(token: str | None) -> Page[BackupScheduleModel]:
            params = backup_schedule_list_params(limit=limit, pagination_token=token)
            logger.info("Listing backup schedules for index %r", index_name)
            with scheduled_backups_plan_gate():
                response = self._http.get(
                    f"/indexes/{quote(index_name, safe='')}/backup-schedules", params=params
                )
            result = self._adapter.to_schedule_list(response.content)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=list(result), pagination_token=next_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)

    def describe(self, *, schedule_id: str) -> BackupScheduleModel:
        """Get the current configuration of one backup schedule.

        Args:
            schedule_id (str): ID of the schedule to describe, from
                :meth:`create` or :meth:`list`. Not the index name.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            carrying the schedule's ``frequency``, its ``enabled`` flag, its
            ``retention_expire_after_days`` window, and
            ``next_scheduled_run``, which is ``None`` while the schedule is
            disabled.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`PineconeTypeError`: If *schedule_id* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> schedule = pc.backup_schedules.describe(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867"
            ... )
            >>> print(schedule.enabled, schedule.next_scheduled_run)
            True 2026-04-03 06:00:00+00:00
        """
        require_non_empty("schedule_id", schedule_id)
        logger.info("Describing backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = self._http.get(f"/backup-schedules/{quote(schedule_id, safe='')}")
        result = self._adapter.to_schedule(response.content)
        logger.debug("Described backup schedule %r", schedule_id)
        return result

    def get(self, *, schedule_id: str) -> BackupScheduleModel:
        """Get the current configuration of one backup schedule (alias for :meth:`describe`).

        Args:
            schedule_id (str): ID of the schedule to describe, from
                :meth:`create` or :meth:`list`. Not the index name.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            carrying the schedule's ``frequency``, its ``enabled`` flag, its
            ``retention_expire_after_days`` window, and
            ``next_scheduled_run``, which is ``None`` while the schedule is
            disabled.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`PineconeTypeError`: If *schedule_id* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> schedule = pc.backup_schedules.get(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867"
            ... )
            >>> schedule.frequency
            'daily'
        """
        return self.describe(schedule_id=schedule_id)

    def update(
        self,
        *,
        schedule_id: str,
        frequency: str | None = None,
        retention_days: int | None = None,
        enabled: bool | None = None,
    ) -> BackupScheduleModel:
        """Update a backup schedule's cadence, retention, or enabled state.

        Only the arguments you pass change; the rest keep their current
        values. A schedule's name and index cannot be changed.

        Args:
            schedule_id (str): ID of the schedule to update.
            frequency (str | None): New cadence: ``"daily"``, ``"weekly"``,
                or ``"monthly"``. ``None`` leaves it unchanged.
            retention_days (int | None): New retention window in days, at
                least 1. Also applies to unexpired backups this schedule
                already produced. ``None`` leaves it unchanged.
            enabled (bool | None): ``False`` pauses the schedule and clears
                ``next_scheduled_run``; ``True`` resumes a paused one (see
                the warning below). ``None`` leaves it unchanged.

        Returns:
            A :class:`~pinecone.models.backups.schedules.BackupScheduleModel`
            with the updated configuration. After ``enabled=False`` its
            ``next_scheduled_run`` is ``None``.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty, if
                *frequency* is set to an unsupported cadence, or if
                *retention_days* is set to less than 1.
            :exc:`PineconeTypeError`: If *schedule_id* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.
            :exc:`ConflictError`: If ``enabled=True`` and another schedule on
                the same index is already enabled.

        Examples:

            Move a schedule to a weekly cadence with a shorter retention
            window. Its name and enabled state stay as they were:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> updated = pc.backup_schedules.update(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867",
            ...     frequency="weekly",
            ...     retention_days=30,
            ... )
            >>> print(updated.frequency, updated.retention_expire_after_days)
            weekly 30
            >>> print(updated.name, updated.enabled)
            compliance-snapshots True

            Pause the schedule, keeping the rest of its configuration:

            >>> paused = pc.backup_schedules.update(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867",
            ...     enabled=False,
            ... )
            >>> print(paused.frequency, paused.next_scheduled_run)
            daily None

        .. warning::
           Re-enabling a paused schedule plans its next run from the time of
           the update, not from its old slot, so a pause shifts the cadence.
           Re-enabling raises :exc:`ConflictError` if another schedule on the
           same index is already enabled.
        """
        require_non_empty("schedule_id", schedule_id)
        with schedule_request_validation():
            request = UpdateBackupScheduleRequest(
                frequency=frequency, retention_days=retention_days, enabled=enabled
            )

        logger.info("Updating backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = self._http.patch(
                f"/backup-schedules/{quote(schedule_id, safe='')}", json=request.to_wire()
            )
        result = self._adapter.to_schedule(response.content)
        logger.debug("Updated backup schedule %r", schedule_id)
        return result

    def delete(self, *, schedule_id: str) -> None:
        """Delete a backup schedule.

        Deleting a schedule stops its future runs. Backups it already produced
        are not deleted.

        Args:
            schedule_id (str): ID of the schedule to delete.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty.
            :exc:`PineconeTypeError`: If *schedule_id* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> pc.backup_schedules.delete(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867"
            ... )

        .. important::
           If a delete's response is lost and the call is retried, the retry
           raises :exc:`NotFoundError` because the first attempt already
           deleted the schedule. Treat :exc:`NotFoundError` from a delete as
           success.
        """
        require_non_empty("schedule_id", schedule_id)
        logger.info("Deleting backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            self._http.delete(f"/backup-schedules/{quote(schedule_id, safe='')}")
        logger.debug("Deleted backup schedule %r", schedule_id)

    def history(
        self,
        *,
        schedule_id: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> BackupScheduleHistoryList:
        """List one page of the backups a schedule has produced.

        Each row is a backup. The schedule's next run is listed as soon as it
        is planned, before it starts, so check each row's ``status``. To walk
        every page, use :meth:`iter_history`; see
        `Pagination <https://sdk.pinecone.io/python/guides/pagination.html>`__.

        Args:
            schedule_id (str): ID of the schedule whose history to list.
            limit (int | None): Maximum results per page, for example
                ``20``. ``None`` uses the server's page size. Ignored when
                *pagination_token* is set, because the token carries its own
                page size.
            pagination_token (str | None): The ``pagination.next`` value from
                the previous page, or ``None`` for the first page.

        Returns:
            A :class:`~pinecone.models.backups.list.BackupScheduleHistoryList`
            of :class:`~pinecone.models.backups.schedules.BackupScheduleHistoryItem`
            that supports iteration, ``len()``, and index access. Its
            ``pagination`` is ``None`` on the last page.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty or *limit*
                is zero or negative.
            :exc:`PineconeTypeError`: If *schedule_id* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:

            Walk the history a page at a time and print the finished backups:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> pagination_token = None
            >>> while True:
            ...     runs = pc.backup_schedules.history(
            ...         schedule_id="e88f7273-42aa-47e9-af73-593827136867",
            ...         pagination_token=pagination_token,
            ...     )
            ...     for run in runs:
            ...         if run.status == "Ready":
            ...             print(run.name, run.size_bytes)
            ...     pagination_token = runs.pagination.next if runs.pagination else None
            ...     if pagination_token is None:
            ...         break
            compliance-snapshots-20260401T060000Z 4194304
        """
        require_non_empty("schedule_id", schedule_id)
        if limit is not None:
            require_positive("limit", limit)
        params: dict[str, Any] = backup_schedule_list_params(
            limit=limit, pagination_token=pagination_token
        )

        logger.info("Listing history for backup schedule %r", schedule_id)
        with scheduled_backups_plan_gate():
            response = self._http.get(
                f"/backup-schedules/{quote(schedule_id, safe='')}/history", params=params
            )
        result = self._adapter.to_history_list(response.content)
        logger.debug("Listed %d backup schedule history rows", len(result))
        return result

    def iter_history(
        self,
        *,
        schedule_id: str,
        limit: int | None = None,
        pagination_token: str | None = None,
    ) -> Paginator[BackupScheduleHistoryItem]:
        """Iterate every backup a schedule has produced, fetching pages on demand.

        The auto-paginating twin of :meth:`history`.

        Args:
            schedule_id (str): ID of the schedule whose history to iterate.
            limit (int | None): Maximum number of backups to yield across all
                pages. Must be positive. ``None`` yields all of them.
            pagination_token (str | None): Token to resume from a previous
                call. *limit* still caps the total yield.

        Returns:
            A :class:`~pinecone.models.pagination.Paginator` over
            :class:`~pinecone.models.backups.schedules.BackupScheduleHistoryItem`
            instances.

        Raises:
            :exc:`PineconeValueError`: If *schedule_id* is empty or *limit*
                is zero or negative. Raised as soon as you call this method,
                before the first page is fetched.
            :exc:`PineconeTypeError`: If *schedule_id* is not a ``str``.
            :exc:`ForbiddenError`: If the project's plan does not include
                scheduled backups. Raised while iterating, when a page is
                fetched.
            :exc:`NotFoundError`: If the schedule does not exist.

        Examples:
            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> for run in pc.backup_schedules.iter_history(
            ...     schedule_id="e88f7273-42aa-47e9-af73-593827136867"
            ... ):
            ...     if run.status == "Ready":
            ...         print(run.name, run.size_bytes)
            compliance-snapshots-20260401T060000Z 4194304

        """
        require_non_empty("schedule_id", schedule_id)
        if limit is not None:
            require_positive("limit", limit)

        def fetch_page(token: str | None) -> Page[BackupScheduleHistoryItem]:
            params = backup_schedule_list_params(limit=limit, pagination_token=token)
            logger.info("Listing history for backup schedule %r", schedule_id)
            with scheduled_backups_plan_gate():
                response = self._http.get(
                    f"/backup-schedules/{quote(schedule_id, safe='')}/history", params=params
                )
            result = self._adapter.to_history_list(response.content)
            next_token = result.pagination.next if result.pagination is not None else None
            return Page(items=list(result), pagination_token=next_token)

        return Paginator(fetch_page=fetch_page, initial_token=pagination_token, limit=limit)
