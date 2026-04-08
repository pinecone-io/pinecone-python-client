"""BackupList and RestoreJobList wrappers for listing responses."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from pinecone.models.backups.model import BackupModel, RestoreJobModel

if TYPE_CHECKING:
    from pinecone.models.vectors.responses import Pagination


class BackupList:
    """Wrapper around a list of BackupModel with convenience methods."""

    def __init__(
        self,
        backups: list[BackupModel],
        *,
        pagination: Pagination | None = None,
    ) -> None:
        """Initialize a BackupList.

        Args:
            backups: List of :class:`BackupModel` instances representing
                index backups.
            pagination: Optional :class:`Pagination` token for fetching
                additional pages of results.
        """
        self._backups = backups
        self.pagination = pagination

    def __iter__(self) -> Iterator[BackupModel]:
        return iter(self._backups)

    def __len__(self) -> int:
        return len(self._backups)

    def __getitem__(self, index: int) -> BackupModel:
        return self._backups[index]

    def names(self) -> list[str]:
        """Return a list of backup names, falling back to backup_id.

        If a backup has no ``name`` set, its ``backup_id`` is used instead.

        Returns:
            list[str]: Backup names (or IDs when names are absent).

        Examples:
            List names of all backups for an index:

            >>> from pinecone import Pinecone
            >>> pc = Pinecone(api_key="your-api-key")
            >>> backups = pc.list_backups(index_name="movie-recommendations")
            >>> backups.names()
            ['daily-2025-01-01', 'weekly-2024-12-29']
        """
        return [b.name or b.backup_id for b in self._backups]

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<name={(b.name or b.backup_id)!r}, status={b.status!r}, "
            f"source={b.source_index_name!r}>"
            for b in self._backups
        )
        return f"BackupList([{summaries}])"


class RestoreJobList:
    """Wrapper around a list of RestoreJobModel with convenience methods."""

    def __init__(
        self,
        restore_jobs: list[RestoreJobModel],
        *,
        pagination: Pagination | None = None,
    ) -> None:
        """Initialize a RestoreJobList.

        Args:
            restore_jobs: List of :class:`RestoreJobModel` instances
                representing restore operations.
            pagination: Optional :class:`Pagination` token for fetching
                additional pages of results.
        """
        self._restore_jobs = restore_jobs
        self.pagination = pagination

    def __iter__(self) -> Iterator[RestoreJobModel]:
        return iter(self._restore_jobs)

    def __len__(self) -> int:
        return len(self._restore_jobs)

    def __getitem__(self, index: int) -> RestoreJobModel:
        return self._restore_jobs[index]

    def __repr__(self) -> str:
        summaries = ", ".join(
            f"<id={r.restore_job_id!r}, status={r.status!r}, target={r.target_index_name!r}>"
            for r in self._restore_jobs
        )
        return f"RestoreJobList([{summaries}])"
