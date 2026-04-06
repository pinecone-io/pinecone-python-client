"""BackupList and RestoreJobList wrappers for listing responses."""

from __future__ import annotations

from collections.abc import Iterator

from pinecone.models.backups.model import BackupModel, RestoreJobModel


class BackupList:
    """Wrapper around a list of BackupModel with convenience methods."""

    def __init__(self, backups: list[BackupModel]) -> None:
        self._backups = backups

    def __iter__(self) -> Iterator[BackupModel]:
        return iter(self._backups)

    def __len__(self) -> int:
        return len(self._backups)

    def __getitem__(self, index: int) -> BackupModel:
        return self._backups[index]

    def names(self) -> list[str]:
        """Return a list of backup names, falling back to backup_id."""
        return [b.name or b.backup_id for b in self._backups]

    def __repr__(self) -> str:
        return f"BackupList(backups={self._backups!r})"


class RestoreJobList:
    """Wrapper around a list of RestoreJobModel with convenience methods."""

    def __init__(self, restore_jobs: list[RestoreJobModel]) -> None:
        self._restore_jobs = restore_jobs

    def __iter__(self) -> Iterator[RestoreJobModel]:
        return iter(self._restore_jobs)

    def __len__(self) -> int:
        return len(self._restore_jobs)

    def __getitem__(self, index: int) -> RestoreJobModel:
        return self._restore_jobs[index]

    def __repr__(self) -> str:
        return f"RestoreJobList(restore_jobs={self._restore_jobs!r})"
