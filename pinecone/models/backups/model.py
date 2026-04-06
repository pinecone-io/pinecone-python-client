"""Backup and restore response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct


class BackupModel(Struct, kw_only=True):
    """Response model for a Pinecone backup.

    Attributes:
        backup_id: Unique identifier for the backup.
        source_index_name: Name of the index that was backed up.
        source_index_id: Unique identifier of the source index.
        status: Current status of the backup.
        cloud: Cloud provider where the backup is stored.
        region: Region where the backup is stored.
        name: User-provided name for the backup.
        description: User-provided description for the backup.
        dimension: Dimensionality of vectors in the backup.
        metric: Distance metric of the backed-up index.
        record_count: Number of records in the backup.
        namespace_count: Number of namespaces in the backup.
        size_bytes: Size of the backup in bytes.
        tags: User-defined key-value tags.
        created_at: Timestamp when the backup was created.
    """

    backup_id: str
    source_index_name: str
    source_index_id: str
    status: str
    cloud: str
    region: str
    name: str | None = None
    description: str | None = None
    dimension: int | None = None
    metric: str | None = None
    record_count: int | None = None
    namespace_count: int | None = None
    size_bytes: int | None = None
    tags: dict[str, str] | None = None
    created_at: str | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. backup['backup_id'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class RestoreJobModel(Struct, kw_only=True):
    """Response model for a Pinecone restore job.

    Attributes:
        restore_job_id: Unique identifier for the restore job.
        backup_id: Identifier of the backup being restored.
        target_index_name: Name of the index being restored to.
        target_index_id: Unique identifier of the target index.
        status: Current status of the restore job.
        created_at: Timestamp when the restore job was created.
        completed_at: Timestamp when the restore job completed.
        percent_complete: Percentage of the restore job that has completed.
    """

    restore_job_id: str
    backup_id: str
    target_index_name: str
    target_index_id: str
    status: str
    created_at: str
    completed_at: str | None = None
    percent_complete: float | None = None

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. job['restore_job_id'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


class CreateIndexFromBackupResponse(Struct, kw_only=True):
    """Response model for creating an index from a backup.

    Attributes:
        restore_job_id: Identifier of the restore job created.
        index_id: Identifier of the new index being created.
    """

    restore_job_id: str
    index_id: str

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. response['restore_job_id'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None
