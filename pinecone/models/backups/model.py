"""Backup and restore response models."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._mixin import StructDictMixin


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
        record_count: Number of records in the backup.
        namespace_count: Number of namespaces in the backup.
        size_bytes: Size of the backup in bytes.
        tags: User-defined key-value tags.
        created_at: Timestamp when the backup was created.
        schema: Metadata schema of the backed-up index, or ``None`` if not returned.
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
    record_count: int | None = None
    namespace_count: int | None = None
    size_bytes: int | None = None
    tags: dict[str, str] | None = None
    created_at: str | None = None
    schema: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for unknown attributes (legacy dict-style delegation)."""
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. backup['backup_id'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'backup_id' in backup``)."""
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this backup model.

        Returns:
            Dictionary with all fields, including optional ones that are ``None``
            (e.g. ``name``, ``description``, ``dimension``,
            ``record_count``, ``namespace_count``, ``size_bytes``, ``tags``,
            ``created_at``). Values are not recursively converted.

        Examples:
            >>> from pinecone.models.backups.model import BackupModel
            >>> backup = BackupModel(
            ...     backup_id="bkp-1",
            ...     source_index_name="my-index",
            ...     source_index_id="idx-abc",
            ...     status="Ready",
            ...     cloud="aws",
            ...     region="us-east-1",
            ...     name="weekly-backup",
            ... )
            >>> d = backup.to_dict()
            >>> d["backup_id"]
            'bkp-1'
            >>> d["name"]
            'weekly-backup'
            >>> d["description"] is None
            True
        """
        return {f: getattr(self, f) for f in self.__struct_fields__}


class RestoreJobModel(Struct, kw_only=True):
    """Response model for a Pinecone restore job.

    Attributes:
        restore_job_id: Unique identifier for the restore job.
        backup_id: Identifier of the backup being restored.
        target_index_name: Name of the index being restored to.
        target_index_id: Unique identifier of the target index.
        status: Current status of the restore job.
        created_at: Timestamp when the restore job was created, or ``None`` if the
            backend has not yet assigned a creation timestamp.
        completed_at: Timestamp when the restore job completed.
        percent_complete: Percentage of the restore job that has completed.
    """

    restore_job_id: str
    backup_id: str
    target_index_name: str
    target_index_id: str
    status: str
    created_at: str | None = None
    completed_at: str | None = None
    percent_complete: float | None = None

    def __getattr__(self, name: str) -> Any:
        """Raise AttributeError for unknown attributes (legacy dict-style delegation)."""
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: str) -> Any:
        """Support bracket access (e.g. job['restore_job_id'])."""
        if key not in self.__struct_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        """Support ``in`` operator (e.g. ``'restore_job_id' in job``)."""
        return key in self.__struct_fields__

    def to_dict(self) -> dict[str, Any]:
        """Return a dict representation of this restore job model.

        Returns:
            Dictionary with all fields, including optional ones that are ``None``
            (``completed_at`` and ``percent_complete``). Values are not
            recursively converted.

        Examples:
            >>> from pinecone.models.backups.model import RestoreJobModel
            >>> job = RestoreJobModel(
            ...     restore_job_id="rj-1",
            ...     backup_id="bkp-1",
            ...     target_index_name="my-index",
            ...     target_index_id="idx-abc",
            ...     status="Running",
            ...     created_at="2024-01-01T00:00:00Z",
            ... )
            >>> d = job.to_dict()
            >>> d["restore_job_id"]
            'rj-1'
            >>> d["completed_at"] is None
            True
        """
        return {f: getattr(self, f) for f in self.__struct_fields__}


class CreateIndexFromBackupResponse(StructDictMixin, Struct, kw_only=True):
    """Response model for creating an index from a backup.

    Attributes:
        restore_job_id: Identifier of the restore job created.
        index_id: Identifier of the new index being created.
    """

    restore_job_id: str
    index_id: str
