"""Preview backup response models (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import render_table

__all__ = ["PreviewBackupModel", "PreviewCreateBackupRequest"]


class PreviewBackupModel(Struct, kw_only=True, omit_defaults=True):
    """Backup metadata returned by preview control-plane operations.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        backup_id: Unique identifier for the backup.
        source_index_id: ID of the source index.
        source_index_name: Name of the index from which the backup was taken.
        status: Current status (``"Initializing"``, ``"Ready"``, or ``"InitializationFailed"``).
        cloud: Cloud provider where the backup is stored.
        region: Cloud region where the backup is stored.
        created_at: ISO 8601 timestamp when the backup was created.
        name: Optional user-defined name for the backup.
        description: Optional description providing context for the backup.
        tags: Custom user tags added to the backup.
        dimension: Vector dimensionality (``None`` for sparse indexes).
        schema: Metadata schema configuration.
        record_count: Total number of records in the backup.
        namespace_count: Number of namespaces in the backup.
        size_bytes: Size of the backup in bytes.
    """

    backup_id: str
    source_index_id: str
    source_index_name: str
    status: str
    cloud: str
    region: str
    created_at: str
    name: str | None = None
    description: str | None = None
    tags: dict[str, Any] | None = None
    dimension: int | None = None
    schema: dict[str, Any] | None = None
    record_count: int | None = None
    namespace_count: int | None = None
    size_bytes: int | None = None

    def __repr__(self) -> str:
        return (
            f"PreviewBackupModel(backup_id={self.backup_id!r}, "
            f"status={self.status!r}, "
            f"source_index_name={self.source_index_name!r}, "
            f"created_at={self.created_at!r})"
        )

    def __dir__(self) -> list[str]:
        attrs = set(super().__dir__())
        public = {name for name in attrs if not name.startswith("_")}
        return sorted(public)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text("PreviewBackupModel(...)")
            return

        p.text("PreviewBackupModel(")
        with p.group(2, "", ")"):
            p.breakable()
            p.text(f"backup_id={self.backup_id!r},")
            p.breakable()
            p.text(f"source_index_name={self.source_index_name!r},")
            p.breakable()
            p.text(f"source_index_id={self.source_index_id!r},")
            p.breakable()
            p.text(f"status={self.status!r},")
            p.breakable()
            p.text(f"cloud={self.cloud!r},")
            p.breakable()
            p.text(f"region={self.region!r},")
            p.breakable()
            p.text(f"created_at={self.created_at!r}")

            if self.name is not None:
                p.breakable()
                p.text(f"name={self.name!r}")
            if self.description is not None:
                p.breakable()
                p.text(f"description={self.description!r}")
            if self.dimension is not None:
                p.breakable()
                p.text(f"dimension={self.dimension}")
            if self.record_count is not None:
                p.breakable()
                p.text(f"record_count={self.record_count}")
            if self.namespace_count is not None:
                p.breakable()
                p.text(f"namespace_count={self.namespace_count}")
            if self.size_bytes is not None:
                p.breakable()
                p.text(f"size_bytes={self.size_bytes}")
            if self.tags:
                p.breakable()
                p.text(f"tags={self.tags!r}")
            if self.schema:
                p.breakable()
                p.text(f"schema={self.schema!r}")

    def _repr_html_(self) -> str:
        rows: list[tuple[str, str | int]] = [
            ("Backup ID:", self.backup_id),
            ("Source Index:", self.source_index_name),
            ("Source Index ID:", self.source_index_id),
            ("Status:", self.status),
            ("Cloud:", self.cloud),
            ("Region:", self.region),
            ("Created:", self.created_at),
        ]

        if self.name is not None:
            rows.append(("Name:", self.name))
        if self.description is not None:
            rows.append(("Description:", self.description))
        if self.dimension is not None:
            rows.append(("Dimension:", self.dimension))
        if self.record_count is not None:
            rows.append(("Records:", self.record_count))
        if self.namespace_count is not None:
            rows.append(("Namespaces:", self.namespace_count))
        if self.size_bytes is not None:
            rows.append(("Size:", f"{self.size_bytes} bytes"))
        if self.tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in self.tags.items())
            rows.append(("Tags:", tags_str))

        return render_table("PreviewBackupModel", rows)


class PreviewCreateBackupRequest(Struct, kw_only=True, omit_defaults=True):
    """Request body for creating a preview backup.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        name: Optional user-defined name for the backup.
        description: Optional description providing context for the backup.
    """

    name: str | None = None
    description: str | None = None
