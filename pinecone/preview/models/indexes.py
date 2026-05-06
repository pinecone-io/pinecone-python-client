"""Preview index response model (2026-01.alpha API)."""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.models._display import render_table
from pinecone.preview.models.deployment import PreviewDeployment
from pinecone.preview.models.read_capacity import PreviewReadCapacity
from pinecone.preview.models.schema import PreviewSchema
from pinecone.preview.models.status import PreviewIndexStatus

__all__ = ["PreviewIndexModel"]


class PreviewIndexModel(Struct, kw_only=True):
    """Index description returned by preview control-plane operations.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        name: Index name.
        host: Data-plane host for this index, or ``None`` if the index is still
            initializing and has not yet been assigned a host.
        status: Current operational status.
        schema: Field-level schema definition.
        deployment: Deployment configuration (managed, pod, or BYOC).
        deletion_protection: Whether deletion protection is enabled
            (``"enabled"`` or ``"disabled"``).
        read_capacity: Read capacity configuration, or ``None`` for the
            default on-demand mode.
        tags: User-defined key-value tags, or ``None``.
    """

    name: str
    status: PreviewIndexStatus
    schema: PreviewSchema
    deployment: PreviewDeployment
    deletion_protection: str
    host: str | None = None
    read_capacity: PreviewReadCapacity | None = None
    tags: dict[str, str] | None = None

    def __repr__(self) -> str:
        dep_name = type(self.deployment).__name__.replace("Deployment", "")
        parts = [
            f"name={self.name!r}",
            f"status={self.status.state!r}",
            f"host={self.host!r}",
            f"deployment={dep_name!r}",
            f"deletion_protection={self.deletion_protection!r}",
        ]
        if self.schema.fields:
            parts.append(f"schema_fields={len(self.schema.fields)}")
        if self.tags:
            parts.append(f"tags={len(self.tags)} items")
        return f"PreviewIndexModel({', '.join(parts)})"

    def __dir__(self) -> list[str]:
        attrs = set(super().__dir__())
        public = {name for name in attrs if not name.startswith("_")}
        return sorted(public)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Pretty-printer support for IPython."""
        if cycle:
            p.text("PreviewIndexModel(...)")
            return
        p.text("PreviewIndexModel(")
        with p.group(2, "", ")"):
            p.breakable()
            p.text(f"name={self.name!r},")
            p.breakable()
            p.text(f"status={self.status.state!r},")
            p.breakable()
            p.text(f"host={self.host!r},")
            p.breakable()
            p.text(f"deployment={self.deployment!r},")
            p.breakable()
            p.text(f"deletion_protection={self.deletion_protection!r},")
            p.breakable()
            p.text(f"schema=Schema(fields={len(self.schema.fields)} fields),")
            if self.read_capacity is not None:
                p.breakable()
                p.text(f"read_capacity={self.read_capacity!r},")
            if self.tags:
                p.breakable()
                p.text(f"tags={self.tags!r},")

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        dep_name = type(self.deployment).__name__.replace("Deployment", "")
        dep_detail = ""
        if hasattr(self.deployment, "cloud") and hasattr(self.deployment, "region"):
            cloud = getattr(self.deployment, "cloud", "")
            region = getattr(self.deployment, "region", "")
            dep_detail = f" ({cloud}/{region})"
        elif hasattr(self.deployment, "environment"):
            dep_detail = f" ({getattr(self.deployment, 'environment', '')})"

        rows: list[tuple[str, str | int]] = [
            ("Name:", self.name),
            ("Status:", self.status.state),
            ("Ready:", "Yes" if self.status.ready else "No"),
            ("Deployment:", f"{dep_name}{dep_detail}"),
            ("Host:", self.host if self.host is not None else "not yet assigned"),
            ("Deletion Protection:", self.deletion_protection),
            ("Schema fields:", len(self.schema.fields)),
        ]
        if self.read_capacity is not None:
            rows.append(
                ("Read capacity:", getattr(self.read_capacity, "mode", str(self.read_capacity)))
            )
        if self.tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in self.tags.items())
            rows.append(("Tags:", tags_str))
        return render_table("PreviewIndexModel", rows)
