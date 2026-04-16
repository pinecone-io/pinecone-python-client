"""Preview index request models (2026-01.alpha API).

These models are the typed boundary for ``PreviewIndexes.create()`` and
``PreviewIndexes.configure()``.  User code passes keyword arguments; the SDK
validates against these models and serialises with msgspec/orjson.
"""

from __future__ import annotations

from typing import Any

from msgspec import Struct

from pinecone.preview.models.read_capacity import PreviewReadCapacity
from pinecone.preview.models.schema import PreviewSchema

__all__ = ["PreviewConfigureIndexRequest", "PreviewCreateIndexRequest"]


class PreviewCreateIndexRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for creating a preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    Attributes:
        schema: Index schema definition (required).
        name: Optional name for the index.
        deployment: Optional deployment configuration.
        read_capacity: Optional read capacity configuration.
        deletion_protection: Optional deletion protection setting.
        tags: Optional key-value tags for the index.
    """

    schema: PreviewSchema
    name: str | None = None
    deployment: dict[str, Any] | None = None
    read_capacity: PreviewReadCapacity | None = None
    deletion_protection: str | None = None
    tags: dict[str, str] | None = None


class PreviewConfigureIndexRequest(Struct, kw_only=True, omit_defaults=True):
    """Request model for configuring an existing preview index.

    .. admonition:: Preview
       :class: warning

       Uses Pinecone API version ``2026-01.alpha``.
       Preview surface is not covered by SemVer — signatures and behavior
       may change in any minor SDK release. Pin your SDK version when
       relying on preview features.

    All fields are optional — only provided fields are updated.

    Attributes:
        schema: Optional updated schema definition.
        deployment: Optional updated deployment configuration.
        read_capacity: Optional updated read capacity configuration.
        deletion_protection: Optional updated deletion protection setting.
        tags: Optional updated key-value tags.
    """

    schema: PreviewSchema | None = None
    deployment: dict[str, Any] | None = None
    read_capacity: PreviewReadCapacity | None = None
    deletion_protection: str | None = None
    tags: dict[str, str] | None = None
