"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.file_model`.

Re-exports :class:`FileModel` that used to live at
:mod:`pinecone_plugins.assistant.models.file_model` before the ``python-sdk2``
rewrite. Preserved to keep pre-rewrite callers working.
New code should import :class:`~pinecone.models.assistant.AssistantFileModel`
from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass


@dataclass
class FileModel(BaseDataclass):
    """A file associated with an assistant."""

    name: str
    id: str
    metadata: dict[str, Any]
    created_on: str
    updated_on: str
    status: str
    signed_url: str | None
    size: int
    multimodal: bool
    crc32c_hash: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileModel":
        return cls(
            name=d.get("name", ""),
            id=d.get("id", ""),
            metadata=d.get("metadata") or {},
            created_on=d.get("created_on", ""),
            updated_on=d.get("updated_on", ""),
            status=d.get("status", ""),
            signed_url=d.get("signed_url"),
            size=d.get("size", 0),
            multimodal=d.get("multimodal", False),
            crc32c_hash=d.get("crc32c_hash"),
        )

    @classmethod
    def from_openapi(cls, file_model: Any) -> "FileModel":
        return cls(
            name=file_model.name,
            id=file_model.id,
            metadata=file_model.metadata or {},
            created_on=file_model.created_on,
            updated_on=file_model.updated_on,
            status=file_model.status,
            signed_url=getattr(file_model, "signed_url", None),
            size=file_model.size,
            multimodal=file_model.multimodal,
            crc32c_hash=getattr(file_model, "crc32c_hash", None),
        )


__all__ = ["FileModel"]
