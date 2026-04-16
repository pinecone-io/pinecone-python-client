"""Backwards-compatibility shim for :mod:`pinecone.models.assistant.list`.

Re-exports :class:`ListFilesResponse` that used to live at
:mod:`pinecone_plugins.assistant.models.list_files_response` before the
``python-sdk2`` rewrite. Preserved to keep pre-rewrite callers working.
New code should import from the canonical module.

:meta private:
"""

from dataclasses import dataclass
from typing import Any, Optional

from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.file_model import FileModel


@dataclass
class ListFilesResponse(BaseDataclass):
    """Paginated list of assistant files."""

    files: list[FileModel]
    next_token: Optional[str]

    @classmethod
    def from_openapi(cls, resp: Any) -> "ListFilesResponse":
        files = [FileModel.from_openapi(f) for f in resp.files]
        next_token: Optional[str] = None
        pagination = getattr(resp, "pagination", None)
        if pagination is not None:
            next_token = getattr(pagination, "next", None)
        return cls(files=files, next_token=next_token)


__all__ = ["ListFilesResponse"]
