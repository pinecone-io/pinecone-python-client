"""Pagination response models for assistant list operations."""

from __future__ import annotations

from msgspec import Struct

from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.model import AssistantModel


class ListAssistantsResponse(Struct, kw_only=True):
    """Paginated response for listing assistants.

    Attributes:
        assistants: The assistants returned in this page.
        next: Token for fetching the next page of results, or ``None``
            when no more pages exist.
    """

    assistants: list[AssistantModel]
    next: str | None = None


class ListFilesResponse(Struct, kw_only=True):
    """Paginated response for listing assistant files.

    Attributes:
        files: The files returned in this page.
        next: Token for fetching the next page of results, or ``None``
            when no more pages exist.
    """

    files: list[AssistantFileModel]
    next: str | None = None
