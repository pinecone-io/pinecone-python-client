"""Adapter for Assistants API responses."""

from __future__ import annotations

from pinecone._internal.adapters._decode import decode_response
from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.model import AssistantModel


class AssistantsAdapter:
    """Transforms raw API JSON into AssistantModel / ListAssistantsResponse instances."""

    @staticmethod
    def to_assistant(data: bytes) -> AssistantModel:
        """Decode raw JSON bytes into an AssistantModel."""
        return decode_response(data, AssistantModel)

    @staticmethod
    def to_assistant_list(data: bytes) -> ListAssistantsResponse:
        """Decode raw JSON bytes into a ListAssistantsResponse."""
        return decode_response(data, ListAssistantsResponse)

    @staticmethod
    def to_file(data: bytes) -> AssistantFileModel:
        """Decode raw JSON bytes into an AssistantFileModel."""
        return decode_response(data, AssistantFileModel)

    @staticmethod
    def to_file_list(data: bytes) -> ListFilesResponse:
        """Decode raw JSON bytes into a ListFilesResponse."""
        return decode_response(data, ListFilesResponse)
