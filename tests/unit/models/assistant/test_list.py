"""Unit tests for ListAssistantsResponse and ListFilesResponse models."""

from __future__ import annotations

from pinecone.models.assistant.file_model import AssistantFileModel
from pinecone.models.assistant.list import ListAssistantsResponse, ListFilesResponse
from pinecone.models.assistant.model import AssistantModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant(name: str = "test-asst") -> AssistantModel:
    return AssistantModel(name=name, status="Ready")


def _make_file(file_id: str = "file-abc") -> AssistantFileModel:
    return AssistantFileModel(id=file_id, name="test.txt", status="Available", size=0)


# ---------------------------------------------------------------------------
# ListAssistantsResponse — next_token alias
# ---------------------------------------------------------------------------


def test_list_assistants_response_next_token_alias_none() -> None:
    """next_token returns None when next is None."""
    resp = ListAssistantsResponse(assistants=[], next=None)
    assert resp.next_token is None
    assert resp.next_token == resp.next


def test_list_assistants_response_next_token_alias_with_value() -> None:
    """next_token returns the same string as next when populated."""
    token = "some-pagination-token"
    resp = ListAssistantsResponse(assistants=[], next=token)
    assert resp.next_token == token
    assert resp.next_token == resp.next


def test_list_assistants_response_next_token_alias_with_assistants() -> None:
    """next_token alias works when assistants list is non-empty."""
    assistants = [_make_assistant("a1"), _make_assistant("a2")]
    resp = ListAssistantsResponse(assistants=assistants, next="tok-2")
    assert resp.next_token == "tok-2"
    assert resp.next_token == resp.next


# ---------------------------------------------------------------------------
# ListFilesResponse — next_token alias
# ---------------------------------------------------------------------------


def test_list_files_response_next_token_alias_none() -> None:
    """next_token returns None when next is None."""
    resp = ListFilesResponse(files=[], next=None)
    assert resp.next_token is None
    assert resp.next_token == resp.next


def test_list_files_response_next_token_alias_with_value() -> None:
    """next_token returns the same string as next when populated."""
    token = "files-pagination-token"
    resp = ListFilesResponse(files=[], next=token)
    assert resp.next_token == token
    assert resp.next_token == resp.next


def test_list_files_response_next_token_alias_with_files() -> None:
    """next_token alias works when files list is non-empty."""
    files = [_make_file("id-1"), _make_file("id-2")]
    resp = ListFilesResponse(files=files, next="tok-files-2")
    assert resp.next_token == "tok-files-2"
    assert resp.next_token == resp.next
