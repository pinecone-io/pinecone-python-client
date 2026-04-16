"""Unit tests for AssistantModel.list_files and list_files_paginated legacy shims (BC-0019)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.models.assistant.list import ListFilesResponse
from pinecone.models.assistant.model import AssistantModel


@pytest.fixture
def mock_assistants() -> MagicMock:
    """Mock Assistants namespace with list_files and list_files_page stubbed."""
    ns: MagicMock = MagicMock()
    ns.list_files.return_value = iter([])
    ns.list_files_page.return_value = ListFilesResponse(files=[])
    return ns


@pytest.fixture
def mock_assistant_model(mock_assistants: MagicMock) -> AssistantModel:
    """An AssistantModel instance with a back-reference to mock_assistants."""
    model = AssistantModel(name="test-assistant", status="Ready")
    model.__dict__["_assistants"] = mock_assistants
    return model


@pytest.fixture
def detached_assistant_model() -> AssistantModel:
    """An AssistantModel instance without a client back-reference."""
    return AssistantModel(name="detached-assistant", status="Ready")


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_assistant_model_list_files_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files delegates to ns.list_files with assistant_name=self.name."""
    mock_assistant_model.list_files()
    mock_assistants.list_files.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        filter=None,
    )


def test_assistant_model_list_files_legacy_with_filter(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files passes filter through to ns.list_files."""
    mock_assistant_model.list_files(filter={"status": "Available"})
    mock_assistants.list_files.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        filter={"status": "Available"},
    )


def test_list_files_returns_list(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files returns a list regardless of the paginator return value."""
    result = mock_assistant_model.list_files()
    assert isinstance(result, list)


def test_assistant_model_list_files_without_client_ref_raises(
    detached_assistant_model: AssistantModel,
) -> None:
    """list_files raises RuntimeError when no client back-reference is set."""
    with pytest.raises(RuntimeError, match="no client reference"):
        detached_assistant_model.list_files()


# ---------------------------------------------------------------------------
# list_files_paginated
# ---------------------------------------------------------------------------


def test_assistant_model_list_files_paginated_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files_paginated delegates to ns.list_files_page with assistant_name=self.name."""
    mock_assistant_model.list_files_paginated()
    mock_assistants.list_files_page.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        filter=None,
        pagination_token=None,
    )


def test_assistant_model_list_files_paginated_with_pagination_token(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files_paginated passes pagination_token through to ns.list_files_page."""
    mock_assistant_model.list_files_paginated(pagination_token="tok-abc")
    mock_assistants.list_files_page.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        filter=None,
        pagination_token="tok-abc",
    )


def test_assistant_model_list_files_paginated_with_filter(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files_paginated passes filter through to ns.list_files_page."""
    mock_assistant_model.list_files_paginated(filter={"status": "Available"})
    mock_assistants.list_files_page.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        filter={"status": "Available"},
        pagination_token=None,
    )


def test_list_files_paginated_returns_response_shape(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files_paginated returns a ListFilesResponse with files and next attributes."""
    resp = mock_assistant_model.list_files_paginated(limit=5)
    assert isinstance(resp, ListFilesResponse)
    assert hasattr(resp, "files")
    assert hasattr(resp, "next")


def test_list_files_paginated_limit_accepted_without_error(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files_paginated accepts limit= without raising even though the API ignores it."""
    # Should not raise; limit is a legacy parameter accepted for compatibility.
    mock_assistant_model.list_files_paginated(limit=10)


def test_list_files_paginated_page_size_accepted_without_error(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """list_files_paginated accepts page_size= without raising even though the API ignores it."""
    mock_assistant_model.list_files_paginated(page_size=10)


def test_assistant_model_list_files_paginated_without_client_ref_raises(
    detached_assistant_model: AssistantModel,
) -> None:
    """list_files_paginated raises RuntimeError when no client back-reference is set."""
    with pytest.raises(RuntimeError, match="no client reference"):
        detached_assistant_model.list_files_paginated()
