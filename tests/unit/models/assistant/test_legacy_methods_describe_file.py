"""Unit tests for AssistantModel.describe_file legacy shim (BC-0018)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.models.assistant.model import AssistantModel


@pytest.fixture
def mock_assistants() -> MagicMock:
    """Mock Assistants namespace with describe_file stubbed."""
    ns: MagicMock = MagicMock()
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


def test_assistant_model_describe_file_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """describe_file delegates to ns.describe_file with assistant_name=self.name."""
    mock_assistant_model.describe_file(file_id="fid-123", include_url=True)
    mock_assistants.describe_file.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        file_id="fid-123",
        include_url=True,
    )


def test_assistant_model_describe_file_default_include_url(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """describe_file defaults include_url to False."""
    mock_assistant_model.describe_file(file_id="fid-456")
    mock_assistants.describe_file.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        file_id="fid-456",
        include_url=False,
    )


def test_assistant_model_describe_file_without_client_ref_raises(
    detached_assistant_model: AssistantModel,
) -> None:
    """describe_file raises RuntimeError when no client back-reference is set."""
    with pytest.raises(RuntimeError, match="no client reference"):
        detached_assistant_model.describe_file(file_id="fid-789")
