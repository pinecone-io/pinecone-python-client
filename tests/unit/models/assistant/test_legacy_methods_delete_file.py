"""Unit tests for AssistantModel.delete_file legacy shim (BC-0020)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.models.assistant.model import AssistantModel


@pytest.fixture
def mock_assistants() -> MagicMock:
    """Mock Assistants namespace with delete_file stubbed."""
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


def test_assistant_model_delete_file_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """delete_file delegates to ns.delete_file with assistant_name=self.name."""
    mock_assistant_model.delete_file(file_id="fid-123", timeout=30)
    mock_assistants.delete_file.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        file_id="fid-123",
        timeout=30,
    )


def test_assistant_model_delete_file_default_timeout(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """delete_file defaults timeout to None."""
    mock_assistant_model.delete_file(file_id="fid-456")
    mock_assistants.delete_file.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        file_id="fid-456",
        timeout=None,
    )


def test_assistant_model_delete_file_without_client_ref_raises(
    detached_assistant_model: AssistantModel,
) -> None:
    """delete_file raises RuntimeError when no client back-reference is set."""
    with pytest.raises(RuntimeError, match="no client reference"):
        detached_assistant_model.delete_file(file_id="fid-789")
