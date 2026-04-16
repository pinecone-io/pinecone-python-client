"""Unit tests for AssistantModel.upload_file legacy shim (BC-0016).

Note on async: there is no separate async AssistantModel — both sync and
async clients return the same ``AssistantModel`` struct. The async shim for
``upload_file`` lives on ``AsyncAssistants`` (the namespace), not on the
model itself. Therefore no ``_legacy_methods_async.py`` counterpart exists.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.models.assistant.model import AssistantModel


@pytest.fixture
def mock_assistants() -> MagicMock:
    """Mock Assistants namespace with upload_file stubbed."""
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


def test_assistant_model_upload_file_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """upload_file delegates to ns.upload_file with assistant_name=self.name."""
    mock_assistant_model.upload_file(
        file_path="/data/test.txt",
        metadata={"k": "v"},
        multimodal=False,
    )
    mock_assistants.upload_file.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        file_path="/data/test.txt",
        metadata={"k": "v"},
        multimodal=False,
        timeout=None,
        file_id=None,
    )


def test_assistant_model_upload_file_without_client_ref_raises(
    detached_assistant_model: AssistantModel,
) -> None:
    """upload_file raises RuntimeError when no client back-reference is set."""
    with pytest.raises(RuntimeError, match="no client reference"):
        detached_assistant_model.upload_file(file_path="/data/test.txt")
