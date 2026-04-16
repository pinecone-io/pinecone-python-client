"""Unit tests for AssistantModel.upload_bytes_stream legacy shim (BC-0017)."""

from __future__ import annotations

import io
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


def test_assistant_model_upload_bytes_stream_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """upload_bytes_stream delegates to ns.upload_file with file_stream and file_name."""
    stream = io.BytesIO(b"hello world")
    mock_assistant_model.upload_bytes_stream(stream, "test.txt")
    mock_assistants.upload_file.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        file_stream=stream,
        file_name="test.txt",
        metadata=None,
        multimodal=None,
        timeout=None,
        file_id=None,
    )


def test_upload_bytes_stream_delegates(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """upload_bytes_stream passes all optional kwargs to upload_file."""
    stream = io.BytesIO(b"hello world")
    mock_assistant_model.upload_bytes_stream(
        stream,
        "test.txt",
        metadata={"key": "val"},
        multimodal=True,
        timeout=30,
        file_id="fid-123",
    )
    mock_assistants.upload_file.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        file_stream=stream,
        file_name="test.txt",
        metadata={"key": "val"},
        multimodal=True,
        timeout=30,
        file_id="fid-123",
    )


def test_upload_bytes_stream_without_client_ref_raises(
    detached_assistant_model: AssistantModel,
) -> None:
    """upload_bytes_stream raises RuntimeError when no client back-reference is set."""
    stream = io.BytesIO(b"data")
    with pytest.raises(RuntimeError, match="no client reference"):
        detached_assistant_model.upload_bytes_stream(stream, "test.txt")
