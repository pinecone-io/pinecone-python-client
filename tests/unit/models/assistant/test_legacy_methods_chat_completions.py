"""Unit tests for AssistantModel.chat_completions legacy shim (BC-0022)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.models.assistant.model import AssistantModel


@pytest.fixture
def mock_assistants() -> MagicMock:
    """Mock Assistants namespace with chat_completions stubbed."""
    ns: MagicMock = MagicMock()
    return ns


@pytest.fixture
def mock_assistant_model(mock_assistants: MagicMock) -> AssistantModel:
    """An AssistantModel instance with a back-reference to mock_assistants."""
    model = AssistantModel(name="test-assistant", status="Ready")
    model.__dict__["_assistants"] = mock_assistants
    return model


def test_chat_completions_delegates(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """chat_completions delegates to ns.chat_completions with assistant_name=self.name."""
    mock_assistant_model.chat_completions(
        messages=[{"role": "user", "content": "hi"}],
    )
    mock_assistants.chat_completions.assert_called_once()
    assert (
        mock_assistants.chat_completions.call_args.kwargs["assistant_name"]
        == mock_assistant_model.name
    )


def test_assistant_model_chat_completions_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """chat_completions delegates with all default params."""
    mock_assistant_model.chat_completions(
        messages=[{"role": "user", "content": "hi"}],
    )
    mock_assistants.chat_completions.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        messages=[{"role": "user", "content": "hi"}],
        filter=None,
        stream=False,
        model=None,
        temperature=None,
    )


def test_assistant_model_chat_completions_legacy_streaming(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """chat_completions delegates stream=True to ns.chat_completions."""
    mock_assistant_model.chat_completions(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert mock_assistants.chat_completions.call_args.kwargs["stream"] is True


def test_assistant_model_chat_completions_legacy_all_params(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """chat_completions forwards all optional parameters to ns.chat_completions."""
    filter_val = {"key": "value"}
    mock_assistant_model.chat_completions(
        messages=[{"role": "user", "content": "hello"}],
        filter=filter_val,
        stream=False,
        model="gpt-4o",
        temperature=0.5,
    )
    mock_assistants.chat_completions.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        messages=[{"role": "user", "content": "hello"}],
        filter=filter_val,
        stream=False,
        model="gpt-4o",
        temperature=0.5,
    )


def test_assistant_model_chat_completions_legacy_no_client_raises() -> None:
    """chat_completions raises RuntimeError when no client back-reference is set."""
    detached = AssistantModel(name="detached", status="Ready")
    with pytest.raises(RuntimeError, match="no client reference"):
        detached.chat_completions(messages=[{"role": "user", "content": "hi"}])
