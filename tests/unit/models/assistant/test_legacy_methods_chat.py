"""Unit tests for AssistantModel.chat legacy shim (BC-0021)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.models.assistant.model import AssistantModel


@pytest.fixture
def mock_assistants() -> MagicMock:
    """Mock Assistants namespace with chat stubbed."""
    ns: MagicMock = MagicMock()
    return ns


@pytest.fixture
def mock_assistant_model(mock_assistants: MagicMock) -> AssistantModel:
    """An AssistantModel instance with a back-reference to mock_assistants."""
    model = AssistantModel(name="test-assistant", status="Ready")
    model.__dict__["_assistants"] = mock_assistants
    return model


def test_assistant_model_chat_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """chat delegates to ns.chat with assistant_name=self.name (non-streaming)."""
    mock_assistant_model.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
    )
    mock_assistants.chat.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        messages=[{"role": "user", "content": "hi"}],
        filter=None,
        stream=False,
        model="gpt-4o",
        temperature=None,
        json_response=False,
        include_highlights=False,
        context_options=None,
    )


def test_assistant_model_chat_legacy_streaming(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """chat delegates stream=True to ns.chat."""
    mock_assistant_model.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert mock_assistants.chat.call_args.kwargs["stream"] is True


def test_assistant_model_chat_legacy_all_params(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """chat forwards all optional parameters to ns.chat."""
    filter_val = {"key": "value"}
    context_opts = {"top_k": 5}
    mock_assistant_model.chat(
        messages=[{"role": "user", "content": "hello"}],
        filter=filter_val,
        stream=False,
        model="claude-3",
        temperature=0.7,
        json_response=True,
        include_highlights=True,
        context_options=context_opts,
    )
    mock_assistants.chat.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        messages=[{"role": "user", "content": "hello"}],
        filter=filter_val,
        stream=False,
        model="claude-3",
        temperature=0.7,
        json_response=True,
        include_highlights=True,
        context_options=context_opts,
    )


def test_assistant_model_chat_legacy_no_client_raises() -> None:
    """chat raises RuntimeError when no client back-reference is set."""
    detached = AssistantModel(name="detached", status="Ready")
    with pytest.raises(RuntimeError, match="no client reference"):
        detached.chat(messages=[{"role": "user", "content": "hi"}])
