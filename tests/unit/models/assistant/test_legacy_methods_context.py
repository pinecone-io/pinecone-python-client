"""Unit tests for AssistantModel.context legacy shim (BC-0023)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinecone.models.assistant.model import AssistantModel
from pinecone.models.assistant.options import ContextOptions


@pytest.fixture
def mock_assistants() -> MagicMock:
    """Mock Assistants namespace with context stubbed."""
    ns: MagicMock = MagicMock()
    return ns


@pytest.fixture
def mock_assistant_model(mock_assistants: MagicMock) -> AssistantModel:
    """An AssistantModel instance with a back-reference to mock_assistants."""
    model = AssistantModel(name="test-assistant", status="Ready")
    model.__dict__["_assistants"] = mock_assistants
    return model


def test_assistant_model_context_legacy(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """context delegates to ns.context with assistant_name=self.name."""
    mock_assistant_model.context(query="what is x?", top_k=5)
    mock_assistants.context.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        query="what is x?",
        messages=None,
        filter=None,
        top_k=5,
        snippet_size=None,
        multimodal=None,
        include_binary_content=None,
    )


def test_assistant_model_context_legacy_with_filter(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """context forwards filter and snippet_size to ns.context."""
    mock_assistant_model.context(
        query="explain Pinecone",
        filter={"source": "docs"},
        top_k=10,
        snippet_size=200,
    )
    mock_assistants.context.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        query="explain Pinecone",
        messages=None,
        filter={"source": "docs"},
        top_k=10,
        snippet_size=200,
        multimodal=None,
        include_binary_content=None,
    )


def test_assistant_model_context_legacy_context_options_struct(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """context unpacks ContextOptions struct into multimodal and include_binary_content."""
    opts = ContextOptions(multimodal=True, include_binary_content=False, top_k=3, snippet_size=150)
    mock_assistant_model.context(query="images?", context_options=opts)
    mock_assistants.context.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        query="images?",
        messages=None,
        filter=None,
        top_k=3,
        snippet_size=150,
        multimodal=True,
        include_binary_content=False,
    )


def test_assistant_model_context_legacy_context_options_dict(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """context unpacks dict context_options into multimodal and include_binary_content."""
    mock_assistant_model.context(
        query="videos?",
        context_options={"multimodal": True, "include_binary_content": True, "top_k": 7},
    )
    mock_assistants.context.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        query="videos?",
        messages=None,
        filter=None,
        top_k=7,
        snippet_size=None,
        multimodal=True,
        include_binary_content=True,
    )


def test_assistant_model_context_legacy_top_k_not_overridden_by_context_options(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """Explicit top_k takes precedence over context_options.top_k."""
    opts = ContextOptions(top_k=99)
    mock_assistant_model.context(query="test", top_k=5, context_options=opts)
    call_kwargs = mock_assistants.context.call_args.kwargs
    assert call_kwargs["top_k"] == 5


def test_assistant_model_context_legacy_no_client_raises() -> None:
    """context raises RuntimeError when no client back-reference is set."""
    detached = AssistantModel(name="detached", status="Ready")
    with pytest.raises(RuntimeError, match="no client reference"):
        detached.context(query="hello")


def test_assistant_model_context_legacy_with_messages(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """context forwards messages to ns.context with query=None."""
    msgs = [{"role": "user", "content": "Tell me about my files"}]
    mock_assistant_model.context(messages=msgs)
    mock_assistants.context.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        query=None,
        messages=msgs,
        filter=None,
        top_k=None,
        snippet_size=None,
        multimodal=None,
        include_binary_content=None,
    )


def test_assistant_model_context_legacy_messages_with_filter_and_top_k(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """messages path forwards filter, top_k, and snippet_size."""
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "follow up"},
    ]
    mock_assistant_model.context(
        messages=msgs,
        filter={"source": "docs"},
        top_k=4,
        snippet_size=120,
    )
    mock_assistants.context.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        query=None,
        messages=msgs,
        filter={"source": "docs"},
        top_k=4,
        snippet_size=120,
        multimodal=None,
        include_binary_content=None,
    )


def test_assistant_model_context_legacy_messages_with_context_options(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """messages path coexists with context_options unpacking."""
    msgs = [{"role": "user", "content": "show images"}]
    opts = ContextOptions(multimodal=True, include_binary_content=True, top_k=5)
    mock_assistant_model.context(messages=msgs, context_options=opts)
    mock_assistants.context.assert_called_once_with(
        assistant_name=mock_assistant_model.name,
        query=None,
        messages=msgs,
        filter=None,
        top_k=5,
        snippet_size=None,
        multimodal=True,
        include_binary_content=True,
    )


def test_assistant_model_context_legacy_explicit_multimodal_overrides_context_options(
    mock_assistants: MagicMock, mock_assistant_model: AssistantModel
) -> None:
    """Explicit multimodal/include_binary_content kwargs win over context_options."""
    opts = ContextOptions(multimodal=False, include_binary_content=False)
    mock_assistant_model.context(
        query="q",
        multimodal=True,
        include_binary_content=True,
        context_options=opts,
    )
    call_kwargs = mock_assistants.context.call_args.kwargs
    assert call_kwargs["multimodal"] is True
    assert call_kwargs["include_binary_content"] is True
