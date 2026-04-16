"""Unit tests for the async AssistantNamespaceProxy via AsyncPinecone.assistant."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone.client._assistant_namespace_proxy import _AsyncAssistantNamespaceProxy
from pinecone.models.assistant.model import AssistantModel

_CANNED_ASSISTANT = AssistantModel(
    name="legacy-name",
    status="Ready",
    created_at="2025-01-15T12:00:00Z",
    updated_at="2025-01-15T12:00:00Z",
    metadata={},
    instructions=None,
    host="test-assistant-abc123.svc.pinecone.io",
)


@pytest.fixture
def mock_async_assistants_ns() -> MagicMock:
    """Mock AsyncAssistants namespace instance."""
    mock = MagicMock()
    mock.describe = AsyncMock(return_value=_CANNED_ASSISTANT)
    mock.create_assistant = AsyncMock(return_value=_CANNED_ASSISTANT)
    mock.list_assistants = AsyncMock(return_value=[_CANNED_ASSISTANT])
    return mock


@pytest.fixture
def mock_async_pinecone(mock_async_assistants_ns: MagicMock) -> MagicMock:
    """Return an AsyncPinecone-like object whose _assistants is fully mocked."""
    from pinecone.async_client.pinecone import AsyncPinecone

    pc = AsyncPinecone.__new__(AsyncPinecone)
    pc._assistants = mock_async_assistants_ns  # type: ignore[attr-defined]
    return pc


@pytest.mark.asyncio
async def test_async_pc_assistant_singular_is_namespace(mock_async_pinecone: MagicMock) -> None:
    """pc.assistant.create_assistant(...) delegates to the AsyncAssistants namespace."""
    result = await mock_async_pinecone.assistant.create_assistant(assistant_name="foo")
    assert result.name == "legacy-name"


@pytest.mark.asyncio
async def test_async_pc_assistant_singular_is_awaitable_callable(
    mock_async_pinecone: MagicMock,
) -> None:
    """await pc.assistant("foo") delegates to AsyncAssistants.describe(name="foo")."""
    result = await mock_async_pinecone.assistant("foo")
    assert result.name == "legacy-name"
    mock_async_pinecone._assistants.describe.assert_called_once_with(name="foo")


@pytest.mark.asyncio
async def test_async_pc_assistant_proxy_forwards_all_attrs(
    mock_async_pinecone: MagicMock,
) -> None:
    """pc.assistant.list_assistants() forwards unknown attrs to the AsyncAssistants namespace."""
    result = await mock_async_pinecone.assistant.list_assistants()
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_async_pc_assistants_plural_still_works(mock_async_pinecone: MagicMock) -> None:
    """pc.assistants (plural) is still accessible and returns the AsyncAssistants instance."""
    assert mock_async_pinecone.assistants is not None


def test_async_pc_assistant_is_proxy_not_coroutine(mock_async_pinecone: MagicMock) -> None:
    """pc.assistant returns a proxy object, not a coroutine."""
    proxy = mock_async_pinecone.assistant
    assert isinstance(proxy, _AsyncAssistantNamespaceProxy)


def test_async_proxy_repr(mock_async_pinecone: MagicMock) -> None:
    """repr(pc.assistant) includes AsyncAssistantNamespaceProxy."""
    r = repr(mock_async_pinecone.assistant)
    assert "AsyncAssistantNamespaceProxy" in r
