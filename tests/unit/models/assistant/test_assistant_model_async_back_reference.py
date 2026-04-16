"""Unit tests for async back-reference detection in AssistantModelLegacyMethodsMixin (BC-0036).

When an AssistantModel is obtained from AsyncAssistants, its back-reference
is set to an AsyncAssistants instance. Calling sync legacy shims on such a
model must raise TypeError with a message directing the caller to use the
async namespace directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone._internal.config import PineconeConfig
from pinecone.async_client.assistants import AsyncAssistants
from pinecone.client.assistants import Assistants
from pinecone.models.assistant.model import AssistantModel

BASE_URL = "https://api.test.pinecone.io"

_CANNED_ASSISTANT = AssistantModel(
    name="foo",
    status="Ready",
    created_at="2025-01-15T12:00:00Z",
    updated_at="2025-01-15T12:00:00Z",
    metadata={},
    instructions=None,
    host="test-assistant-abc123.svc.pinecone.io",
)


@pytest.fixture
def mock_async_assistants() -> AsyncAssistants:
    """AsyncAssistants with mocked HTTP and adapter returning a canned model."""
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    obj = AsyncAssistants(config=config)

    mock_http = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = b"{}"
    mock_http.get.return_value = mock_response
    obj._http = mock_http  # type: ignore[attr-defined]

    mock_adapter = MagicMock()
    mock_adapter.to_assistant.return_value = _CANNED_ASSISTANT
    obj._adapter = mock_adapter  # type: ignore[attr-defined]

    return obj


@pytest.fixture
def mock_sync_assistants() -> Assistants:
    """Assistants with mocked HTTP, adapter, and upload_file returning a canned model."""
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    obj = Assistants(config=config)

    mock_http = MagicMock()
    mock_response = MagicMock()
    mock_response.content = b"{}"
    mock_http.get.return_value = mock_response
    obj._http = mock_http  # type: ignore[attr-defined]

    mock_adapter = MagicMock()
    mock_adapter.to_assistant.return_value = _CANNED_ASSISTANT
    obj._adapter = mock_adapter  # type: ignore[attr-defined]

    # Stub upload_file so the test doesn't attempt real filesystem I/O.
    obj.upload_file = MagicMock()  # type: ignore[method-assign]

    return obj


@pytest.mark.asyncio
async def test_async_sourced_model_raises_typeerror_on_sync_shim(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Sync shim on an async-sourced model must raise TypeError with a clear message."""
    model = await mock_async_assistants.describe(name="foo")
    with pytest.raises(TypeError, match="sync-only"):
        model.upload_file(file_path="/fake/test.txt")


@pytest.mark.asyncio
async def test_async_sourced_model_raises_typeerror_on_chat_shim(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """chat() sync shim raises TypeError when back-ref is AsyncAssistants."""
    model = await mock_async_assistants.describe(name="foo")
    with pytest.raises(TypeError, match="sync-only"):
        model.chat(messages=[{"role": "user", "content": "hi"}])


def test_sync_sourced_model_does_not_raise(
    mock_sync_assistants: Assistants,
) -> None:
    """Sync shim on a model from sync Assistants must not raise TypeError."""
    model = mock_sync_assistants.describe(name="foo")
    # Should NOT raise — back-ref is sync Assistants. Return value is a MagicMock.
    model.upload_file(file_path="/fake/test.txt")  # mocked


def test_resolve_assistants_raises_typeerror_for_async_ref() -> None:
    """_resolve_assistants raises TypeError when back-ref is AsyncAssistants."""
    from pinecone.async_client.assistants import AsyncAssistants as _AsyncAssistants

    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    async_ns = _AsyncAssistants(config=config)

    model = AssistantModel(name="bar", status="Ready")
    model.__dict__["_assistants"] = async_ns

    with pytest.raises(TypeError, match="sync-only"):
        model._resolve_assistants()


def test_resolve_assistants_raises_runtime_error_for_no_ref() -> None:
    """_resolve_assistants raises RuntimeError when no back-ref is set."""
    model = AssistantModel(name="orphan", status="Ready")
    with pytest.raises(RuntimeError, match="no client reference"):
        model._resolve_assistants()


def test_resolve_assistants_returns_sync_ref() -> None:
    """_resolve_assistants returns the sync Assistants ref without error."""
    config = PineconeConfig(api_key="test-key", host=BASE_URL)
    sync_ns = Assistants(config=config)

    model = AssistantModel(name="baz", status="Ready")
    model.__dict__["_assistants"] = sync_ns

    result = model._resolve_assistants()
    assert result is sync_ns
