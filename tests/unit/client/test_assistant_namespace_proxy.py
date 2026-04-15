"""Unit tests for _AssistantNamespaceProxy and _AsyncAssistantNamespaceProxy."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pinecone.client._assistant_namespace_proxy import (
    _AssistantNamespaceProxy,
    _AsyncAssistantNamespaceProxy,
)
from pinecone.models.assistant.model import AssistantModel

_CANNED_ASSISTANT = AssistantModel(
    name="foo",
    status="Ready",
    created_at="2025-01-15T12:00:00Z",
    updated_at="2025-01-15T12:00:00Z",
    metadata={},
    instructions=None,
    host="foo-abc123.svc.pinecone.io",
)


# ---------------------------------------------------------------------------
# _AssistantNamespaceProxy (sync)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_assistants_ns() -> MagicMock:
    """Mock Assistants namespace instance."""
    mock = MagicMock()
    mock.describe.return_value = _CANNED_ASSISTANT
    mock.create.return_value = _CANNED_ASSISTANT
    mock.list_assistants.return_value = [_CANNED_ASSISTANT]
    return mock


@pytest.fixture
def proxy(mock_assistants_ns: MagicMock) -> _AssistantNamespaceProxy:
    return _AssistantNamespaceProxy(mock_assistants_ns)


def test_proxy_call_delegates_to_describe(
    proxy: _AssistantNamespaceProxy, mock_assistants_ns: MagicMock
) -> None:
    """proxy("foo") calls assistants.describe(name="foo")."""
    result = proxy("foo")
    mock_assistants_ns.describe.assert_called_once_with(name="foo")
    assert result is _CANNED_ASSISTANT


def test_proxy_getattr_forwards_to_assistants(
    proxy: _AssistantNamespaceProxy, mock_assistants_ns: MagicMock
) -> None:
    """proxy.create_assistant(...) forwards to the underlying Assistants attribute."""
    proxy.create_assistant(name="foo")
    mock_assistants_ns.create_assistant.assert_called_once_with(name="foo")


def test_proxy_list_assistants(
    proxy: _AssistantNamespaceProxy, mock_assistants_ns: MagicMock
) -> None:
    """proxy.list_assistants() delegates to assistants.list_assistants()."""
    result = proxy.list_assistants()
    mock_assistants_ns.list_assistants.assert_called_once()
    assert result is not None


def test_proxy_repr(proxy: _AssistantNamespaceProxy, mock_assistants_ns: MagicMock) -> None:
    """repr(proxy) includes the underlying Assistants repr."""
    r = repr(proxy)
    assert "AssistantNamespaceProxy" in r


# ---------------------------------------------------------------------------
# Integration: _AssistantNamespaceProxy via Pinecone.assistant property
# ---------------------------------------------------------------------------


def test_pc_assistant_singular_is_namespace(mock_assistants: Any) -> None:
    """pc.assistant.create_assistant(...) works via namespace-style access."""
    from pinecone._client import Pinecone

    pc = Pinecone.__new__(Pinecone)
    pc._assistants = mock_assistants  # type: ignore[attr-defined]
    result = pc.assistant.create_assistant(assistant_name="foo")
    assert result.name == "legacy-name"


def test_pc_assistant_singular_is_callable(mock_assistants: Any) -> None:
    """pc.assistant("foo") works via legacy call form."""
    from pinecone._client import Pinecone

    pc = Pinecone.__new__(Pinecone)
    pc._assistants = mock_assistants  # type: ignore[attr-defined]
    result = pc.assistant("foo")
    assert isinstance(result, AssistantModel)


def test_pc_assistant_proxy_forwards_all_attrs(mock_assistants: Any) -> None:
    """pc.assistant.list_assistants() delegates unknown attrs to Assistants."""
    from pinecone._client import Pinecone

    pc = Pinecone.__new__(Pinecone)
    pc._assistants = mock_assistants  # type: ignore[attr-defined]
    assert pc.assistant.list_assistants() is not None


def test_pc_assistants_plural_still_works(mock_assistants: Any) -> None:
    """pc.assistants (plural) is still accessible and returns the Assistants instance."""
    from pinecone._client import Pinecone

    pc = Pinecone.__new__(Pinecone)
    pc._assistants = mock_assistants  # type: ignore[attr-defined]
    assert pc.assistants is mock_assistants


# ---------------------------------------------------------------------------
# _AsyncAssistantNamespaceProxy (async)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_async_assistants_ns() -> MagicMock:
    """Mock AsyncAssistants namespace instance."""
    mock = MagicMock()
    mock.describe = AsyncMock(return_value=_CANNED_ASSISTANT)
    mock.create = AsyncMock(return_value=_CANNED_ASSISTANT)
    return mock


@pytest.fixture
def async_proxy(mock_async_assistants_ns: MagicMock) -> _AsyncAssistantNamespaceProxy:
    return _AsyncAssistantNamespaceProxy(mock_async_assistants_ns)


@pytest.mark.asyncio
async def test_async_proxy_call_delegates_to_describe(
    async_proxy: _AsyncAssistantNamespaceProxy, mock_async_assistants_ns: MagicMock
) -> None:
    """await proxy("foo") calls assistants.describe(name="foo")."""
    result = await async_proxy("foo")
    mock_async_assistants_ns.describe.assert_called_once_with(name="foo")
    assert result is _CANNED_ASSISTANT


@pytest.mark.asyncio
async def test_async_proxy_getattr_forwards_to_async_assistants(
    async_proxy: _AsyncAssistantNamespaceProxy, mock_async_assistants_ns: MagicMock
) -> None:
    """async_proxy.create(...) forwards to the underlying AsyncAssistants attribute."""
    await async_proxy.create(name="foo")
    mock_async_assistants_ns.create.assert_called_once_with(name="foo")


def test_async_proxy_repr(
    async_proxy: _AsyncAssistantNamespaceProxy, mock_async_assistants_ns: MagicMock
) -> None:
    """repr(async_proxy) includes the underlying AsyncAssistants repr."""
    r = repr(async_proxy)
    assert "AsyncAssistantNamespaceProxy" in r


# ---------------------------------------------------------------------------
# Integration: _AsyncAssistantNamespaceProxy via AsyncPinecone.assistant property
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_pc_assistant_singular_is_callable() -> None:
    """await pc.assistant("foo") works via the async proxy."""
    from pinecone.async_client.pinecone import AsyncPinecone

    mock_async = MagicMock()
    mock_async.describe = AsyncMock(return_value=_CANNED_ASSISTANT)

    pc = AsyncPinecone.__new__(AsyncPinecone)
    pc._assistants = mock_async  # type: ignore[attr-defined]
    result = await pc.assistant("foo")
    assert isinstance(result, AssistantModel)
    mock_async.describe.assert_called_once_with(name="foo")


def test_async_pc_assistant_is_property() -> None:
    """pc.assistant accessed without calling returns the proxy (not a coroutine)."""
    from pinecone.async_client.pinecone import AsyncPinecone

    mock_async = MagicMock()
    pc = AsyncPinecone.__new__(AsyncPinecone)
    pc._assistants = mock_async  # type: ignore[attr-defined]
    proxy = pc.assistant
    assert isinstance(proxy, _AsyncAssistantNamespaceProxy)
