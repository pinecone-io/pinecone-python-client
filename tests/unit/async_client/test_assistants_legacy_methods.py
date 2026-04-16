"""Unit tests for AsyncAssistantsLegacyNamespaceMixin shim methods.

This module covers the create_assistant() shim introduced in BC-0030.
Subsequent tasks (BC-0031..BC-0034) append tests for additional shim methods.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from pinecone.async_client.assistants import AsyncAssistants
from pinecone.models.assistant.list import ListAssistantsResponse
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


# ---------------------------------------------------------------------------
# list_assistants shim tests (BC-0032)
# ---------------------------------------------------------------------------


async def test_async_list_assistants_returns_list(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """list_assistants() materializes the async paginator and returns a plain list."""
    mock_async_assistants._adapter.to_assistant_list.return_value = ListAssistantsResponse(  # type: ignore[attr-defined]
        assistants=[_CANNED_ASSISTANT], next=None
    )
    result = await mock_async_assistants.list_assistants()
    assert isinstance(result, list)
    assert all(isinstance(a, AssistantModel) for a in result)


async def test_async_list_assistants_paginated_returns_response_shape(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """list_assistants_paginated() returns a ListAssistantsResponse."""
    mock_async_assistants._adapter.to_assistant_list.return_value = ListAssistantsResponse(  # type: ignore[attr-defined]
        assistants=[_CANNED_ASSISTANT], next=None
    )
    resp = await mock_async_assistants.list_assistants_paginated(limit=2)
    assert isinstance(resp, ListAssistantsResponse)
    assert hasattr(resp, "assistants")
    assert hasattr(resp, "next")


async def test_async_list_assistants_paginated_legacy_limit_alias(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """list_assistants_paginated(limit=...) passes page_size= to list_page."""
    canned_response = ListAssistantsResponse(assistants=[], next=None)
    spy_list_page = AsyncMock(return_value=canned_response)
    mock_async_assistants.list_page = spy_list_page  # type: ignore[method-assign]

    await mock_async_assistants.list_assistants_paginated(limit=5)

    spy_list_page.assert_called_once_with(page_size=5, pagination_token=None)


# ---------------------------------------------------------------------------
# create_assistant shim tests (BC-0030)
# ---------------------------------------------------------------------------


async def test_async_create_assistant_legacy_method(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """create_assistant(assistant_name=...) delegates to create() and returns a model."""
    result = await mock_async_assistants.create_assistant(
        assistant_name="foo",
        instructions="be nice",
    )
    # Returns the canned fixture value from mock_async_assistants
    assert result.name == "legacy-name"


async def test_async_create_assistant_forwards_to_create(
    mock_async_assistants: AsyncAssistants,
    spy_async_create: AsyncMock,
) -> None:
    """create_assistant() forwards the right arguments to create()."""
    await mock_async_assistants.create_assistant(assistant_name="foo")
    spy_async_create.assert_called_once_with(
        name="foo",
        instructions=None,
        metadata=None,
        region="us",
        timeout=None,
    )


# ---------------------------------------------------------------------------
# describe_assistant shim tests (BC-0031)
# ---------------------------------------------------------------------------


async def test_async_describe_assistant_legacy_method(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """describe_assistant(assistant_name=...) delegates to describe() and returns a model."""
    result = await mock_async_assistants.describe_assistant(assistant_name="foo")
    # Returns the canned fixture value from mock_async_assistants
    assert result.name == "legacy-name"


async def test_async_describe_assistant_legacy_with_name_kwarg(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """describe_assistant(name=...) accepts the current-style kwarg."""
    result = await mock_async_assistants.describe_assistant(name="foo")
    assert result.name == "legacy-name"


async def test_async_describe_assistant_rejects_both_names(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """describe_assistant() raises TypeError when both assistant_name and name are given."""
    with pytest.raises(TypeError, match="Pass only one"):
        await mock_async_assistants.describe_assistant("a", name="b")


async def test_async_describe_assistant_forwards_to_describe(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """describe_assistant() forwards assistant_name as name= to describe()."""
    spy = AsyncMock(side_effect=mock_async_assistants.describe)
    mock_async_assistants.describe = spy  # type: ignore[method-assign]

    await mock_async_assistants.describe_assistant(assistant_name="my-assistant")
    spy.assert_called_once_with(name="my-assistant")


# ---------------------------------------------------------------------------
# update_assistant shim tests (BC-0033)
# ---------------------------------------------------------------------------


async def test_async_update_assistant_legacy_method(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """update_assistant(assistant_name=...) delegates to update() and returns a model."""
    result = await mock_async_assistants.update_assistant(
        assistant_name="foo",
        instructions="be precise",
        metadata={"k": "v"},
    )
    # Returns the canned fixture value from mock_async_assistants
    assert result.name == "legacy-name"


async def test_async_update_assistant_legacy_positional(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """update_assistant() accepts positional args (legacy call convention)."""
    result = await mock_async_assistants.update_assistant("foo", "be nice", {"k": "v"})
    # Returns the canned fixture value from mock_async_assistants
    assert result.name == "legacy-name"


# ---------------------------------------------------------------------------
# delete_assistant shim tests (BC-0034)
# ---------------------------------------------------------------------------


async def test_async_delete_assistant_legacy_method(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """delete_assistant(assistant_name=...) delegates to delete()."""
    spy = AsyncMock()
    mock_async_assistants.delete = spy  # type: ignore[method-assign]
    await mock_async_assistants.delete_assistant(assistant_name="foo", timeout=30)
    spy.assert_called_once_with(name="foo", timeout=30)


async def test_async_delete_assistant_legacy_positional(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """delete_assistant("foo", 30) using positional legacy-style call."""
    spy = AsyncMock()
    mock_async_assistants.delete = spy  # type: ignore[method-assign]
    await mock_async_assistants.delete_assistant("foo", 30)
    spy.assert_called_once_with(name="foo", timeout=30)
