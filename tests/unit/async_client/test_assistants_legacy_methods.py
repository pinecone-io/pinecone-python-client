"""Unit tests for AsyncAssistantsLegacyNamespaceMixin shim methods.

This module covers the create_assistant() shim introduced in BC-0030.
Subsequent tasks (BC-0031..BC-0034) append tests for additional shim methods.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from pinecone.async_client.assistants import AsyncAssistants


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
