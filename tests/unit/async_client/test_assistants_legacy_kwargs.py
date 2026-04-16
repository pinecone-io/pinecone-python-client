"""Unit tests for legacy kwarg aliases on AsyncAssistants.create."""

from __future__ import annotations

import pytest

from pinecone.async_client.assistants import AsyncAssistants
from pinecone.errors.exceptions import PineconeValueError


@pytest.mark.asyncio
async def test_async_create_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async create."""
    result = await mock_async_assistants.create(assistant_name="legacy-name")
    assert result.name == "legacy-name"


@pytest.mark.asyncio
async def test_async_create_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.create(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_create_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.create(name="foo", bogus_param=1)


@pytest.mark.asyncio
async def test_async_create_missing_name_raises(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Calling create() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        await mock_async_assistants.create()


@pytest.mark.asyncio
async def test_async_create_with_name_still_works(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """The canonical name= parameter continues to work as before."""
    result = await mock_async_assistants.create(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture
