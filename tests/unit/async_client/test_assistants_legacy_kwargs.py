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


@pytest.mark.asyncio
async def test_async_describe_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async describe."""
    result = await mock_async_assistants.describe(assistant_name="foo")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_describe_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.describe(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_describe_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.describe(name="foo", random=1)


@pytest.mark.asyncio
async def test_async_describe_missing_name_raises(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Calling describe() without name or assistant_name raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="missing required"):
        await mock_async_assistants.describe()


@pytest.mark.asyncio
async def test_async_describe_with_name_still_works(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """The canonical name= parameter continues to work as before."""
    result = await mock_async_assistants.describe(name="my-assistant")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_update_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async update."""
    result = await mock_async_assistants.update(assistant_name="foo", instructions="be precise")
    assert result.name == "legacy-name"  # canned response from fixture


@pytest.mark.asyncio
async def test_async_update_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.update(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_update_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.update(name="foo", bogus=1)


@pytest.mark.asyncio
async def test_async_delete_accepts_legacy_assistant_name(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """assistant_name= is accepted as a legacy alias for name= on async delete."""
    await mock_async_assistants.delete(assistant_name="foo", timeout=-1)


@pytest.mark.asyncio
async def test_async_delete_rejects_both(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing both name= and assistant_name= raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="both"):
        await mock_async_assistants.delete(name="a", assistant_name="b")


@pytest.mark.asyncio
async def test_async_delete_rejects_unknown(
    mock_async_assistants: AsyncAssistants,
) -> None:
    """Passing an unrecognised kwarg raises PineconeValueError."""
    with pytest.raises(PineconeValueError, match="unexpected"):
        await mock_async_assistants.delete(name="foo", bogus=1)
