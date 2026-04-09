"""Integration tests for index CRUD operations (async / REST async)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from pinecone import AsyncPinecone


# ---------------------------------------------------------------------------
# list-indexes
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0001: PodSpecInfo.metadata_config typed dict[str,str] but API returns dict[str,list[str]]",
)
async def test_list_indexes_returns_index_list(async_client: AsyncPinecone) -> None:
    """async pc.indexes.list() returns an IndexList that is iterable and supports len()."""
    result = await async_client.indexes.list()

    # IndexList supports len()
    count = len(result)
    assert isinstance(count, int)
    assert count >= 0

    # IndexList supports iteration
    items = list(result)
    assert len(items) == count

    # .names() returns a list of strings
    names = result.names()
    assert isinstance(names, list)
    assert len(names) == count
    for name in names:
        assert isinstance(name, str)
        assert len(name) > 0
