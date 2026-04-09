"""Integration tests for index CRUD operations (sync / REST + gRPC)."""

from __future__ import annotations

import pytest
from pinecone import Pinecone

from tests.integration.conftest import cleanup_resource, unique_name


# ---------------------------------------------------------------------------
# list-indexes
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0001: PodSpecInfo.metadata_config typed dict[str,str] but API returns dict[str,list[str]]",
)
def test_list_indexes_returns_index_list(client: Pinecone) -> None:
    """pc.indexes.list() returns an IndexList that is iterable and supports len()."""
    result = client.indexes.list()

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
