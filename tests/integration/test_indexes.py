"""Integration tests for index CRUD operations (sync / REST + gRPC)."""

from __future__ import annotations

import pytest
from pinecone import Pinecone
from pinecone.models.indexes.specs import ServerlessSpec

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


# ---------------------------------------------------------------------------
# create-index
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_create_serverless_index_becomes_ready(client: Pinecone) -> None:
    """Create a serverless index, wait for ready state, verify fields, then delete."""
    name = unique_name("idx")
    try:
        model = client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        assert model.name == name
        assert model.dimension == 2
        assert model.metric == "cosine"
        assert model.status.ready is True
        assert model.status.state == "Ready"
        assert model.spec.serverless is not None
        assert model.spec.serverless.cloud == "aws"
        assert model.spec.serverless.region == "us-east-1"
        assert model.deletion_protection == "disabled"
        assert isinstance(model.host, str)
        assert len(model.host) > 0
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )
