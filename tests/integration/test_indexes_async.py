"""Integration tests for index CRUD operations (async / REST async)."""

from __future__ import annotations

import pytest

from pinecone import AsyncIndex, AsyncPinecone
from pinecone.models.indexes.index import IndexModel, IndexSpec, IndexStatus
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import async_cleanup_resource, unique_name

# ---------------------------------------------------------------------------
# list-indexes
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# create-index
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_serverless_index_becomes_ready(async_client: AsyncPinecone) -> None:
    """Create a serverless index asynchronously, wait for ready state, verify fields, then delete."""
    name = unique_name("idx")
    try:
        model = await async_client.indexes.create(
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
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# describe-index
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_describe_index_returns_full_model(async_client: AsyncPinecone) -> None:
    """Create a serverless index asynchronously, describe it, verify all IndexModel fields."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        desc = await async_client.indexes.describe(name)

        assert isinstance(desc, IndexModel)
        assert desc.name == name
        assert desc.dimension == 4
        assert desc.metric == "dotproduct"
        assert desc.vector_type == "dense"
        assert desc.deletion_protection == "disabled"

        # Status fields
        assert isinstance(desc.status, IndexStatus)
        assert desc.status.ready is True
        assert isinstance(desc.status.state, str)
        assert len(desc.status.state) > 0

        # Spec is serverless
        assert isinstance(desc.spec, IndexSpec)
        assert desc.spec.serverless is not None
        assert desc.spec.pod is None
        assert desc.spec.serverless.cloud == "aws"
        assert desc.spec.serverless.region == "us-east-1"

        # Host is a non-empty string
        assert isinstance(desc.host, str)
        assert len(desc.host) > 0
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-handle
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_index_handle_rest_async(async_client: AsyncPinecone) -> None:
    """async pc.index(name=...) returns an AsyncIndex with the correct host.

    AsyncPinecone.index() requires the host to be cached via a prior
    describe call. We call describe() first, which populates the host cache,
    then pc.index(name=name) should succeed.
    """
    name = unique_name("idx")
    idx = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # describe() caches the host in AsyncPinecone's host cache
        desc = await async_client.indexes.describe(name)
        expected_host = desc.host

        # Get an AsyncIndex handle by name (uses cached host)
        idx = async_client.index(name=name)

        assert isinstance(idx, AsyncIndex)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # AsyncIndex normalizes host by prepending 'https://', so the raw describe
        # host (bare hostname) will appear within idx.host
        assert expected_host in idx.host
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-tags
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_index_with_tags(async_client: AsyncPinecone) -> None:
    """Create a serverless index with tags asynchronously and verify they are returned by describe."""
    name = unique_name("idx")
    tags = {"env": "integration-test", "version": "1"}
    try:
        model = await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            tags=tags,
            timeout=300,
        )

        # Tags should be present on the create response
        assert model.tags is not None
        assert model.tags.get("env") == "integration-test"
        assert model.tags.get("version") == "1"

        # Tags should also be present on describe
        desc = await async_client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"
        assert desc.tags.get("version") == "1"
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# configure-index
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_configure_index_updates_tags(async_client: AsyncPinecone) -> None:
    """async configure() merges tags — add new tags, update existing, remove via empty string."""
    name = unique_name("idx")
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            tags={"env": "integration-test", "version": "1", "to-remove": "yes"},
            timeout=300,
        )

        # Add a new tag and update an existing tag
        await async_client.indexes.configure(
            name,
            tags={"version": "2", "new-key": "new-val"},
        )

        desc = await async_client.indexes.describe(name)
        assert desc.tags is not None
        assert desc.tags.get("env") == "integration-test"   # untouched
        assert desc.tags.get("version") == "2"              # updated
        assert desc.tags.get("new-key") == "new-val"        # added
        assert desc.tags.get("to-remove") == "yes"          # not yet removed

        # Remove a tag by setting its value to ""
        await async_client.indexes.configure(
            name,
            tags={"to-remove": ""},
        )

        desc2 = await async_client.indexes.describe(name)
        assert desc2.tags is not None
        assert "to-remove" not in desc2.tags or desc2.tags.get("to-remove") == ""
        assert desc2.tags.get("version") == "2"             # preserved from previous configure
    finally:
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )
