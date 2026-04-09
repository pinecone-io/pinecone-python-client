"""Integration tests for index CRUD operations (sync / REST + gRPC)."""

from __future__ import annotations

import pytest

from pinecone import GrpcIndex, Index, Pinecone
from pinecone.models.indexes.index import IndexModel, IndexSpec, IndexStatus
from pinecone.models.indexes.specs import ServerlessSpec
from tests.integration.conftest import cleanup_resource, unique_name

# ---------------------------------------------------------------------------
# list-indexes
# ---------------------------------------------------------------------------

@pytest.mark.integration
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


# ---------------------------------------------------------------------------
# describe-index
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_describe_index_returns_full_model(client: Pinecone) -> None:
    """Create a serverless index, describe it, verify all IndexModel fields."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        desc = client.indexes.describe(name)

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
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# index-handle
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_index_handle_rest(client: Pinecone) -> None:
    """pc.index(name=...) returns a REST Index with the correct host."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Get the expected host from describe
        desc = client.indexes.describe(name)
        expected_host = desc.host

        # Get an Index handle by name — triggers a describe call internally
        idx = client.index(name=name)

        assert isinstance(idx, Index)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # Index normalizes host by prepending 'https://', so the raw describe
        # host (bare hostname) will be a suffix of idx.host
        assert expected_host in idx.host
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )


@pytest.mark.integration
def test_index_handle_grpc(client: Pinecone) -> None:
    """pc.index(name=..., grpc=True) returns a GrpcIndex with the correct host."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Get the expected host from describe
        desc = client.indexes.describe(name)
        expected_host = desc.host

        # Get a GrpcIndex handle by name
        idx = client.index(name=name, grpc=True)

        assert isinstance(idx, GrpcIndex)
        assert isinstance(idx.host, str)
        assert len(idx.host) > 0
        # GrpcIndex normalizes host similarly; bare hostname should appear in idx.host
        assert expected_host in idx.host
    finally:
        cleanup_resource(
            lambda: client.indexes.delete(name),
            name,
            "index",
        )
