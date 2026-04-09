"""Integration tests for data-plane vector operations (sync REST + gRPC)."""

from __future__ import annotations

import pytest
from pinecone import Pinecone
from pinecone.models.vectors.responses import QueryResponse, UpsertResponse
from pinecone.models.vectors.vector import ScoredVector
from pinecone.models.indexes.specs import ServerlessSpec

from tests.integration.conftest import cleanup_resource, poll_until, unique_name


# ---------------------------------------------------------------------------
# upsert — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_upsert_vectors_rest(client: Pinecone) -> None:
    """Upsert vectors via REST Index and verify upserted_count."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        result = index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_query_by_vector_rest(client: Pinecone) -> None:
    """Query by vector via REST Index and verify matches structure."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name)

        index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        # Wait for all 3 vectors to be queryable (eventual consistency)
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=3),
            check_fn=lambda r: len(r.matches) == 3,
            timeout=120,
            description="all 3 vectors queryable after upsert",
        )

        result = index.query(vector=[0.1, 0.2], top_k=2)

        assert isinstance(result, QueryResponse)
        assert len(result.matches) == 2
        for match in result.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)
        # Scores must be in descending order
        scores = [m.score for m in result.matches]
        assert scores == sorted(scores, reverse=True)
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# upsert — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_upsert_vectors_grpc(client: Pinecone) -> None:
    """Upsert vectors via GrpcIndex and verify upserted_count."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        result = index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
                {"id": "v3", "values": [0.5, 0.6]},
            ]
        )

        assert isinstance(result, UpsertResponse)
        assert result.upserted_count == 3
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# query — gRPC (xfail: IT-0002)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="SDK bug IT-0002: pinecone._grpc Rust extension not installed; ModuleNotFoundError on GrpcIndex creation",
)
def test_query_by_vector_grpc(client: Pinecone) -> None:
    """Query by vector via GrpcIndex and verify matches structure."""
    name = unique_name("idx")
    try:
        client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        index = client.index(name=name, grpc=True)

        index.upsert(
            vectors=[
                {"id": "v1", "values": [0.1, 0.2]},
                {"id": "v2", "values": [0.3, 0.4]},
            ]
        )

        result = index.query(vector=[0.1, 0.2], top_k=1)

        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1
        assert result.matches[0].id == "v1"
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
