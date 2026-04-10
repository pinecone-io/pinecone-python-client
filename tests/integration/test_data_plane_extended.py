"""Integration tests for extended data-plane operations (sync REST + gRPC).

Phase 2 Tier 1: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
include-values-metadata, query-namespaces.
"""

from __future__ import annotations

import pytest

from pinecone import Pinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.responses import (
    FetchResponse,
    QueryResponse,
    UpsertResponse,
)
from pinecone.models.vectors.vector import ScoredVector, Vector
from tests.integration.conftest import cleanup_resource, poll_until, unique_name

# ---------------------------------------------------------------------------
# metadata-filter — REST sync
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_metadata_filter_rest(client: Pinecone) -> None:
    """Query with metadata filters ($eq, $in) returns only matching vectors (REST sync)."""
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

        # Upsert vectors with metadata
        index.upsert(
            vectors=[
                {"id": "mf-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy", "year": 2020}},
                {"id": "mf-v2", "values": [0.3, 0.4], "metadata": {"genre": "action", "year": 2021}},
                {"id": "mf-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy", "year": 2022}},
            ]
        )

        # Wait for all 3 vectors to be queryable (eventual consistency)
        poll_until(
            query_fn=lambda: index.query(vector=[0.1, 0.2], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 vectors queryable before filter tests",
        )

        # Test $eq filter: only comedy vectors should return
        comedy_result = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "comedy"}},
            include_metadata=True,
        )
        assert isinstance(comedy_result, QueryResponse)
        comedy_ids = {m.id for m in comedy_result.matches}
        assert "mf-v1" in comedy_ids
        assert "mf-v3" in comedy_ids
        assert "mf-v2" not in comedy_ids

        # Test $in filter: only action vector should return
        action_result = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$in": ["action", "thriller"]}},
            include_metadata=True,
        )
        assert isinstance(action_result, QueryResponse)
        action_ids = {m.id for m in action_result.matches}
        assert "mf-v2" in action_ids
        assert "mf-v1" not in action_ids
        assert "mf-v3" not in action_ids

        # Verify metadata is present in results (since include_metadata=True)
        for match in comedy_result.matches:
            assert isinstance(match, ScoredVector)
            assert match.metadata is not None
            assert match.metadata.get("genre") == "comedy"
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")


# ---------------------------------------------------------------------------
# metadata-filter — gRPC
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_metadata_filter_grpc(client: Pinecone) -> None:
    """Query with metadata filter via GrpcIndex returns only matching vectors."""
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
                {"id": "mf-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy"}},
                {"id": "mf-v2", "values": [0.3, 0.4], "metadata": {"genre": "action"}},
            ]
        )

        result = index.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "comedy"}},
        )
        assert isinstance(result, QueryResponse)
        ids = {m.id for m in result.matches}
        assert "mf-v1" in ids
        assert "mf-v2" not in ids
    finally:
        cleanup_resource(lambda: client.indexes.delete(name), name, "index")
