"""Integration tests for extended data-plane operations (async REST).

Phase 2 Tier 1: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
include-values-metadata, query-namespaces.
"""

from __future__ import annotations

import pytest

from pinecone import AsyncIndex, AsyncPinecone
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.responses import (
    FetchResponse,
    QueryResponse,
    UpsertResponse,
)
from pinecone.models.vectors.sparse import SparseValues
from pinecone.models.vectors.vector import ScoredVector, Vector
from tests.integration.conftest import async_cleanup_resource, async_poll_until, unique_name

# ---------------------------------------------------------------------------
# metadata-filter — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_filter_rest_async(async_client: AsyncPinecone) -> None:
    """Query with metadata filters ($eq, $in) returns only matching vectors (REST async)."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Populate host cache so pc.index(name=...) can resolve it
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert vectors with metadata
        await idx.upsert(
            vectors=[
                {"id": "mf-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy", "year": 2020}},
                {"id": "mf-v2", "values": [0.3, 0.4], "metadata": {"genre": "action", "year": 2021}},
                {"id": "mf-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy", "year": 2022}},
            ]
        )

        # Wait for all 3 vectors to be queryable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 vectors queryable before filter tests",
        )

        # Test $eq filter: only comedy vectors should return
        comedy_result = await idx.query(
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
        action_result = await idx.query(
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
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# sparse-vectors — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_sparse_vectors_rest_async(async_client: AsyncPinecone) -> None:
    """Upsert hybrid (dense+sparse) vectors; fetch returns sparse_values; query with sparse_vector works (REST async)."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=4,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )

        # Populate host cache so pc.index(name=...) can resolve it
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert vectors with both dense values and sparse values
        upsert_resp = await idx.upsert(
            vectors=[
                {
                    "id": "sv-v1",
                    "values": [0.1, 0.2, 0.3, 0.4],
                    "sparse_values": {"indices": [0, 5], "values": [0.5, 0.8]},
                },
                {
                    "id": "sv-v2",
                    "values": [0.5, 0.6, 0.7, 0.8],
                    "sparse_values": {"indices": [2, 7], "values": [0.3, 0.9]},
                },
            ]
        )
        assert isinstance(upsert_resp, UpsertResponse)
        assert upsert_resp.upserted_count == 2

        # Wait for vectors to be fetchable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["sv-v1", "sv-v2"]),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="sparse vectors fetchable (async)",
        )

        # Fetch and verify sparse_values are returned
        fetch_resp = await idx.fetch(ids=["sv-v1", "sv-v2"])
        assert isinstance(fetch_resp, FetchResponse)
        assert "sv-v1" in fetch_resp.vectors
        v1 = fetch_resp.vectors["sv-v1"]
        assert isinstance(v1, Vector)
        assert v1.sparse_values is not None
        assert isinstance(v1.sparse_values, SparseValues)
        assert v1.sparse_values.indices == [0, 5]
        assert len(v1.sparse_values.values) == 2
        assert abs(v1.sparse_values.values[0] - 0.5) < 1e-4
        assert abs(v1.sparse_values.values[1] - 0.8) < 1e-4

        # Also verify v2 has sparse values
        v2 = fetch_resp.vectors["sv-v2"]
        assert v2.sparse_values is not None
        assert isinstance(v2.sparse_values, SparseValues)
        assert v2.sparse_values.indices == [2, 7]

        # Query with a sparse vector and verify matches are returned
        query_resp = await idx.query(
            vector=[0.1, 0.2, 0.3, 0.4],
            sparse_vector={"indices": [0, 5], "values": [0.5, 0.8]},
            top_k=5,
            include_values=True,
        )
        assert isinstance(query_resp, QueryResponse)
        assert len(query_resp.matches) >= 1
        match_ids = {m.id for m in query_resp.matches}
        assert "sv-v1" in match_ids
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )
