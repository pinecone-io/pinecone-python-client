"""Integration tests for extended data-plane operations (async REST).

Phase 2 Tier 1: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
include-values-metadata, query-namespaces.
"""
# area tags covered: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
# include-values-metadata

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


# ---------------------------------------------------------------------------
# query-by-id — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_by_id_rest_async(async_client: AsyncPinecone) -> None:
    """Query by stored vector ID returns a QueryResponse with same structure as query-by-vector (REST async)."""
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

        # Upsert 3 vectors
        await idx.upsert(
            vectors=[
                {"id": "qbi-v1", "values": [0.1, 0.2]},
                {"id": "qbi-v2", "values": [0.3, 0.4]},
                {"id": "qbi-v3", "values": [0.9, 0.1]},
            ]
        )

        # Wait for all 3 vectors to be queryable
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 vectors queryable (async) before query-by-id",
        )

        # Query by ID
        result = await idx.query(id="qbi-v1", top_k=3)
        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1

        for match in result.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        match_ids = [m.id for m in result.matches]
        assert "qbi-v1" in match_ids
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# fetch-missing-ids — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_missing_ids_rest_async(async_client: AsyncPinecone) -> None:
    """fetch() with a mix of existing and non-existent IDs returns only existing vectors, no error (REST async)."""
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

        # Populate host cache
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert 2 known vectors
        await idx.upsert(
            vectors=[
                {"id": "fmi-v1", "values": [0.1, 0.2]},
                {"id": "fmi-v2", "values": [0.3, 0.4]},
            ]
        )

        # Wait for both vectors to be fetchable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.fetch(ids=["fmi-v1", "fmi-v2"]),
            check_fn=lambda r: len(r.vectors) == 2,
            timeout=120,
            description="both vectors fetchable (async) before missing-id test",
        )

        # Fetch with a mix of existing and non-existent IDs — no error expected
        fetch_resp = await idx.fetch(ids=["fmi-v1", "fmi-v2", "fmi-does-not-exist"])
        assert isinstance(fetch_resp, FetchResponse)

        # Only existing vectors are returned
        assert "fmi-v1" in fetch_resp.vectors
        assert "fmi-v2" in fetch_resp.vectors
        assert "fmi-does-not-exist" not in fetch_resp.vectors
        assert len(fetch_resp.vectors) == 2

        # Each returned vector has correct structure
        v1 = fetch_resp.vectors["fmi-v1"]
        assert isinstance(v1, Vector)
        assert v1.id == "fmi-v1"
        assert len(v1.values) == 2
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# include-values-metadata — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_include_values_metadata_rest_async(async_client: AsyncPinecone) -> None:
    """Query with include_values=True/include_metadata=True returns values and metadata on matches;
    query with defaults returns empty values and None metadata (REST async)."""
    name = unique_name("idx")
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=3,
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
                {"id": "ivm-v1", "values": [0.1, 0.2, 0.3], "metadata": {"color": "red", "rank": 1}},
                {"id": "ivm-v2", "values": [0.4, 0.5, 0.6], "metadata": {"color": "blue", "rank": 2}},
            ]
        )

        # Wait for both vectors to be queryable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2, 0.3], top_k=10),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="both vectors queryable (async) before include-values-metadata test",
        )

        # Query with include_values=True, include_metadata=True
        result = await idx.query(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
            include_values=True,
            include_metadata=True,
        )
        assert isinstance(result, QueryResponse)
        assert len(result.matches) >= 1

        top_match = result.matches[0]
        assert isinstance(top_match, ScoredVector)
        # values should be a non-empty list of floats
        assert isinstance(top_match.values, list)
        assert len(top_match.values) == 3
        assert all(isinstance(v, float) for v in top_match.values)
        # metadata should be a dict
        assert isinstance(top_match.metadata, dict)
        assert "color" in top_match.metadata
        assert "rank" in top_match.metadata

        # Verify all matches have values and metadata
        for match in result.matches:
            assert len(match.values) == 3
            assert match.metadata is not None

        # Query with defaults (no include_values, no include_metadata)
        default_result = await idx.query(vector=[0.1, 0.2, 0.3], top_k=5)
        assert isinstance(default_result, QueryResponse)
        assert len(default_result.matches) >= 1

        default_match = default_result.matches[0]
        assert isinstance(default_match, ScoredVector)
        # Default: values should be empty, metadata should be None
        assert default_match.values == []
        assert default_match.metadata is None
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )
