"""Integration tests for extended data-plane operations (async REST).

Phase 2 Tier 1: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
include-values-metadata, query-namespaces.
"""
# area tags covered: metadata-filter, sparse-vectors, query-by-id, fetch-missing-ids,
# include-values-metadata, query-namespaces

from __future__ import annotations

import pytest

from pinecone import AsyncIndex, AsyncPinecone, Field
from pinecone.models.indexes.specs import ServerlessSpec
from pinecone.models.vectors.query_aggregator import QueryNamespacesResults
from pinecone.models.vectors.responses import (
    FetchByMetadataResponse,
    FetchResponse,
    QueryResponse,
    UpdateResponse,
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


# ---------------------------------------------------------------------------
# query-namespaces — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_namespaces_rest_async(async_client: AsyncPinecone) -> None:
    """query_namespaces() fans out queries across multiple namespaces and merges results (REST async)."""
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

        # Upsert different vectors into two named namespaces
        await idx.upsert(
            vectors=[
                {"id": "qn-ns1-v1", "values": [0.1, 0.2]},
                {"id": "qn-ns1-v2", "values": [0.3, 0.4]},
            ],
            namespace="qn-ns1",
        )
        await idx.upsert(
            vectors=[
                {"id": "qn-ns2-v1", "values": [0.5, 0.6]},
                {"id": "qn-ns2-v2", "values": [0.7, 0.8]},
            ],
            namespace="qn-ns2",
        )

        # Wait until at least one vector from each namespace is queryable
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace="qn-ns1"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns1 vectors queryable (async) before query_namespaces",
        )
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10, namespace="qn-ns2"),
            check_fn=lambda r: len(r.matches) >= 2,
            timeout=120,
            description="ns2 vectors queryable (async) before query_namespaces",
        )

        # Call query_namespaces across both namespaces
        results = await idx.query_namespaces(
            vector=[0.1, 0.2],
            namespaces=["qn-ns1", "qn-ns2"],
            metric="cosine",
            top_k=5,
        )

        # Verify result type and structure
        assert isinstance(results, QueryNamespacesResults)

        # Merged matches list: sorted by score (descending for cosine), up to top_k
        assert isinstance(results.matches, list)
        assert len(results.matches) >= 1

        # Each match is a ScoredVector with id and score
        for match in results.matches:
            assert isinstance(match, ScoredVector)
            assert isinstance(match.id, str)
            assert isinstance(match.score, float)

        # Matches come from both namespaces
        match_ids = {m.id for m in results.matches}
        assert len(match_ids & {"qn-ns1-v1", "qn-ns1-v2"}) >= 1
        assert len(match_ids & {"qn-ns2-v1", "qn-ns2-v2"}) >= 1

        # Scores should be in descending order (cosine: higher is better)
        scores = [m.score for m in results.matches]
        assert scores == sorted(scores, reverse=True)

        # usage has read_units (sum across namespaces)
        assert results.usage is not None
        assert isinstance(results.usage.read_units, int)
        assert results.usage.read_units >= 2  # at least 1 per namespace

        # ns_usage has per-namespace usage
        assert isinstance(results.ns_usage, dict)
        assert "qn-ns1" in results.ns_usage
        assert "qn-ns2" in results.ns_usage
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# metadata-filter numeric comparisons ($gt, $gte, $lt, $lte, $ne) — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_filter_numeric_operators_rest_async(async_client: AsyncPinecone) -> None:
    """Numeric comparison operators ($gt, $gte, $lt, $lte, $ne) filter vectors correctly (REST async).

    Verifies unified-filter-0001: Can build metadata filters using not-equal,
    greater-than, greater-than-or-equal, less-than, and less-than-or-equal operators.
    """
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
        idx = async_client.index(name=name)

        # Upsert 4 vectors with distinct integer year metadata values
        await idx.upsert(
            vectors=[
                {"id": "nf-v1", "values": [0.1, 0.2], "metadata": {"year": 2019}},
                {"id": "nf-v2", "values": [0.3, 0.4], "metadata": {"year": 2020}},
                {"id": "nf-v3", "values": [0.5, 0.6], "metadata": {"year": 2021}},
                {"id": "nf-v4", "values": [0.7, 0.8], "metadata": {"year": 2022}},
            ]
        )

        # Wait until all 4 vectors are visible
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.5, 0.5], top_k=10),
            check_fn=lambda r: len(r.matches) >= 4,
            timeout=120,
            description="all 4 vectors queryable before numeric filter tests (async)",
        )

        # $gt: 2020 — only years strictly greater than 2020 (2021, 2022)
        gt_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$gt": 2020}},
            include_metadata=True,
        )
        assert isinstance(gt_result, QueryResponse)
        gt_ids = {m.id for m in gt_result.matches}
        assert "nf-v3" in gt_ids
        assert "nf-v4" in gt_ids
        assert "nf-v1" not in gt_ids
        assert "nf-v2" not in gt_ids

        # $gte: 2020 — years >= 2020 (2020, 2021, 2022)
        gte_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$gte": 2020}},
            include_metadata=True,
        )
        assert isinstance(gte_result, QueryResponse)
        gte_ids = {m.id for m in gte_result.matches}
        assert "nf-v2" in gte_ids
        assert "nf-v3" in gte_ids
        assert "nf-v4" in gte_ids
        assert "nf-v1" not in gte_ids

        # $lt: 2021 — years strictly less than 2021 (2019, 2020)
        lt_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$lt": 2021}},
            include_metadata=True,
        )
        assert isinstance(lt_result, QueryResponse)
        lt_ids = {m.id for m in lt_result.matches}
        assert "nf-v1" in lt_ids
        assert "nf-v2" in lt_ids
        assert "nf-v3" not in lt_ids
        assert "nf-v4" not in lt_ids

        # $lte: 2021 — years <= 2021 (2019, 2020, 2021)
        lte_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$lte": 2021}},
            include_metadata=True,
        )
        assert isinstance(lte_result, QueryResponse)
        lte_ids = {m.id for m in lte_result.matches}
        assert "nf-v1" in lte_ids
        assert "nf-v2" in lte_ids
        assert "nf-v3" in lte_ids
        assert "nf-v4" not in lte_ids

        # $ne: 2021 — years not equal to 2021 (2019, 2020, 2022)
        ne_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter={"year": {"$ne": 2021}},
            include_metadata=True,
        )
        assert isinstance(ne_result, QueryResponse)
        ne_ids = {m.id for m in ne_result.matches}
        assert "nf-v1" in ne_ids
        assert "nf-v2" in ne_ids
        assert "nf-v4" in ne_ids
        assert "nf-v3" not in ne_ids
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# metadata-filter logical operators ($nin, &, |) via Field builder — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_filter_logical_operators_rest_async(async_client: AsyncPinecone) -> None:
    """$nin, logical AND (&), and logical OR (|) via the Field builder filter correctly (REST async).

    Verifies:
      unified-filter-0002 — not-in-list ($nin) operator
      unified-filter-0004 — combining filters with & (AND) and | (OR)
    """
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

        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # Upsert 4 vectors with genre + year metadata
        await idx.upsert(
            vectors=[
                {"id": "lo-v1", "values": [0.1, 0.2], "metadata": {"genre": "comedy", "year": 2020}},
                {"id": "lo-v2", "values": [0.3, 0.4], "metadata": {"genre": "action", "year": 2021}},
                {"id": "lo-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy", "year": 2022}},
                {"id": "lo-v4", "values": [0.7, 0.8], "metadata": {"genre": "horror", "year": 2021}},
            ]
        )

        # Wait until all 4 vectors are visible
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.5, 0.5], top_k=10),
            check_fn=lambda r: len(r.matches) >= 4,
            timeout=120,
            description="all 4 vectors queryable before logical filter tests (async)",
        )

        # $nin: genres NOT in ["horror", "action"] → comedy vectors only (v1, v3)
        nin_filter = Field("genre").not_in(["horror", "action"])
        nin_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=nin_filter.to_dict(),
            include_metadata=True,
        )
        assert isinstance(nin_result, QueryResponse)
        nin_ids = {m.id for m in nin_result.matches}
        assert "lo-v1" in nin_ids
        assert "lo-v3" in nin_ids
        assert "lo-v2" not in nin_ids
        assert "lo-v4" not in nin_ids

        # & (AND): genre == comedy AND year >= 2021 → v3 only (comedy + 2022)
        and_filter = (Field("genre") == "comedy") & Field("year").gte(2021)
        and_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=and_filter.to_dict(),
            include_metadata=True,
        )
        assert isinstance(and_result, QueryResponse)
        and_ids = {m.id for m in and_result.matches}
        assert "lo-v3" in and_ids
        assert "lo-v1" not in and_ids
        assert "lo-v2" not in and_ids
        assert "lo-v4" not in and_ids

        # | (OR): genre == horror OR genre == action → v2 and v4
        or_filter = (Field("genre") == "horror") | (Field("genre") == "action")
        or_result = await idx.query(
            vector=[0.5, 0.5],
            top_k=10,
            filter=or_filter.to_dict(),
            include_metadata=True,
        )
        assert isinstance(or_result, QueryResponse)
        or_ids = {m.id for m in or_result.matches}
        assert "lo-v2" in or_ids
        assert "lo-v4" in or_ids
        assert "lo-v1" not in or_ids
        assert "lo-v3" not in or_ids
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# fetch-by-metadata — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_by_metadata_rest_async(async_client: AsyncPinecone) -> None:
    """fetch_by_metadata() returns vectors matching a filter, with correct response shape (REST async).

    Verifies:
    - unified-vec-0010: Can fetch vectors by metadata filter from a namespace.
    - unified-vec-0024: No pagination token on a single-page result.
    """
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
        idx = async_client.index(name=name)

        # Upsert vectors with metadata; only v1 and v3 have genre=comedy
        await idx.upsert(
            vectors=[
                {"id": "fm-v1", "values": [0.1, 0.2, 0.3], "metadata": {"genre": "comedy", "year": 2020}},
                {"id": "fm-v2", "values": [0.4, 0.5, 0.6], "metadata": {"genre": "action", "year": 2021}},
                {"id": "fm-v3", "values": [0.7, 0.8, 0.9], "metadata": {"genre": "comedy", "year": 2022}},
            ]
        )

        # Wait until all 3 vectors are indexed (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.describe_index_stats(),
            check_fn=lambda r: r.total_vector_count >= 3,
            timeout=120,
            description="all 3 vectors indexed",
        )

        # Fetch only comedy vectors
        response = await idx.fetch_by_metadata(filter={"genre": {"$eq": "comedy"}})

        # Verify response type and shape
        assert isinstance(response, FetchByMetadataResponse)
        assert isinstance(response.vectors, dict)
        assert isinstance(response.namespace, str)

        # Only comedy vectors should be returned
        assert "fm-v1" in response.vectors
        assert "fm-v3" in response.vectors
        assert "fm-v2" not in response.vectors

        # Single page of 2 results — no pagination token
        assert response.pagination is None or response.pagination.next is None

        # Each returned vector has id and values
        for vid, vec in response.vectors.items():
            assert isinstance(vid, str)
            assert vec.id == vid
            assert isinstance(vec.values, list)
            assert len(vec.values) == 3
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# metadata-filter $exists operator — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_filter_exists_operator_rest_async(async_client: AsyncPinecone) -> None:
    """Field.exists() filter ($exists: True) returns only vectors that have the field (REST async).

    Verifies unified-filter-0003: Can build metadata filters using a field-exists operator.

    Upserts 3 vectors: two with a "premium" field (True/False) and one without.
    Queries with Field("premium").exists() and asserts only the two vectors that
    carry the "premium" key are returned — the third (which lacks the key) is excluded.
    """
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

        # Populate host cache
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        # v1 and v2 carry "premium" field; v3 does not
        await idx.upsert(
            vectors=[
                {"id": "ex-v1", "values": [0.1, 0.2, 0.3], "metadata": {"category": "A", "premium": True}},
                {"id": "ex-v2", "values": [0.4, 0.5, 0.6], "metadata": {"category": "B", "premium": False}},
                {"id": "ex-v3", "values": [0.7, 0.8, 0.9], "metadata": {"category": "C"}},
            ]
        )

        # Wait until all 3 vectors are queryable (eventual consistency)
        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2, 0.3], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 vectors queryable before exists filter test (async)",
        )

        # Query using Field.exists() — only vectors with "premium" field should match
        result = await idx.query(
            vector=[0.1, 0.2, 0.3],
            top_k=10,
            filter=Field("premium").exists().to_dict(),
            include_metadata=True,
        )

        assert isinstance(result, QueryResponse)
        matched_ids = {m.id for m in result.matches}

        # v1 and v2 both have "premium" — must appear
        assert "ex-v1" in matched_ids, f"Expected ex-v1 in matches (async), got {matched_ids}"
        assert "ex-v2" in matched_ids, f"Expected ex-v2 in matches (async), got {matched_ids}"

        # v3 has no "premium" field — must not appear
        assert "ex-v3" not in matched_ids, f"Expected ex-v3 excluded (async), got {matched_ids}"

        # Metadata is returned and each match has the "premium" key
        for match in result.matches:
            assert isinstance(match.metadata, dict)
            assert "premium" in match.metadata, (
                f"Match {match.id!r} missing 'premium' key in metadata (async): {match.metadata}"
            )
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# fetch_by_metadata multi-page pagination — REST async
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_by_metadata_pagination_rest_async(async_client: AsyncPinecone) -> None:
    """fetch_by_metadata() with limit and pagination_token iterates across pages (REST async).

    Upserts 5 vectors all with genre=scifi.  Calls fetch_by_metadata with
    limit=2 to force multiple pages.  Uses async_poll_until to wait until the
    paginated traversal covers all 5 vectors (the cursor-scan index can lag the
    count index after a recent upsert).  Verifies:
    - Each page returns at most 2 vectors (limit respected).
    - Pagination tokens are returned when more results exist.
    - No vector ID appears on more than one page (no cursor back-tracking).
    - All 5 genre=scifi vectors appear across the collected pages.
    - The final page has no pagination token (last-page sentinel).

    Verifies:
    - unified-vec-0010: Can fetch vectors by metadata filter with pagination.
    - unified-vec-0024: Fetch-by-metadata respects the limit parameter.
    """
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
        idx = async_client.index(name=name)

        # Upsert 5 vectors — all with genre=scifi so all match the filter
        target_ids = {"fbm-p1", "fbm-p2", "fbm-p3", "fbm-p4", "fbm-p5"}
        await idx.upsert(
            vectors=[
                {"id": "fbm-p1", "values": [0.1, 0.2, 0.3], "metadata": {"genre": "scifi"}},
                {"id": "fbm-p2", "values": [0.2, 0.3, 0.4], "metadata": {"genre": "scifi"}},
                {"id": "fbm-p3", "values": [0.3, 0.4, 0.5], "metadata": {"genre": "scifi"}},
                {"id": "fbm-p4", "values": [0.4, 0.5, 0.6], "metadata": {"genre": "scifi"}},
                {"id": "fbm-p5", "values": [0.5, 0.6, 0.7], "metadata": {"genre": "scifi"}},
            ]
        )

        async def _paginate_all() -> set[str]:
            """Collect all IDs returned by limit=2 pagination, asserting invariants."""
            seen: set[str] = set()
            tok: str | None = None
            page = 0
            while True:
                r = await idx.fetch_by_metadata(  # type: ignore[union-attr]
                    filter={"genre": {"$eq": "scifi"}},
                    limit=2,
                    pagination_token=tok,
                )
                assert isinstance(r, FetchByMetadataResponse)
                # Limit must be respected
                assert len(r.vectors) <= 2, (
                    f"Page {page} returned {len(r.vectors)} vectors (limit=2, async)"
                )
                # No duplicate vector IDs across pages
                for vid in r.vectors:
                    assert vid not in seen, f"Duplicate ID {vid!r} on page {page} (async)"
                    seen.add(vid)
                page += 1
                tok = r.pagination.next if r.pagination else None
                if not tok:
                    break
            return seen

        # Poll until the cursor-scan index is consistent with the upserted data.
        # The count index and cursor-scan index can temporarily diverge after a
        # very recent upsert; retrying the full paginated traversal is the
        # correct eventual-consistency guard here.
        collected_ids = await async_poll_until(
            query_fn=_paginate_all,
            check_fn=lambda ids: target_ids.issubset(ids),
            timeout=120,
            description="all 5 fbm-pagination vectors reachable via cursor scan (async)",
        )

        # All 5 target vectors must appear and no extra vectors
        assert collected_ids == target_ids, (
            f"Unexpected IDs in paginated result (async): {collected_ids - target_ids}"
        )
    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# query-filter-reflects-metadata-update — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_filter_reflects_metadata_update_rest_async(
    async_client: AsyncPinecone,
) -> None:
    """After updating a vector's metadata via update(id=...), a query with a
    metadata filter must reflect the new value — verifying the async REST path.

    Claim: unified-vec-0011 — Can update a single vector's metadata by
    identifier; the change is reflected in subsequent filtered queries.
    Area tag: update-and-query-sequence
    Transport: rest-async
    """
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
        await async_client.indexes.describe(name)
        idx = async_client.index(name=name)

        vectors = [
            {"id": "qmua-v1", "values": [0.1, 0.2], "metadata": {"genre": "drama"}},
            {"id": "qmua-v2", "values": [0.3, 0.4], "metadata": {"genre": "drama"}},
            {"id": "qmua-v3", "values": [0.5, 0.6], "metadata": {"genre": "comedy"}},
        ]
        upsert_resp = await idx.upsert(vectors=vectors)
        assert isinstance(upsert_resp, UpsertResponse)
        assert upsert_resp.upserted_count == 3

        await async_poll_until(
            query_fn=lambda: idx.query(vector=[0.1, 0.2], top_k=10),
            check_fn=lambda r: len(r.matches) >= 3,
            timeout=120,
            description="all 3 qmua vectors queryable (async)",
        )

        # Baseline: drama filter returns qmua-v1 and qmua-v2
        drama_baseline = await idx.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "drama"}},
            include_metadata=True,
        )
        assert isinstance(drama_baseline, QueryResponse)
        drama_ids_before = {m.id for m in drama_baseline.matches}
        assert "qmua-v1" in drama_ids_before
        assert "qmua-v2" in drama_ids_before
        assert "qmua-v3" not in drama_ids_before

        # Update qmua-v1: change genre from drama to comedy
        update_resp = await idx.update(
            id="qmua-v1",
            set_metadata={"genre": "comedy"},
        )
        assert isinstance(update_resp, UpdateResponse)

        # Poll until drama filter no longer returns qmua-v1
        async def _drama_updated() -> QueryResponse | None:
            r = await idx.query(
                vector=[0.1, 0.2],
                top_k=10,
                filter={"genre": {"$eq": "drama"}},
                include_metadata=True,
            )
            ids = {m.id for m in r.matches}
            if "qmua-v1" not in ids and "qmua-v2" in ids:
                return r
            return None

        updated_drama_resp = await async_poll_until(
            query_fn=_drama_updated,
            check_fn=lambda r: r is not None,
            timeout=180,
            description="drama filter reflects metadata update for qmua-v1 (async)",
        )
        assert updated_drama_resp is not None
        drama_ids_after = {m.id for m in updated_drama_resp.matches}
        assert "qmua-v2" in drama_ids_after
        assert "qmua-v1" not in drama_ids_after

        # qmua-v1 should now appear in comedy filter
        comedy_resp = await idx.query(
            vector=[0.1, 0.2],
            top_k=10,
            filter={"genre": {"$eq": "comedy"}},
            include_metadata=True,
        )
        assert isinstance(comedy_resp, QueryResponse)
        comedy_ids = {m.id for m in comedy_resp.matches}
        assert "qmua-v1" in comedy_ids, \
            f"qmua-v1 should match comedy filter after update; got: {comedy_ids}"
        assert "qmua-v3" in comedy_ids, \
            f"qmua-v3 (original comedy) should still match; got: {comedy_ids}"

        # Confirm metadata in result has updated value
        v1_match = next((m for m in comedy_resp.matches if m.id == "qmua-v1"), None)
        assert v1_match is not None
        assert v1_match.metadata is not None
        assert v1_match.metadata.get("genre") == "comedy"

    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )


# ---------------------------------------------------------------------------
# unusual ASCII vector IDs — REST async
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_unusual_ascii_ids_round_trip_rest_async(async_client: AsyncPinecone) -> None:
    """Vectors with unusual but valid ASCII IDs survive upsert→fetch→query→list (REST async).

    Verifies unified-ids-0001: IDs containing spaces, slashes, dots, colons, and
    other printable ASCII punctuation are accepted end-to-end by the live API via
    the async client.
    """
    name = unique_name("idx")
    unusual_ids = [
        "id/with/slashes",
        "id.with.dots",
        "id:colon:separated",
        "id@special!chars",
        "id[brackets]",
    ]
    idx: AsyncIndex | None = None
    try:
        await async_client.indexes.create(
            name=name,
            dimension=3,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            timeout=300,
        )
        idx = async_client.index(name=name)

        # Upsert vectors with unusual IDs
        vectors = [(vid, [float(i + 1) * 0.1, float(i + 2) * 0.1, float(i + 3) * 0.1])
                   for i, vid in enumerate(unusual_ids)]
        upsert_resp = await idx.upsert(vectors=vectors)
        assert isinstance(upsert_resp, UpsertResponse)
        assert upsert_resp.upserted_count == len(unusual_ids)

        # Wait until all vectors are fetchable (eventual consistency)
        fetch_resp = await async_poll_until(
            query_fn=lambda: idx.fetch(ids=unusual_ids),
            check_fn=lambda r: len(r.vectors) == len(unusual_ids),
            timeout=120,
            description="all unusual-id vectors fetchable (async)",
        )
        assert isinstance(fetch_resp, FetchResponse)
        for vid in unusual_ids:
            assert vid in fetch_resp.vectors, f"ID {vid!r} missing from fetch response (async)"
            assert fetch_resp.vectors[vid].id == vid

        # Query by ID — queried vector must appear somewhere in results
        first_id = unusual_ids[0]
        query_resp = await async_poll_until(
            query_fn=lambda: idx.query(id=first_id, top_k=5),
            check_fn=lambda r: any(m.id == first_id for m in r.matches),
            timeout=60,
            description=f"query-by-id returns {first_id!r} (async)",
        )
        assert isinstance(query_resp, QueryResponse)
        match_ids = {m.id for m in query_resp.matches}
        assert first_id in match_ids, \
            f"queried ID {first_id!r} should appear in its own query results (async); got {match_ids}"
        for m in query_resp.matches:
            assert isinstance(m, ScoredVector)
            assert isinstance(m.id, str)

        # List — all unusual IDs must appear across pages
        all_listed_ids: set[str] = set()
        async for page in idx.list():
            for item in page.vectors:
                all_listed_ids.add(item.id)
        for vid in unusual_ids:
            assert vid in all_listed_ids, f"ID {vid!r} missing from list() output (async)"

    finally:
        if idx is not None:
            await idx.close()
        await async_cleanup_resource(
            lambda: async_client.indexes.delete(name),
            name,
            "index",
        )
